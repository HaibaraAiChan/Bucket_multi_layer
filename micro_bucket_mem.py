import sys
sys.path.insert(0,'..')
# sys.path.insert(0,'../..')
import dgl
from dgl.data.utils import save_graphs
import numpy as np
from statistics import mean
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from block_dataloader import generate_dataloader_block

import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm

import random
# from graphsage_model_products_mem import GraphSAGE
from graphsage_model_bucket import GraphSAGE
import dgl.function as fn
from load_graph import load_reddit, inductive_split, load_ogb, load_cora, load_karate, prepare_data, load_pubmed
# from load_graph import load_ogbn_mag    ###### TODO
from load_graph import load_ogbn_dataset
from memory_usage import see_memory_usage, nvidia_smi_usage
import tracemalloc
from cpu_mem_usage import get_memory
from statistics import mean


import matplotlib.pyplot as plt

import pickle
from utils import Logger
import os 
import numpy

import math
from collections import Counter, OrderedDict
class OrderedCounter(Counter, OrderedDict):
	'Counter that remembers the order elements are first encountered'

	def __repr__(self):
		return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

	def __reduce__(self):
		return self.__class__, (OrderedDict(self),)






def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.device >= 0:
		torch.cuda.manual_seed_all(args.seed)
		torch.cuda.manual_seed(args.seed)
		torch.backends.cudnn.enabled = False
		torch.backends.cudnn.deterministic = True
		dgl.seed(args.seed)
		dgl.random.seed(args.seed)

def CPU_DELTA_TIME(tic, str1):
	toc = time.time()
	print(str1 + ' spend:  {:.6f}'.format(toc - tic))
	return toc


def compute_acc(pred, labels):
	"""
	Compute the accuracy of prediction given the labels.
	"""
	labels = labels.long()
	return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, nfeats, labels, train_nid, val_nid, test_nid, device, args):
	"""
	Evaluate the model on the validation set specified by ``val_nid``.
	g : The entire graph.
	inputs : The features of all the nodes.
	labels : The labels of all the nodes.
	val_nid : the node Ids for validation.
	device : The GPU device to evaluate on.
	"""
	# train_nid = train_nid.to(device)
	# val_nid=val_nid.to(device)
	# test_nid=test_nid.to(device)
	nfeats=nfeats.to(device)
	g=g.to(device)
	# print('device ', device)
	model.eval()
	with torch.no_grad():
		# pred = model(g=g, x=nfeats)
		pred = model.inference(g, nfeats,  args, device)
	model.train()
	
	train_acc= compute_acc(pred[train_nid], labels[train_nid].to(pred.device))
	val_acc=compute_acc(pred[val_nid], labels[val_nid].to(pred.device))
	test_acc=compute_acc(pred[test_nid], labels[test_nid].to(pred.device))
	return (train_acc, val_acc, test_acc)


def load_subtensor(nfeat, labels, seeds, input_nodes, device):
	"""
	Extracts features and labels for a subset of nodes
	"""
	batch_inputs = nfeat[input_nodes].to(device)
	batch_labels = labels[seeds].to(device)
	return batch_inputs, batch_labels

def load_block_subtensor(nfeat, labels, blocks, device,args):
	"""
	Extracts features and labels for a subset of nodes
	"""

	# if args.GPUmem:
	# 	see_memory_usage("----------------------------------------before batch input features to device")
	batch_inputs = nfeat[blocks[0].srcdata[dgl.NID]].to(device)
	# if args.GPUmem:
	# 	see_memory_usage("----------------------------------------after batch input features to device")
	batch_labels = labels[blocks[-1].dstdata[dgl.NID]].to(device)
	# if args.GPUmem:
	# 	see_memory_usage("----------------------------------------after  batch labels to device")
	return batch_inputs, batch_labels


def load_bucket_labels( bucket_output,batch_pred, batch_labels, device ):
	"""
	Extracts features and labels for a subset of nodes with some degrees
	# """
	bucket_labels = batch_labels[bucket_output.long()].to(device)
	bucket_pred = batch_pred[bucket_output.long()].to(device) # it should be local
	
	return bucket_pred, bucket_labels





def get_bucket_inputs(bucket_outputs, blocks,local_nid_2_global):
	eid_bkt_in = blocks[0].in_edges(bucket_outputs, form="all")[0] # full batch block: local eid

	c=OrderedCounter(eid_bkt_in.tolist())
	list(map(c.__delitem__, filter(c.__contains__, bucket_outputs.tolist())))
	r_=list(c.keys())
	bucket_inputs_local = torch.tensor(bucket_outputs.tolist() + r_, dtype=torch.long).tolist()
	# local to global
	bucket_inputs = list(map(local_nid_2_global.get, bucket_inputs_local))
	return bucket_inputs

def cal_bucket_loss(degs, bucket_loss, loss_sum, len_bucket_nid, len_dst_nid):
	ratio = len_bucket_nid/len_dst_nid
	bucket_loss = bucket_loss * ratio
	bucket_loss_item = bucket_loss.cpu().detach()
	print('degree: '+str(degs)+' # of output: '+str(len_bucket_nid)+' ratio: '+str(ratio) + ' bucket_loss : '+ str(bucket_loss_item))
	loss_sum.append(bucket_loss_item ) 
	return bucket_loss, loss_sum

def  group_degrees_buckets(degrees_group, num_split,step, parameters):
	sorted_val, idx,local_nid_2_global, blocks, model,batch_inputs, batch_labels, labels,  loss_fcn, device, args = parameters
	dst_nid = blocks[0].dstdata['_ID']
	loss_sum = []

	bucket_outputs = torch.tensor([],dtype=torch.int32).to(device)
	for deg in degrees_group:
		tmp = idx[ sorted_val == deg].int().to(device)
		bucket_outputs = torch.cat((bucket_outputs , tmp))
	# bucket_input = get_bucket_inputs(bucket_outputs, blocks,local_nid_2_global)
	degrees_group = degrees_group.to(torch.int32)
	
	see_memory_usage("------------------------------------ before batch_pred = model(blocks, batch_inputs, deg.to(device)) ")
	batch_pred = model(blocks, batch_inputs, degrees_group.to(device), num_split, step)
	see_memory_usage("------------------------------------- after batch_pred = model(blocks, batch_inputs, deg.to(device)) ")

	bucket_pred, bucket_labels = load_bucket_labels( bucket_outputs, batch_pred, batch_labels, device)
	bucket_loss = loss_fcn(bucket_pred, bucket_labels)
	bucket_loss, loss_sum = cal_bucket_loss(degrees_group, bucket_loss, loss_sum, len(bucket_outputs),len(dst_nid))
			
	bucket_loss.backward()
	see_memory_usage("------------------------------------- after loss backward ")

	return loss_sum, model
	

def run_split_degree_bucket(degree_split, num_split, step, parameters):
	sorted_val, idx,local_nid_2_global, blocks, model,batch_inputs, batch_labels, labels,  loss_fcn, device, args = parameters
	dst_nid = blocks[0].dstdata['_ID']
	loss_sum = []

	for step in range(num_split):
		block_outputs_local_idx = idx[ sorted_val == degree_split].int().to(device)
		N = math.ceil(len(block_outputs_local_idx)/num_split)
		step_bkt_idx = block_outputs_local_idx[step*N:((step+1)*N)]
		bucket_outputs_local = torch.index_select(blocks[-1].dstnodes(), 0, step_bkt_idx.long())
		bucket_outputs_local.to(torch.int32).squeeze()
		# print('main.py : output nodes local nid[-3:-1]: ', bucket_outputs_local[-3:-1])
		
		degree_split.to(device)
		see_memory_usage("----------------------------------- before batch_pred = model(blocks, batch_inputs, deg.to(device)) ")
		batch_pred = model(blocks, batch_inputs, degree_split, num_split, step)
		see_memory_usage("------------------------------------after batch_pred = model(blocks, batch_inputs, deg.to(device)) ")

		bucket_pred, bucket_labels = load_bucket_labels( bucket_outputs_local, batch_pred, batch_labels, device)
		bucket_loss = loss_fcn(bucket_pred, bucket_labels)
		bucket_loss, loss_sum = cal_bucket_loss(degree_split, bucket_loss, loss_sum, len(bucket_pred),len(dst_nid))
		bucket_loss.backward()
		see_memory_usage("------------------------------------ after loss backward ")
			
	return loss_sum, model




#### Entry point
def run(args, device, data):
	if args.GPUmem:
		see_memory_usage("----------------------------------------start of run function ")
	# Unpack data
	g, nfeats, labels, n_classes, train_nid, val_nid, test_nid = data
	in_feats = len(nfeats[0])
	print('in feats: ', in_feats)
	nvidia_smi_list=[]

	# if args.selection_method =='metis':
	# 	args.o_graph = dgl.node_subgraph(g, train_nid)

	sampler = dgl.dataloading.MultiLayerNeighborSampler(
		[int(fanout) for fanout in args.fan_out.split(',')])
	full_batch_size = len(train_nid)
	
	args.num_workers = 0
	full_batch_dataloader = dgl.dataloading.NodeDataLoader(
		g,
		train_nid,
		sampler,
		# device='cpu',
		batch_size=full_batch_size,
		drop_last=False,
		shuffle=False,
		num_workers=args.num_workers)
	# if args.GPUmem:
		# see_memory_usage("----------------------------------------before model to device ")
	model = GraphSAGE(
					in_feats,
					args.num_hidden,
					n_classes,
					args.aggre,
					args.num_layers,
					F.relu,
					args.dropout).to(device)
					
	loss_fcn = nn.CrossEntropyLoss()
	# if args.GPUmem:
	# 			see_memory_usage("----------------------------------------after model to device")
	logger = Logger(args.num_runs, args)
	dur = []
	time_block_gen=[]
	for run in range(args.num_runs):
		model.reset_parameters()
		# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
		for epoch in range(args.num_epochs):
			num_src_node =0
			num_out_node_FL=0
			gen_block=0
			tmp_t=0
			model.train()
			if epoch >= args.log_indent:
				t0 = time.time()
			loss_sum=0
			# start of data preprocessing part---s---------s--------s-------------s--------s------------s--------s----
			if args.load_full_batch:
				full_batch_dataloader=[]
				file_name=r'./../dataset/fan_out_'+args.fan_out+'/'+args.dataset+'_'+str(epoch)+'_items.pickle'
				with open(file_name, 'rb') as handle:
					item=pickle.load(handle)
					full_batch_dataloader.append(item)

			if args.GPUmem:
				see_memory_usage("----------------------------------------- after full batch dataloader loading ")
			loss_sum =[]
			num_split = -1
			step = -1
			for step_out, (input_nodes, seeds, blocks) in enumerate(full_batch_dataloader):
			
				batch_inputs, batch_labels = load_block_subtensor(nfeats, labels, blocks, device,args)#------------*
				if args.GPUmem:
					see_memory_usage("----------------------------------------- after block subtensor loading ")
				degrees = blocks[0].in_degrees() # local nid as index for degree
				# distribution =  dict(Counter(degrees.tolist())) #####
				# distribution = dict(OrderedDict(sorted(distribution.items())))####
				# print(distribution)####
				
				idx_dict = dict(zip(range(len(degrees)),degrees.tolist()))            ######
				sorted_res = dict(sorted(idx_dict.items(), key=lambda item: item[1])) ######
				sorted_val = torch.tensor(list(sorted_res.values()))                  ######
				idx = torch.tensor(list(sorted_res.keys()))                           ######
    
				# sorted_val, idx = torch.sort(degrees) # local nid index, 
				# we don't use torch.sort(), because it can't keep the original order of duplicated items 
				unique_degrees = torch.unique(sorted_val)
				dst_nid = blocks[0].dstdata['_ID']
				src_nid = blocks[0].srcdata['_ID']

				local_nid_2_global = dict(zip(range(len(src_nid)), src_nid.tolist()))
				if args.GPUmem:
					see_memory_usage("----------------------------------------- after schedule preparing ")
				blocks = [block.int().to(device) for block in blocks]#------------*
				# find out the degrees you need to group, single degree bucket or need to split
				# e.g. (1-9) degree group, degree 10 split into 4 parts
				if args.GPUmem:
					see_memory_usage("----------------------------------------- after block to device loading ")
				parameters = sorted_val, idx, local_nid_2_global, blocks,model, batch_inputs, batch_labels, labels, loss_fcn,device, args
				loss_sum = []
				degrees_group = unique_degrees[:-1] 
				##===----===---===-----=== single degree -----------
				# sub_sum_loss, model = run_degrees_bucket(single_degree, num_split, step, parameters)
				# loss_sum= loss_sum + sub_sum_loss 
				if args.GPUmem:
					see_memory_usage("----------------------------------------- before degree grouping ")
				##===----===---===-----=== degrees group -------------------------------
				group_sum_loss, model = group_degrees_buckets(degrees_group, num_split, step, parameters) # step is useless in grouping, just a point holder
				loss_sum= loss_sum + group_sum_loss 
				if args.GPUmem:
					see_memory_usage("----------------------------------------- after degree grouping ")
				##===----===---===-----=== degree split, here we use the last degree 10 as an example
				degree_split = unique_degrees[-1].to(torch.int64)       # it should be torch.tensor(xxx, dtype=torch.long)
				num_split = args.num_split_degree                       # degree need to split
				if (degree_split.dim() == 0) and (num_split >= 1):
					split_sum_loss, model = run_split_degree_bucket(degree_split, num_split, step, parameters)
					loss_sum = loss_sum + split_sum_loss 
				if args.GPUmem:
					see_memory_usage("----------------------------------------- after degree split ")
				print('-------------------------------------------------------------------------------loss_sum  : ', sum(loss_sum))
				optimizer.step()
				optimizer.zero_grad()
				if args.GPUmem:
					see_memory_usage("----------------------------------------- after optimizer.zero_grad() ")
				print()
    
		# 	if args.eval:
			
		# 		train_acc, val_acc, test_acc = evaluate(model, g, nfeats, labels, train_nid, val_nid, test_nid, device, args)

		# 		logger.add_result(run, (train_acc, val_acc, test_acc))
					
		# 		print("Run {:02d} | Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f}".format(run, epoch, loss_sum.item(), train_acc, val_acc, test_acc))
		# 	else:
		# 		print(' Run '+str(run)+'| Epoch '+ str( epoch)+' |')
			
		# if args.eval:
		# 	logger.print_statistics(run)

	

def main():
	# get_memory("-----------------------------------------main_start***************************")
	tt = time.time()
	print("main start at this time " + str(tt))
	argparser = argparse.ArgumentParser("multi-gpu training")
	argparser.add_argument('--device', type=int, default=0,
		help="GPU device ID. Use -1 for CPU training")
	argparser.add_argument('--seed', type=int, default=1236)
	argparser.add_argument('--setseed', type=bool, default=True)
	argparser.add_argument('--GPUmem', type=bool, default=True)
	argparser.add_argument('--load-full-batch', type=bool, default=True)
	# argparser.add_argument('--load-full-batch', type=bool, default=False)
	# argparser.add_argument('--root', type=str, default='../my_full_graph/')
	# argparser.add_argument('--dataset', type=str, default='ogbn-arxiv')
	# argparser.add_argument('--dataset', type=str, default='ogbn-mag')
	argparser.add_argument('--dataset', type=str, default='ogbn-products')
	# argparser.add_argument('--dataset', type=str, default='cora')
	# argparser.add_argument('--dataset', type=str, default='karate')
	# argparser.add_argument('--dataset', type=str, default='reddit')
	argparser.add_argument('--aggre', type=str, default='lstm')
	# argparser.add_argument('--aggre', type=str, default='mean')
	argparser.add_argument('--num-split-degree', type=float, default=32) # $$$$$$$$$$$$$$$
	# argparser.add_argument('--selection-method', type=str, default='range')
	# argparser.add_argument('--selection-method', type=str, default='random')
	# argparser.add_argument('--selection-method', type=str, default='metis')
	argparser.add_argument('--selection-method', type=str, default='REG')
	argparser.add_argument('--num-batch', type=int, default=1)
	argparser.add_argument('--batch-size', type=int, default=0)


	# argparser.add_argument('--balanced_init_ratio', type=float, default=0.2)
	argparser.add_argument('--num-runs', type=int, default=1)
	argparser.add_argument('--num-epochs', type=int, default=1)

	argparser.add_argument('--num-hidden', type=int, default=2)

	argparser.add_argument('--num-layers', type=int, default=1)
	# argparser.add_argument('--fan-out', type=str, default='2')
	argparser.add_argument('--fan-out', type=str, default='10')
	# argparser.add_argument('--num-layers', type=int, default=2)
	# argparser.add_argument('--fan-out', type=str, default='2,4')
	# argparser.add_argument('--fan-out', type=str, default='10,25')
	
	

	argparser.add_argument('--log-indent', type=float, default=0)
#--------------------------------------------------------------------------------------
	

	argparser.add_argument('--lr', type=float, default=1e-3)
	argparser.add_argument('--dropout', type=float, default=0.5)
	argparser.add_argument("--weight-decay", type=float, default=5e-4,
						help="Weight for L2 loss")
	argparser.add_argument("--eval", action='store_true', 
						help='If not set, we will only do the training part.')

	argparser.add_argument('--num-workers', type=int, default=4,
		help="Number of sampling processes. Use 0 for no extra process.")
	

	argparser.add_argument('--log-every', type=int, default=5)
	argparser.add_argument('--eval-every', type=int, default=5)
	
	args = argparser.parse_args()
	if args.setseed:
		set_seed(args)
	device = "cpu"
	if args.GPUmem:
		see_memory_usage("-----------------------------------------before load data ")
	if args.dataset=='karate':
		g, n_classes = load_karate()
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='cora':
		g, n_classes = load_cora()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='pubmed':
		g, n_classes = load_pubmed()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='reddit':
		g, n_classes = load_reddit()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
	elif args.dataset == 'ogbn-arxiv':
		data = load_ogbn_dataset(args.dataset,  args)
		device = "cuda:0"

	elif args.dataset=='ogbn-products':
		g, n_classes = load_ogb(args.dataset,args)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='ogbn-mag':
		# data = prepare_data_mag(device, args)
		data = load_ogbn_mag(args)
		device = "cuda:0"
		# run_mag(args, device, data)
		# return
	else:
		raise Exception('unknown dataset')
		
	
	best_test = run(args, device, data)
	

if __name__=='__main__':
	main()

