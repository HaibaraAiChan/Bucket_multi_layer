import numpy
import dgl
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../utils/')
from numpy.core.numeric import Infinity
import multiprocessing as mp
import torch
import time
from statistics import mean
from my_utils import *
import networkx as nx
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import cupy as cp

from collections import Counter
from math import ceil
from cpu_mem_usage import get_memory

def asnumpy(input):
	return input.cpu().detach().numpy()

def equal (x,y):
	return x == y

def nonzero_1d(input):
	x = torch.nonzero(input, as_tuple=False).squeeze()
	return x if x.dim() == 1 else x.view(-1)

def gather_row(data, row_index):
	return torch.index_select(data, 0, row_index.long())

def zerocopy_from_numpy(np_array):
	return torch.as_tensor(np_array)


class Bucket_Partitioner:  # ----------------------*** split the output layer block ***---------------------
	def __init__(self, layer_block, args):
		# self.balanced_init_ratio=args.balanced_init_ratio
		self.dataset=args.dataset
		self.layer_block=layer_block # local graph with global nodes indices
		self.local=False
		self.output_nids=layer_block.dstdata['_ID'] # tensor type
		self.local_output_nids=[]
		self.local_src_nids=[]
		self.src_nids_tensor= layer_block.srcdata['_ID']
		self.src_nids_list= layer_block.srcdata['_ID'].tolist()
		self.full_src_len=len(layer_block.srcdata['_ID'])
		self.global_batched_seeds_list=[]
		self.local_batched_seeds_list=[]
		self.weights_list=[]
		# self.alpha=args.alpha
		# self.walkterm=args.walkterm
		self.num_batch=args.num_batch
		self.selection_method=args.selection_method
		self.batch_size=0
		self.ideal_partition_size=0

		# self.bit_dict={}
		self.side=0
		self.partition_nodes_list=[]
		self.partition_len_list=[]

		self.time_dict={}
		self.red_before=[]
		self.red_after=[]
		self.args=args


		self.in_degrees = self.layer_block.in_degrees()

	def my_sort_1d(val):  # add new function here, to replace torch.sort()
		idx_dict = dict(zip(range(len(val)),val.tolist())) #####
		sorted_res = dict(sorted(idx_dict.items(), key=lambda item: item[1])) ######
		sorted_val = torch.tensor(list(sorted_res.values())).to(val.device)  ######
		idx = torch.tensor(list(sorted_res.keys())).to(val.device) ######
		return sorted_val, idx

	def _bucketing(self, val):
		degs = self.layer_block.in_degrees() # local nid index
		sorted_val, idx = torch.sort(degs)
		unique_val = asnumpy(torch.unique(sorted_val))
		bkt_idx = []
		for v in unique_val:
			eqidx = nonzero_1d(equal(sorted_val, v))
			bkt_idx.append(gather_row(idx, eqidx))
		def bucketor(data):
			bkts = [gather_row(data, idx) for idx in bkt_idx]
			return bkts
		return unique_val, bucketor

	def get_in_degree_bucketing(self):
		num_fanout_degree_split = self.args.num_split_degree
		degs = self.layer_block.in_degrees()
		print('degs', degs)
		nodes = self.layer_block.dstnodes()
		dst_nid =self.layer_block.dstdata['_ID']
		
		# degree bucketing
		unique_degs, bucketor = self._bucketing(degs)
		bkt_nodes = []
		for deg, node_bkt in zip(unique_degs, bucketor(nodes)):
			if deg == 0:
				# skip reduce function for zero-degree nodes
				continue
			bkt_nodes.append(dst_nid[node_bkt]) # local nid idx->global

		return bkt_nodes  # global nid



	def get_src(self, seeds):
		in_ids=list(self.layer_block.in_edges(seeds))[0].tolist()
		src= list(set(in_ids+seeds))
		return src



	def gen_batches_seeds_list(self, bkt_dst_nodes_list):

		if "bucketing" in self.selection_method :
			fanout_dst_nids = bkt_dst_nodes_list[-1]
			if self.args.num_batch ==  1:
				print('no need to split fanout degree, full batch train ')
				return
			if self.args.num_batch > 1:
				fanout_batch_size = ceil(len(fanout_dst_nids)/(self.args.num_batch-1))
				# args.batch_size = batch_size
			if 'random' in self.selection_method:
				# print('before  shuffle ', fanout_dst_nids)
				indices = torch.randperm(len(fanout_dst_nids))
				map_output_list = fanout_dst_nids.view(-1)[indices].view(fanout_dst_nids.size())
				# print('after shuffle ', map_output_list)

				batches_nid_list = [map_output_list[i:i + fanout_batch_size] for i in range(0, len(map_output_list), fanout_batch_size)]
				group_nids_list = bkt_dst_nodes_list[:-1]
				if len(group_nids_list) == 1 :
					batches_nid_list.insert(0, group_nids_list[0])
				else:
					batches_nid_list.insert(0, torch.cat(group_nids_list))
				length = len(self.output_nids)
				weights_list = [len(batch_nids)/length  for batch_nids in batches_nid_list]
				print('batches_nid_list ', batches_nid_list)
				print('weights_list ', weights_list)
				

				self.local_batched_seeds_list = batches_nid_list
		return


	def get_src_len(self,seeds):
		in_nids=self.layer_block.in_edges(seeds)[0]
		src =torch.unique(in_nids)
		return src.size()



	def get_partition_src_len_list(self):
		partition_src_len_list=[]
		for seeds_nids in self.local_batched_seeds_list:
			partition_src_len_list.append(self.get_src_len(seeds_nids))

		self.partition_src_len_list=partition_src_len_list
		return partition_src_len_list


	def buckets_partition(self):

		# self.ideal_partition_size = (self.full_src_len/self.num_batch)
		bkt_dst_nodes_list = self.get_in_degree_bucketing()
		t2 = time.time()

		self.gen_batches_seeds_list(bkt_dst_nodes_list)
		# print('total k batches seeds list generation spend ', time.time()-t2 )

		weight_list = get_weight_list(self.local_batched_seeds_list)
		src_len_list = self.get_partition_src_len_list()

		# print('after graph partition')

		self.weights_list = weight_list
		self.partition_len_list = src_len_list

		return self.local_batched_seeds_list, weight_list, src_len_list



	def global_to_local(self):

		sub_in_nids = self.src_nids_list

		global_nid_2_local = dict(zip(sub_in_nids,range(len(sub_in_nids))))
		self.local_output_nids = list(map(global_nid_2_local.get, self.output_nids.tolist()))

		self.local_src_nids = list(map(global_nid_2_local.get, self.src_nids_list))
		# print('self.local_src_nids', self.local_src_nids)
		self.local=True

		return


	def local_to_global(self):
		
		idx = torch.arange(0,len(self.src_nids_tensor))
		global_batched_seeds_list = []
		for local_in_nids in self.local_batched_seeds_list:
			global_batched_seeds_list.append(gather_row(idx, local_in_nids))

		self.global_batched_seeds_list=global_batched_seeds_list

		self.local=False

		return


	def init_partition(self):
		ts = time.time()

		self.global_to_local() # global to local            self.local_batched_seeds_list
		
		t2=time.time()
		# Then, the graph_parition is run in block to graph local nids,it has no relationship with raw graph
		self.buckets_partition()

		# after that, we transfer the nids of batched output nodes from local to global.
		self.local_to_global() # local to global         self.global_batched_seeds_list
		t_total=time.time()-ts

		return self.global_batched_seeds_list, self.weights_list, t_total, self.partition_len_list
