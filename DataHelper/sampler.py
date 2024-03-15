import torch
from dgl import function as fn, edge_subgraph
import numpy as np
import tqdm
from dgl.utils import check_eq_shape, expand_as_pair
from dgl.dataloading import (
	DataLoader,
	MultiLayerFullNeighborSampler,
	NeighborSampler,
	BlockSampler
)
from dgl.base import EID, NID
from dgl.transforms import to_block
from dgl.sampling.neighbor import sample_neighbors_biased, sample_neighbors
import random
from tqdm import tqdm

CLASS = [0,1,2]

class LWNeighborSampler(BlockSampler):
	# label aware neighbor sampler
    '''
    3 layers , 3 relations, 5 neighbors sampled in each layers
    fanouts: [{('review', 'net_rsr', 'review'): 5, ('review', 'net_rtr', 'review'): 5, ('review', 'net_rur', 'review'): 5},  一个block
              {('review', 'net_rsr', 'review'): 5, ('review', 'net_rtr', 'review'): 5, ('review', 'net_rur', 'review'): 5}, 
              {('review', 'net_rsr', 'review'): 5, ('review', 'net_rtr', 'review'): 5, ('review', 'net_rur', 'review'): 5}]
    '''
    def __init__(self, fanouts, num_layers, sampling_type, train_fr_nodes, prefetch_node_feats=None, prefetch_labels=None, prefetch_edge_feats=None, output_device=None):
        super().__init__(prefetch_node_feats, prefetch_labels, prefetch_edge_feats, output_device)
        self.num_layers = num_layers
        self.fanouts = fanouts
        self.sampling_type = sampling_type
        self.train_fr_nodes = train_fr_nodes

    def sample_frontier(self, block_id, g, seed_nodes, *args, **kwargs):  
        with g.local_scope():
            new_edges_masks = {}
            for etype in g.canonical_etypes: 
                edge_mask = torch.zeros(g.num_edges(etype))
                for node in seed_nodes:
                    if self.sampling_type == 'in':
                        nei, _, eid  = g.in_edges(node, form="all", etype=etype)  # 一阶邻居 idx
                    elif self.sampling_type == 'out':
                        _, nei, eid  = g.out_edges(node, form="all", etype=etype) 
                    else:
                        in_nei, _, in_eid  = g.in_edges(node, form="all", etype=etype)
                        _, out_nei, out_eid = g.out_edges(node, form="all", etype=etype)
                        nei     = torch.cat((in_nei,out_nei), dim = 0)
                        eid     = torch.cat((in_eid, out_eid), dim = 0)
                    
                    if nei.shape[0] == 0: break    
                    num_neigh   = self.fanouts[block_id][etype]             
                    nei_fake_labels = g.ndata['label_unk'][nei]            
                    classes = {}
                    eids    = {}
                    for i, fake_label in enumerate(nei_fake_labels):
                        fake_label = int(fake_label)
                        if fake_label not in classes:
                            classes[fake_label] = []
                        if fake_label not in eids:
                            eids[fake_label] = []
                        classes[fake_label].append(nei[i])
                        eids[fake_label].append(eid[i])
                    eid2class = {}
                    for class_, eids in eids.items():
                        class_eid = random.sample(sorted(eids), k=min(num_neigh, len(eids)))
                        eid2class[class_] = torch.stack(class_eid) if len(class_eid) > 0 else torch.tensor([]) 

                    sampled_eids = torch.cat(list(eid2class.values()), dim = -1).long()

                    edge_mask[sampled_eids] = 1
                new_edges_masks[etype] = edge_mask.bool()
            return edge_subgraph(g, new_edges_masks, relabel_nodes=False)
    

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        for block_id in reversed(range(self.num_layers)):
            frontier = self.sample_frontier(block_id, g, seed_nodes)
            eid = frontier.edata[EID]
            block = to_block(frontier, seed_nodes)
            block.edata[EID] = eid
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks
    
    def __len__(self):
        return self.num_layers


class CARESampler(BlockSampler):
    def __init__(self, p, dists, num_layers):
        super().__init__()
        self.p = p
        self.dists = dists
        self.num_layers = num_layers

    def sample_frontier(self, block_id, g, seed_nodes, *args, **kwargs):  
        with g.local_scope():
            new_edges_masks = {}
            for etype in g.canonical_etypes:
                edge_mask = torch.zeros(g.num_edges(etype))
                # extract each node from dict because of single node type
                for node in seed_nodes:  
                    edges = g.in_edges(node, form="eid", etype=etype)
                    num_neigh = (
                        torch.ceil(
                            g.in_degrees(node, etype=etype)
                            * self.p[block_id][etype]  
                        )  
                        .int()
                        .item()
                    )
                    neigh_dist = self.dists[block_id][etype][edges]  
                    if neigh_dist.shape[0] > num_neigh:
                        neigh_index = np.argpartition(neigh_dist, num_neigh)[
                            :num_neigh
                        ]
                    else:
                        neigh_index = np.arange(num_neigh)
                    edge_mask[edges[neigh_index]] = 1
                new_edges_masks[etype] = edge_mask.bool()

            return edge_subgraph(g, new_edges_masks, relabel_nodes=False)

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        for block_id in reversed(range(self.num_layers)):  
            frontier = self.sample_frontier(block_id, g, seed_nodes)

            eid = frontier.edata[EID]
            block = to_block(frontier, seed_nodes)
            block.edata[EID] = eid
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks

    def __len__(self):
        return self.num_layers

