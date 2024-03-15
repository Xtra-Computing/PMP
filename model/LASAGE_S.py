# label imbalance enhanced
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from dgl import function as fn
import tqdm
import math
from dgl.utils import check_eq_shape, expand_as_pair
from dgl.utils import check_eq_shape, expand_as_pair
from dgl.base import DGLError
from torch_geometric.nn.norm import GraphNorm, GraphSizeNorm
from torch import Tensor
from .base import MLP
class LISeq(nn.Module):
    ''' 
    An extension of nn.Sequential. 
    Args: 
        modlist an iterable of modules to add.
    '''
    def __init__(self, modlist):
        super().__init__()
        self.modlist = nn.ModuleList(modlist)

    def forward(self, *args, **kwargs):
        out, src = self.modlist[0](*args, **kwargs)
        for i in range(1, len(self.modlist)):
            if isinstance(self.modlist[i], LILinear):
                out, src = self.modlist[i](out, src)
            elif isinstance(self.modlist[i], nn.Dropout):
                out = self.modlist[i](out)
            elif isinstance(self.modlist[i], GraphNorm):
                out = self.modlist[i](out)
        return out

class LIMLP(nn.Module):
    '''
    Multi-Layer Perception.
    Args:
        tail_activation: whether to use activation function at the last layer.
        activation: activation function.
        gn: whether to use GraphNorm layer.
    '''
    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 output_channels: int,
                 num_layers: int,
                 batch_size: int, 
                 origin_infeat: int,
                 dropout=0,
                 tail_activation=False,
                 activation=nn.ReLU(inplace=True),
                 gn=False):
        super().__init__()
        modlist = []
        self.seq = None
        if num_layers == 1:
            modlist.append(LILinear(input_channels, output_channels, batch_size, origin_infeat))
            if tail_activation:
                if gn:
                    modlist.append(GraphNorm(output_channels))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)
            self.seq = LISeq(modlist)
        else:
            modlist.append(LILinear(input_channels, hidden_channels, batch_size, origin_infeat))
            for _ in range(num_layers - 2):
                if gn:
                    modlist.append(GraphNorm(hidden_channels))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)
                modlist.append(LILinear(hidden_channels, hidden_channels, batch_size, origin_infeat))
            if gn:
                modlist.append(GraphNorm(hidden_channels))
            if dropout > 0:
                modlist.append(nn.Dropout(p=dropout, inplace=True))
            modlist.append(activation)
            modlist.append(LILinear(hidden_channels, output_channels, batch_size, origin_infeat))
            if tail_activation:
                if gn:
                    modlist.append(GraphNorm(output_channels))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)
            self.seq = LISeq(modlist)

    def forward(self, x, h_self):
        return self.seq(x, h_self)


class LILinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, batch_size: int, origin_infeat: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LILinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight           = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.w_src            = nn.Parameter(torch.empty((self.in_features, 1), **factory_kwargs))
        self.trans_src        = nn.Linear(origin_infeat, in_features)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_src, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, src: Tensor) -> Tensor:
        # w_src: 
        # input: (batch_size, D)
        # src: (batch_size, in_D)
        # so w_src should be (in_D, batch_size)

        # 1st layer: w_src: (32, bs)  src: (bs, 32)   -> tran_src: (32,32)
        #            weight: (256, 32)   tran_src: (32,32) -> tran_weight: (256,32)
        #            input: (bs, 32)   tran_weight: (256,32)  -> out: (bs, 256)
        # 2st later: w_src: (256, bs) src should be (bs, 256)   tran_src should be (256,256)
        #            weight: (256,256)           

        t_src = self.trans_src(src)   # (BS, in_dim) 

        trans_input = input * t_src   # (BS, in_dim)
        # tran_src = F.linear(self.w_src, t_src.transpose(1,0))  # d * d
        # tran_weight should be (out_D, in_D)
        # self.weight is (Out_D, in_D)
        # so train_src should be (in_D, in_D)
        # tran_weight = F.linear(self.weight, tran_src)    # 
        # return F.linear(input, self.weight, self.bias) # input* weight^T 

        out = F.linear(trans_input, self.weight, self.bias)  # (BS, 1, in_dim) (BS, in_dim, out_dim)
        
        return out, src  # input is (batch_size,in_D)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


def agg_func(nodes):
    # return {'neighbor_fraud': , }

    # src_feats = nodes.mailbox['m']  # [9, 1, 32]
    # src_nodes = nodes.mailbox['src'] # [9, 1]
    # src_fake_labels = nodes.mailbox['src_fake_label'] # [9, 1]
    

    return {'neigh_fr': (nodes.mailbox['m'] * (nodes.mailbox['src_fake_label']==1).unsqueeze(-1)).sum(1),
            'neigh_be': (nodes.mailbox['m'] * (nodes.mailbox['src_fake_label']==0).unsqueeze(-1)).sum(1),
            'neigh_unk': (nodes.mailbox['m'] * (nodes.mailbox['src_fake_label']==2).unsqueeze(-1)).sum(1)}

def mp_func(edges):
    src_fake_label = edges.src['label_unk']
    # src =  edges.edges()[0]
    src = edges.src['_ID']  
    dst = edges.dst['_ID']
    return {'m': edges.src['h'], 'src': src, 'src_fake_label': src_fake_label}

class LASAGESConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        aggregator_type,
        batch_size, 
        origin_infeat, 
        num_trans, 
        activation = nn.ReLU(),
        mlp_activation = nn.ReLU(inplace=True), 
        hid_feats = None,
        feat_drop=0.0,
        bias=True,
        norm=None,
    ):
        super(LASAGESConv, self).__init__()
        valid_aggre_types = {"mean", "gcn", "pool", "lstm"}
        if aggregator_type not in valid_aggre_types:
            raise DGLError(
                "Invalid aggregator_type. Must be one of {}. "
                "But got {!r} instead.".format(
                    valid_aggre_types, aggregator_type
                )
            )

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == "pool":
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == "lstm":
            self.lstm = nn.LSTM(
                self._in_src_feats, self._in_src_feats, batch_first=True
            )

        # self.fc_neigh_benign = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self.fc_neigh_benign  = LIMLP(self._in_src_feats, hidden_channels=hid_feats, output_channels=out_feats,
                                     num_layers=num_trans, batch_size = batch_size, origin_infeat = origin_infeat, activation=mlp_activation)
        
        # self.w_be             = nn.Parameter(torch.empty(self._in_src_feats, 1))



        self.fc_neigh_fraud  = LIMLP(self._in_src_feats, hidden_channels=hid_feats, output_channels=out_feats,
                                     num_layers=num_trans, batch_size = batch_size, origin_infeat = origin_infeat, activation=mlp_activation)
        
        # self.w_fr            = nn.Parameter(torch.empty(self._in_src_feats, 1))
        self.fc_neigh        = nn.Linear(self._in_src_feats, out_feats, bias=False)

        self.fc_balance      = MLP(self._in_src_feats, hidden_channels=hid_feats, output_channels=1,
                                     num_layers=2)
        

        self.balance_w       = nn.Sigmoid()

        if aggregator_type != "gcn":
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        elif bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if self._aggre_type == "pool":
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == "lstm":
            self.lstm.reset_parameters()
        if self._aggre_type != "gcn":
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

        # nn.init.kaiming_uniform_(self.w_be.weight, a = math.sqrt(5))
        # nn.init.kaiming_uniform_(self.w_fr.weight, a = math.sqrt(5))
        # nn.init.xavier_uniform_(self.fc_neigh_benign.weight, gain=gain)
        # nn.init.xavier_uniform_(self.fc_neigh_fraud.weight, gain=gain)

    def _lstm_reducer(self, nodes):
        m = nodes.mailbox["m"]  # (B, L, D)
        batch_size = m.shape[0]
        h = (
            m.new_zeros((1, batch_size, self._in_src_feats)),
            m.new_zeros((1, batch_size, self._in_src_feats)),
        )
        _, (rst, _) = self.lstm(m, h)
        return {"neigh": rst.squeeze(0)}
    
    def get_self_score(self, feats):
        return self.balance_w(self.fc_balance(feats))


    def forward(self, graph, feat, edge_weight=None):  
        with graph.local_scope():
            graph.srcdata['h'] = feat
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)  
                if graph.is_block:
                    # 终点的node features
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]  
                    ndst     = graph.srcdata['_ID'][: graph.number_of_dst_nodes()]
            # graph.srcdata['hl'] = torch.cat((graph.srcdata['h'], graph.srcdata['label_unk'].unsqueeze(1)), dim=-1)        
            msg_fn = fn.copy_u("h", "m")  
            # msg_fn = fn.copy_u("hl", "m")
            
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                msg_fn = fn.u_mul_e("h", "_edge_weight", "m")

            h_self = feat_dst  # 终点是自身节点


            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats  # False

            # Message Passing
            if self._aggre_type == "mean":
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src  # 所有两层节点
                ) 

                graph.update_all(mp_func, agg_func)
                # graph.update_all(msg_fn, fn.mean("m", "neigh")) # 邻居u的h信息放入message box中，然后message box中的信息做聚合后放入neigh中
                neigh_fr = graph.dstdata["neigh_fr"] # 中心节点聚合邻居后的表示
                neigh_be = graph.dstdata["neigh_be"]
                neigh_unk = graph.dstdata["neigh_unk"]
                # if not lin_before_mp:
                #     h_neigh = self.fc_neigh(h_neigh)


                neigh_fr  = self.fc_neigh_fraud(neigh_fr, h_self)
                neigh_be  = self.fc_neigh_benign(neigh_be, h_self)
                balance   = self.balance_w(self.fc_balance(h_self))
                neigh_unk = balance * self.fc_neigh_fraud(neigh_unk, h_self) + (1-balance) * self.fc_neigh_benign(neigh_unk, h_self)
                h_neigh = neigh_fr+ neigh_be + neigh_unk

            elif self._aggre_type == "gcn":
                check_eq_shape(feat)
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src
                )
                if isinstance(feat, tuple):  # heterogeneous
                    graph.dstdata["h"] = (
                        self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
                    )
                else:
                    if graph.is_block:
                        graph.dstdata["h"] = graph.srcdata["h"][
                            : graph.num_dst_nodes()
                        ]
                    else:
                        graph.dstdata["h"] = graph.srcdata["h"]
                graph.update_all(msg_fn, fn.sum("m", "neigh"))
                # divide in_degrees
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata["neigh"] + graph.dstdata["h"]) / (
                    degs.unsqueeze(-1) + 1
                )
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            elif self._aggre_type == "pool":
                graph.srcdata["h"] = F.relu(self.fc_pool(feat_src))
                graph.update_all(msg_fn, fn.max("m", "neigh"))
                h_neigh = self.fc_neigh(graph.dstdata["neigh"])
            elif self._aggre_type == "lstm":
                graph.srcdata["h"] = feat_src
                graph.update_all(msg_fn, self._lstm_reducer)
                h_neigh = self.fc_neigh(graph.dstdata["neigh"])
            else:
                raise KeyError(
                    "Aggregator type {} not recognized.".format(
                        self._aggre_type
                    )
                )

            # GraphSAGE GCN does not require fc_self.
            if self._aggre_type == "gcn":
                rst = h_neigh
                # add bias manually for GCN
                if self.bias is not None:
                    rst = rst + self.bias
            else:
                rst = self.fc_self(h_self) + h_neigh

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst


class LASAGE_S(nn.Module):
    def __init__(self, in_size, 
	      			   hid_size, 
					   out_size, 
					   num_layers, 
					   dropout, 
					   proj, 
					   num_relations,
                       batch_size,
                       num_trans,
                       mlp_activation = nn.ReLU(inplace=True),
					   out_proj_size = None, 
					   agg = "mean", 
					   relation_agg = None):
        super(LASAGE_S, self).__init__()
        origin_infeat = in_size
        self.layers = nn.ModuleList()
        self.in_size = in_size
        self.hid_size = hid_size
        self.out_size = out_size
        self.proj = proj
        self.relation_agg = relation_agg
        self.num_relations = num_relations
        for i in range(num_layers):
            if i == num_layers-1:
                hid_size = out_size
            self.layers.append(LASAGESConv(in_size, 
                                            hid_size, 
                                            agg, 
                                            hid_feats=hid_size, 
                                            batch_size=batch_size, 
                                            origin_infeat = origin_infeat, 
                                            num_trans=num_trans,
                                            mlp_activation=mlp_activation))
            in_size = hid_size

        self.dropout = dropout

        # if self.relation_agg == 'cat':
        self.relation_mlp = nn.ModuleList()
        for j in range(num_layers):
            if i == num_layers-1:
                hid_size = out_size
            self.relation_mlp.append(nn.Linear(hid_size*num_relations if self.relation_agg == 'cat' else hid_size, hid_size))

        if proj:
            self.Conv = nn.Conv1d(out_size, out_proj_size, kernel_size=1)


    def forward(self, blocks, relations, feats):   # blocks 
        # print(blocks)
        # print(feats.shape) torch.Size([42689, 32])
        h = feats 
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):  # blocks of one relation in one layer
            layer_emb = []  
            for etype in relations:
                b_graph = block[etype]  
                # dgl_subgraph = dgl.block_to_graph(b_graph)
                # print(dgl_subgraph.ndata['label'])
                # print(b_graph.srcdata['label'])
                # print(b_graph.dstdata['label'])
                layer_emb.append(layer(b_graph, h))
            if self.relation_agg == 'cat':
                relation_agg_emb = torch.cat(layer_emb, dim=1)
            elif self.relation_agg == 'mean':
                relation_agg_emb = torch.mean(layer_emb, dim=0)
            elif self.relation_agg == 'add':
                relation_agg_emb = torch.sum(layer_emb, dim=0)
            h = self.relation_mlp[l](relation_agg_emb)
                    
            # h = layer(block, h)  
            if l != len(self.layers) - 1:
                h = F.relu(h)
                # h = self.dropout(h)  #
        if self.proj:
            if self.dropout > 0:
                h = F.dropout(h, self.dropout, training=self.training)
            h = h.unsqueeze(0)
            h = h.permute((0,2,1))
            h = self.Conv(h)
            h = h.permute((0,2,1)).squeeze()
        return h
