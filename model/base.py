import torch.nn as nn
import torch
from torch_geometric.nn.norm import GraphNorm, GraphSizeNorm
from torch import Tensor
import math
from torch.nn import functional as F

class Seq(nn.Module):
    ''' 
    An extension of nn.Sequential. 
    Args: 
        modlist an iterable of modules to add.
    '''
    def __init__(self, modlist):
        super().__init__()
        self.modlist = nn.ModuleList(modlist)

    def forward(self, *args, **kwargs):
        out = self.modlist[0](*args, **kwargs)
        for i in range(1, len(self.modlist)):
            out = self.modlist[i](out)
        return out


class MLP(nn.Module):
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
                 dropout=0,
                 tail_activation=False,
                 activation=nn.ReLU(inplace=True),
                 gn=False):
        super().__init__()
        modlist = []
        self.seq = None
        if num_layers == 1:
            modlist.append(nn.Linear(input_channels, output_channels))
            if tail_activation:
                if gn:
                    modlist.append(GraphNorm(output_channels))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)
            self.seq = Seq(modlist)
        else:
            modlist.append(nn.Linear(input_channels, hidden_channels))
            for _ in range(num_layers - 2):
                if gn:
                    modlist.append(GraphNorm(hidden_channels))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)
                modlist.append(nn.Linear(hidden_channels, hidden_channels))
            if gn:
                modlist.append(GraphNorm(hidden_channels))
            if dropout > 0:
                modlist.append(nn.Dropout(p=dropout, inplace=True))
            modlist.append(activation)
            modlist.append(nn.Linear(hidden_channels, output_channels))
            if tail_activation:
                if gn:
                    modlist.append(GraphNorm(output_channels))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)
            self.seq = Seq(modlist)

    def forward(self, x):
        return self.seq(x)

#################################################################################################################
#################################################################################################################


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
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
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
        self.trans_src        = nn.Linear(origin_infeat, out_features)
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

        # w_src 256*64  src 64*32
        t_src = self.trans_src(src)   # (BS, out_dim) 

        weight = torch.matmul(self.w_src.unsqueeze(0), t_src.unsqueeze(1))  # (BS, in_dim, out_dim)

        # tran_src = F.linear(self.w_src, t_src.transpose(1,0))  # d * d
        # tran_weight should be (out_D, in_D)
        # self.weight is (Out_D, in_D)
        # so train_src should be (in_D, in_D)
        # tran_weight = F.linear(self.weight, tran_src)    # 
        # return F.linear(input, self.weight, self.bias) # input* weight^T 

        out = torch.matmul(input.unsqueeze(1), weight).squeeze()  # (BS, 1, in_dim) (BS, in_dim, out_dim)
        
        return out, src  # input is (batch_size,in_D)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
