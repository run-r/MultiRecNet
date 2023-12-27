""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from .CTrans_F import *
from collections import OrderedDict
from torch_geometric.nn import SAGEConv,LayerNorm

from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from .pretrainmodel import SAINT



class StdConv3d(nn.Conv3d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv3d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv3d = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv3d(out))
        out = out * x
        return out


def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn


    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            )


def init_max_weights(module):
    r"""
    Initialize Weights function.

    args:
        modules (torch.nn.Module): Initalize weight using normal distribution
    """
    import math
    import torch.nn as nn

    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


def GNN_relu_Block(dim2, dropout=0.3):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)
    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    return nn.Sequential(
            nn.ReLU(),
            LayerNorm(dim2),
            nn.Dropout(p=dropout))




def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


class my_GlobalAttention(torch.nn.Module):
    def __init__(self, gate_nn, nn=None):
        super(my_GlobalAttention, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, x, batch, size=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = x.shape[0] if size is None else size

        gate = self.gate_nn(x)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        for i in range(x.shape[0]):
            gate[i] = softmax(gate[i], batch[i].squeeze(-1),num_nodes=1)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)

        return out, gate

    def __repr__(self):
        return '{}(gate_nn={}, nn={})'.format(self.__class__.__name__,
                                              self.gate_nn, self.nn)



class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True, cat_dims= None):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc_t2w = DoubleConv(1, 16)
        self.inc_dwi = DoubleConv(1, 16)
        self.inc_adc = DoubleConv(1, 16)

        self.inc_t2w = DoubleConv(1, 16)
        self.inc_dwi = DoubleConv(1, 16)
        self.inc_adc = DoubleConv(1, 16)
        self.inc_k = DoubleConv(1, 16)

        self.skip_conv = DoubleConv(64, 32)
        self.skip_fuse = SpatialAttentionModule()
        self.modality_attention2 = ChannelTransformer2(vis=False, img_size=[64, 64, 8], dim=32,
                                                      channel_num=[32, 32, 32, 32],
                                                      patchSize=[1, 1, 1, 1])

        self.feature_fusion2 = nn.Sequential(OrderedDict([
            ('conv', StdConv3d(32 * 4, 64, kernel_size=3, stride=1, bias=False, padding=1)),
            ('gn', nn.GroupNorm(64, 64, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        self.down1_t = Down(16, 32)
        self.down1_d = Down(16, 32)
        self.down1_a = Down(16, 32)
        self.down1_k = Down(16, 32)

        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)


        self.classifier_censorship = nn.Linear(512, 1).cuda()
        self.Risk_evalutor = nn.Linear(512, 1).cuda()
        self.saint_process = SAINT(categories = tuple(cat_dims),
                                num_continuous =1)



    def forward(self, x,x_categ_enc, x_cont_enc):
        x_cli = self.saint_process.transformer(x_categ_enc, x_cont_enc)

        t2w = x[:, 0, :, :].unsqueeze(1)
        d = x[:, 1, :, :].unsqueeze(1)
        adc = x[:, 2, :, :].unsqueeze(1)
        k = x[:, 3, :, :].unsqueeze(1)

        t2w = self.inc_t2w(t2w)
        d = self.inc_dwi(d)
        adc = self.inc_adc(adc)
        k = self.inc_adc(k)

        skip = torch.cat([t2w, d, adc, k], dim=1)

        skip = self.skip_conv(skip)
        skip = self.skip_fuse(skip)

        t2w = self.down1_t(t2w)
        d = self.down1_d(d)
        adc = self.down1_a(adc)
        k = self.down1_k(k)

        attention_features = self.modality_attention2(t2w, d, adc, k)

        attention_feature = torch.cat(attention_features, dim=2)
        attention_feature = attention_feature.transpose(-1, -2)
        attention_feature = attention_feature.reshape(
            (attention_feature.shape[0], attention_feature.shape[1], 64, 64, 8))

        fusion1 = self.feature_fusion2(attention_feature)

        x3 = self.down2(fusion1)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        img_feature = x5
        b,c,w,h,d = img_feature.shape
        img_feature = nn.AdaptiveAvgPool1d(1)(img_feature.view(b,c,-1))
        x_cli = x_cli.permute(0,2,1)

        mix_feature = torch.cat((img_feature.squeeze(-1),x_cli[:,:,0]),dim=1)
        mix_feature = mix_feature.view(b,-1)
        mix_feature = self.classifier_censorship(mix_feature)


        risk_img = nn.AdaptiveAvgPool1d(1)(x5.clone().view(b,c,-1))
        risk_cli = nn.AdaptiveAvgPool1d(1)(x_cli[:, :, :-1])
        risk = torch.cat((risk_img.squeeze(-1), risk_cli.squeeze(-1)), dim=1)
        risk = self.Risk_evalutor(risk)


        x = self.up1(x5, x4)
        x_2 = self.up2(x, x3)
        x_3 = self.up3(x_2, fusion1)
        x_4 = self.up4(x_3, skip)
        logits = self.outc(x_4)


        return logits,nn.Sigmoid()(mix_feature),nn.Sigmoid()(risk)


