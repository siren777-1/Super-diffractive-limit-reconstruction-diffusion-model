import torch
import torch.nn as nn
import functools

class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.SCAU = SCAU(nf)
        self.SGFE = SGFE(nf)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, get_fea=False):
        feas = []
        x = (x + 1) / 2  

        fea_first = fea = self.conv_first(x)

        for l in self.RRDB_trunk:
            fea = l(fea)
            feas.append(fea)

        fea = self.SCAU(fea)
        fea = self.SGFE(fea)
        trunk = self.trunk_conv(fea)
        fea = fea_first + trunk
        fea_hr = self.HRconv(fea)
        out = self.conv_last(self.lrelu(fea_hr))
        out = out.clamp(0, 1)
        out = out * 2 - 1  

        if get_fea:
            return out, feas
        else:
            return out
