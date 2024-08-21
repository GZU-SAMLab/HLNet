import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from einops import rearrange


#SpaRG##########################################################################

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

# Define ChannelPool
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale



class RSAB(nn.Module):
    def __init__(self, n_feat, kernel_size):

        super(RSAB, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=True)
        )
        self.sa = spatial_attn_layer()


    def forward(self, x):
        res = self.body(x)
        res = self.sa(res)
        res = res + x
        return res


## Residual Group (RG)
class spaResidualGroup(nn.Module):
    def __init__(self, n_feat, kernel_size):
        super(spaResidualGroup, self).__init__()

        self.body1 = RSAB(n_feat, kernel_size)
        self.body2 = RSAB(n_feat, kernel_size)
        self.body3 = RSAB(n_feat, kernel_size)
        self.body4 = RSAB(n_feat, kernel_size)
        self.body5 = RSAB(n_feat, kernel_size)
        self.conv0 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=True)


    def forward(self, x):
        res = self.body1(x)
        res = self.body2(res)
        res = self.body3(res)
        res = self.body4(res)
        res = self.body5(res)
        res = self.conv0(res)
        res += x
        return res

#SpaRG#########################################################################

#SpeRG#########################################################################
## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, n_feat, kernel_size):

        super(RCAB, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=True)
        )
        self.ca = CALayer(n_feat)


    def forward(self, x):
        res = self.body(x)
        res = self.ca(res)
        res = res + x
        return res

## Residual Group (RG)
class speResidualGroup(nn.Module):
    def __init__(self, n_feat, kernel_size):
        super(speResidualGroup, self).__init__()

        self.body1 = RCAB(n_feat, kernel_size)
        self.body2 = RCAB(n_feat, kernel_size)
        self.body3 = RCAB(n_feat, kernel_size)
        self.body4 = RCAB(n_feat, kernel_size)
        self.body5 = RCAB(n_feat, kernel_size)
        self.conv0 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=True)


    def forward(self, x):
        res = self.body1(x)
        res = self.body2(res)
        res = self.body3(res)
        res = self.body4(res)
        res = self.body5(res)
        res = self.conv0(res)
        res += x
        return res

#SpeRG#########################################################################


#SSRG#########################################################################
## Dual Attention Block (DAB)

# modify
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class MSAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)       # b h w c
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)     # b c h w
        return out

class DAB(nn.Module):
    # in: [b, ch, h, w]
    # out: [b, ch, h, w]
    def __init__(self, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True)):

        super(DAB, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=True)
        )

        self.SA = spatial_attn_layer()  ## Spatial Attention
        self.CA = CALayer(n_feat)  ## Channel Attention
        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        res += x
        return res


## Recursive Residual Group (RRG)
class RRG(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction=16, act=nn.ReLU(True),  num_blocks = 3):
        super(RRG, self).__init__()
        modules_body = []
        modules_body = [
            MSAB(dim=n_feat, num_blocks=num_blocks, dim_head=n_feat, heads=1) \
            for _ in range(num_blocks)]
        self.conv = nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=True)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.conv(res)
        res += x
        return res
# modify
#SSRG#########################################################################

##########################################################################
class SpatialBlock(nn.Module):
    def __init__(self, kernel_size=3, n_channels=64):
        super(SpatialBlock, self).__init__()
        self.spatial_layers = spaResidualGroup(n_channels, kernel_size=kernel_size)

    def forward(self, spatial_x):
        spatial_x = self.spatial_layers(spatial_x)
        return spatial_x


class SpectralBlock(nn.Module):
    def __init__(self,kernel_size=3, n_channels=64):
        super(SpectralBlock, self).__init__()
        self.spectral_layers = speResidualGroup(n_channels, kernel_size=kernel_size)

    def forward(self, spectral_x):
        spectral_x = self.spectral_layers(spectral_x)
        return spectral_x


class FusionBlock(nn.Module):
    def __init__(self, kernel_size=3, n_channels=64):
        super(FusionBlock, self).__init__()
        self.fusion_layers = nn.Sequential(
            nn.Conv2d(n_channels *2, n_channels *2, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels * 2, n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True)
        )
        self.spatial_broadcast = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            # nn.BatchNorm2d(n_channels),
        )
        self.spectral_broadcast = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            # nn.BatchNorm2d(n_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, spatial_x, spectral_x):
        ss_x = torch.cat([spatial_x, spectral_x], dim=1)
        ss_x = self.fusion_layers(ss_x)
        spatial_x = spatial_x + self.spatial_broadcast(ss_x)
        spatial_x = self.relu(spatial_x)
        spectral_x = spectral_x + self.spectral_broadcast(ss_x)
        spectral_x = self.relu(spectral_x)

        return ss_x, spatial_x, spectral_x


class FusionBlock2(nn.Module):
    def __init__(self, kernel_size=3, n_channels=64):
        super(FusionBlock2, self).__init__()
        self.fusion_layers = nn.Sequential(
            nn.Conv2d(n_channels *3, n_channels *3, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels * 3, n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True)
        )
        self.spatial_broadcast = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            # nn.BatchNorm2d(n_channels),
        )
        self.spectral_broadcast = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            # nn.BatchNorm2d(n_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, ss_x, spatial_x, spectral_x):
        ss_x2 = torch.cat([ss_x, spatial_x, spectral_x], dim=1)
        ss_x2 = self.fusion_layers(ss_x2)
        spatial_x = spatial_x + self.spatial_broadcast(ss_x2)
        spatial_x = self.relu(spatial_x)
        spectral_x = spectral_x + self.spectral_broadcast(ss_x2)
        spectral_x = self.relu(spectral_x)

        return ss_x2, spatial_x, spectral_x


class FusionBlock3(nn.Module):
    def __init__(self, kernel_size=3, n_channels=64):
        super(FusionBlock3, self).__init__()
        self.fusion_layers = nn.Sequential(
            nn.Conv2d(n_channels *4, n_channels *4, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels * 4, n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True)
        )
        self.spatial_broadcast = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            # nn.BatchNorm2d(n_channels),
        )
        self.spectral_broadcast = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            # nn.BatchNorm2d(n_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, ss_x, ss_x2, spatial_x, spectral_x):
        ss_x3 = torch.cat([ss_x, ss_x2, spatial_x, spectral_x], dim=1)
        ss_x3 = self.fusion_layers(ss_x3)
        spatial_x = spatial_x + self.spatial_broadcast(ss_x3)
        spatial_x = self.relu(spatial_x)
        spectral_x = spectral_x + self.spectral_broadcast(ss_x3)
        spectral_x = self.relu(spectral_x)

        return ss_x3, spatial_x, spectral_x

class DLB_cir(nn.Module):
    def __init__(self, n_channels, kernel_size):
        super(DLB_cir, self).__init__()
        self.dlb_cir = dlb_cir = nn.Sequential(
            nn.Conv2d(in_channels=4 * n_channels, out_channels=4 * n_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self,x ):
        return self.dlb_cir(x)



class DLB(nn.Module):
    def __init__(self,n_channels, kernel_size, n_dlb):
        super(DLB, self).__init__()

        self.dlb = nn.Sequential(
            nn.Conv2d(in_channels=5 * n_channels, out_channels=4 * n_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True)
        )


        dlb_bank = [DLB_cir(n_channels=n_channels, kernel_size=kernel_size) for _ in range(n_dlb-1)]

        self.dlb_bank = nn.Sequential(*dlb_bank)

    def forward(self, f_n):

        x = self.dlb(f_n)
        f_L = self.dlb_bank(x)

        return f_L

class SpatialSpectralSRNet_test(nn.Module):
    def __init__(self, in_channels=4, out_channels=102, n_channels=64, n_blocks=7, n_dlb=1, kernel_size=3, upscale_factor=2):
        super(SpatialSpectralSRNet_test, self).__init__()
        self.n_blocks = n_blocks
        self.pre_spatial_layers = nn.Sequential(
            nn.Conv2d(in_channels, n_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True),
            # nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True)
        )
        self.pre_spectral_layers = nn.Sequential(
            nn.Conv2d(in_channels, n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            # nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True)
        )
        relu = nn.ReLU(inplace=True)

        self.spa1 = SpatialBlock(kernel_size=kernel_size, n_channels=n_channels)
        self.spe1 = SpectralBlock(kernel_size=kernel_size, n_channels=n_channels)
        self.fusion1 = FusionBlock(kernel_size = kernel_size, n_channels=n_channels)

        self.spa2 = SpatialBlock(kernel_size = kernel_size, n_channels=n_channels)
        self.spe2 = SpectralBlock(kernel_size = kernel_size, n_channels=n_channels)
        self.fusion2 = FusionBlock2(kernel_size = kernel_size, n_channels=n_channels)

        self.spa3 = SpatialBlock(kernel_size=kernel_size, n_channels=n_channels)
        self.spe3 = SpectralBlock(kernel_size=kernel_size, n_channels=n_channels)
        self.fusion3 = FusionBlock3(kernel_size = kernel_size, n_channels=n_channels)

        # isolated spatial and spectral loss layers
        self.post_spatial_layers = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(in_channels=n_channels // (upscale_factor * upscale_factor), out_channels=in_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        )
        self.post_spectral_layers = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
        )

        # fusion and reconstruction module
        self.fusion_net =  nn.Sequential(
            RRG(n_channels, kernel_size=kernel_size)
        )

        self.dlb = DLB(n_channels = n_channels, kernel_size=kernel_size, n_dlb=n_dlb)

        self.pre_fusion_layers = nn.Sequential(
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(in_channels=4 * n_channels // (upscale_factor * upscale_factor), out_channels=n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        )

        self.upscale = nn.PixelShuffle(upscale_factor) # channel/4  height*2 width*2

        # !!!!!!!!!!!!!!!!!!!!!!!
        # just change the channel
        # modified the kernel_size to 1
        self.pre_branch_layers = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=4 * n_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(in_channels=4 * n_channels // (upscale_factor * upscale_factor), out_channels=n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        )


        self.pre_fout_layer = nn.Sequential(
            nn.Conv2d(in_channels=6 * n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        )
        # self.conv1 = nn.Conv2d(in_channels=6*n_channels, out_channels=3*n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
        # nn.ReLU(inplace=True),
        # self.conv2 = nn.Conv2d(in_channels=3*n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)

        self.fusion_block = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        )

    def forward(self, x):
        spatial_x = self.pre_spatial_layers(x)
        spectral_x = self.pre_spectral_layers(x)

        spatial_x_res = self.spa1(spatial_x)
        spectral_x_res = self.spe1(spectral_x)
        ss_x, spatial_x_res, spectral_x_res = self.fusion1(spatial_x_res, spectral_x_res)

        spatial_x_res = self.spa2(spatial_x_res)
        spectral_x_res = self.spe2(spectral_x_res)
        ss_x2, spatial_x_res, spectral_x_res = self.fusion2(ss_x, spatial_x_res, spectral_x_res)

        spatial_x_res = self.spa3(spatial_x_res)
        spectral_x_res = self.spe3(spectral_x_res)
        ss_x3, spatial_x_res, spectral_x_res = self.fusion3(ss_x, ss_x2, spatial_x_res, spectral_x_res)


        spatial_x = spatial_x + spatial_x_res
        spectral_x = spectral_x + spectral_x_res
        out_spatial = self.post_spatial_layers(spatial_x)
        out_spectral = self.post_spectral_layers(spectral_x)

        x = torch.cat([ss_x, ss_x2, ss_x3, spatial_x, spectral_x], dim=1)


        x = self.dlb(x)
        x = self.pre_fusion_layers(x)
        res1 = self.fusion_net(x)
        res2 = self.fusion_net(res1)
        res3 = self.fusion_net(res2)
        res4 = self.fusion_net(res3)



        spatial_x_fout = self.pre_branch_layers(spatial_x)
        spectral_x_fout = self.pre_branch_layers(spectral_x)




        res = torch.cat([res1, res2, res3, res4, spatial_x_fout, spectral_x_fout], dim=1) # add here
        res = self.pre_fout_layer(res)
        x = x + res
        out = self.fusion_block(x)
        return out_spatial, out_spectral, out
