import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from gi.modules.lama.saicinpainting.training.modules.base import get_activation, BaseDiscriminator
from gi.modules.lama.saicinpainting.training.modules.spatial_transform import LearnableSpatialTransformWrapper
from gi.modules.lama.saicinpainting.training.modules.squeeze_excitation import SELayer
from gi.modules.lama.saicinpainting.utils import get_shape
from gi.modules.RAFT.flow_utils import resized_flow_sample


def resize_mods(x, mods):
    b,c,h,w = x.shape
    if (h%mods[0]==0) and (w%mods[1])==0:
        return x
    h = mods[0] * round(h / mods[0])
    w = mods[1] * round(w / mods[1])
    return torch.nn.functional.interpolate(
        x, size=(h, w), mode="bilinear", align_corners=True
    )


class FFCSE_block(nn.Module):

    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r,
                               kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(
            channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(
            channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))

        x_l = 0 if self.conv_a2l is None else id_l * \
            self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * \
            self.sigmoid(self.conv_a2g(x))
        return x_l, x_g


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        # squeeze and excitation block
        self.use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output


class SeparableFourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, kernel_size=3):
        # bn_layer not used
        super(SeparableFourierUnit, self).__init__()
        self.groups = groups
        row_out_channels = out_channels // 2
        col_out_channels = out_channels - row_out_channels
        self.row_conv = torch.nn.Conv2d(in_channels=in_channels * 2,
                                        out_channels=row_out_channels * 2,
                                        kernel_size=(kernel_size, 1),  # kernel size is always like this, but the data will be transposed
                                        stride=1, padding=(kernel_size // 2, 0),
                                        padding_mode='reflect',
                                        groups=self.groups, bias=False)
        self.col_conv = torch.nn.Conv2d(in_channels=in_channels * 2,
                                        out_channels=col_out_channels * 2,
                                        kernel_size=(kernel_size, 1),  # kernel size is always like this, but the data will be transposed
                                        stride=1, padding=(kernel_size // 2, 0),
                                        padding_mode='reflect',
                                        groups=self.groups, bias=False)
        self.row_bn = torch.nn.BatchNorm2d(row_out_channels * 2)
        self.col_bn = torch.nn.BatchNorm2d(col_out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def process_branch(self, x, conv, bn):
        batch = x.shape[0]

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft(x, norm="ortho")
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.relu(bn(conv(ffted)))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        output = torch.fft.irfft(ffted, s=x.shape[-1:], norm="ortho")
        return output


    def forward(self, x):
        rowwise = self.process_branch(x, self.row_conv, self.row_bn)
        colwise = self.process_branch(x.permute(0, 1, 3, 2), self.col_conv, self.col_bn).permute(0, 1, 3, 2)
        out = torch.cat((rowwise, colwise), dim=1)
        return out


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, separable_fu=False, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        fu_class = SeparableFourierUnit if separable_fu else FourierUnit
        self.fu = fu_class(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = fu_class(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

        return out_xl, out_xg


def embed_layer(in_dim, out_dim):
    return nn.Sequential(
        nn.SiLU(),
        nn.Linear(
            in_dim,
            out_dim,
        ),
    )


class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True,
                 time_embed_dim=None,
                 **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

        self.time_embed_dim = time_embed_dim
        if self.time_embed_dim is not None:
            embed_l = (lambda *args, **kwargs: None) if ratio_gout == 1 else embed_layer
            embed_g = (lambda *args, **kwargs: None) if ratio_gout == 0 else embed_layer
            self.embed_l = embed_l(self.time_embed_dim, out_channels - global_channels)
            self.embed_g = embed_g(self.time_embed_dim, global_channels)

    def forward(self, x, emb=None):
        x_l, x_g = self.ffc(x)
        if emb is not None:
            if self.embed_l is not None:
                emb_l = self.embed_l(emb)
                while len(emb_l.shape) < len(x_l.shape):
                    emb_l = emb_l[..., None]
                x_l = x_l + emb_l
            if self.embed_g is not None:
                emb_g = self.embed_g(emb)
                while len(emb_g.shape) < len(x_g.shape):
                    emb_g = emb_g[..., None]
                x_g = x_g + emb_g
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class CONV_NORM_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        self.norm = norm_layer(out_channels)
        self.act = activation_layer(inplace=True)


    def forward(self, x, emb=None):
        return self.act(self.norm(self.conv(x)))


class RESIDUAL_CONV_NORM_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        self.norm = norm_layer(out_channels)
        self.act = activation_layer(inplace=True)


    def forward(self, x, emb=None):
        return x+self.act(self.norm(self.conv(x)))



class FFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline

    def forward(self, x, emb=None):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g), emb=emb)
        x_l, x_g = self.conv2((x_l, x_g), emb=emb)

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out


class SplitTupleLayer(nn.Module):
    def forward(self, x):
        return x[:, :128], x[:, 128:]


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)


class LatentInitLayer(nn.Module):
    def __init__(self, in_channels, num_latents_sqrt, d_latents):
        super().__init__()
        num_latents = num_latents_sqrt**2
        self.latents = nn.Parameter(torch.randn(num_latents, d_latents))
        self.pos_embeddings = PerceiverFourierPositionEncoding(
            num_bands=64, max_resolution=(32, 32), concat_pos=True, sine_only=False,
        )
        self.project = nn.Conv2d(in_channels+258, in_channels, 1)

    def forward(self, x):
        b = x.shape[0]
        pos_emb = self.pos_embeddings(
            x.shape[-2:], batch_size=b, device=x.device,
        )
        pos_emb = rearrange(pos_emb, "b (h w) c -> b c h w", b=b, h=x.shape[2], w=x.shape[3])
        x = self.project(torch.cat((x, pos_emb), dim=1))
        return x, self.latents.expand(b, -1, -1)  # Thanks^2, Phil Wang


class DropLatentLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        return x[0]


class PerceiverSelfAttention(nn.Module):
    """Multi-headed {cross, self}-attention. Can be used both in the encoder as well as in the decoder."""

    def __init__(
        self,
        attention_probs_dropout_prob=0,
        is_cross_attention=False,
        qk_channels=None,
        v_channels=None,
        num_heads=1,
        q_dim=None,
        kv_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        # Q and K must have the same number of channels.
        # Default to preserving Q's input's shape.
        if qk_channels is None:
            qk_channels = q_dim
        # V's num_channels determines the shape of the output of QKV-attention.
        # Default to the same number of channels used in the key-query operation.
        if v_channels is None:
            v_channels = qk_channels
        if qk_channels % num_heads != 0:
            raise ValueError(f"qk_channels ({qk_channels}) must be divisible by num_heads ({num_heads}).")
        if v_channels % num_heads != 0:
            raise ValueError(f"v_channels ({v_channels}) must be divisible by num_heads ({num_heads}).")

        self.qk_channels = qk_channels
        self.v_channels = v_channels
        self.qk_channels_per_head = self.qk_channels // num_heads
        self.v_channels_per_head = self.v_channels // num_heads

        # Layer normalization
        self.layernorm1 = nn.LayerNorm(q_dim)
        self.layernorm2 = nn.LayerNorm(kv_dim) if is_cross_attention else nn.Identity()

        # Projection matrices
        self.query = nn.Linear(q_dim, qk_channels)
        self.key = nn.Linear(kv_dim, qk_channels)
        self.value = nn.Linear(kv_dim, v_channels)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x, channels_per_head):
        new_x_shape = x.size()[:-1] + (self.num_heads, channels_per_head)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        inputs=None,
        kv=None,
        return_kv=False,
    ):
        hidden_states = self.layernorm1(hidden_states)
        inputs = self.layernorm2(inputs)

        # Project queries, keys and values to a common feature dimension. If this is instantiated as a cross-attention module,
        # the keys and values come from the inputs; the attention mask needs to be such that the inputs's non-relevant tokens are not attended to.
        is_cross_attention = inputs is not None
        queries = self.query(hidden_states)

        if is_cross_attention:
            keys = self.key(inputs)
            values = self.value(inputs)
        else:
            keys = self.key(hidden_states)
            values = self.value(hidden_states)

        # Reshape channels for multi-head attention.
        # We reshape from (batch_size, time, channels) to (batch_size, num_heads, time, channels per head)
        queries = self.transpose_for_scores(queries, self.qk_channels_per_head)
        keys = self.transpose_for_scores(keys, self.qk_channels_per_head)
        values = self.transpose_for_scores(values, self.v_channels_per_head)
        if return_kv:
            kv_out = [keys, values]
        if kv is not None:
            keys = torch.cat((keys, kv[0]), dim=2)
            values = torch.cat((values, kv[1]), dim=2)

        # Take the dot product between the queries and keys to get the raw attention scores.
        attention_scores = torch.matmul(queries, keys.transpose(-1, -2))

        batch_size, num_heads, seq_len, q_head_dim = queries.shape
        _, _, _, v_head_dim = values.shape
        hiddens = self.num_heads * v_head_dim

        attention_scores = attention_scores / math.sqrt(q_head_dim)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, values)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (hiddens,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if return_kv:
            return context_layer, kv_out

        return context_layer


class PerceiverMLP(nn.Module):
    """A Transformer-style dense module to follow attention."""

    def __init__(self, input_size, widening_factor):
        super().__init__()
        self.dense1 = nn.Linear(input_size, widening_factor * input_size)
        self.intermediate_act_fn = nn.GELU()
        self.dense2 = nn.Linear(widening_factor * input_size, input_size)

    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return hidden_states


class PerceiverMLPConv(nn.Module):
    """A Transformer-style dense module to follow attention."""

    def __init__(self, input_size, widening_factor):
        super().__init__()
        self.dense1 = nn.Conv2d(input_size, widening_factor * input_size, 1)
        self.intermediate_act_fn = nn.GELU()
        self.dense2 = nn.Conv2d(widening_factor * input_size, input_size, 1)

    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return hidden_states


class AttentionLayer(nn.Module):
    def __init__(self, q_dim, num_heads, widening_factor=1):
        super().__init__()
        self.attn = PerceiverSelfAttention(
            attention_probs_dropout_prob=0,
            is_cross_attention=False,
            num_heads=num_heads,
            q_dim=q_dim,
            kv_dim=q_dim,
        )
        self.projection = nn.Linear(q_dim, q_dim)
        self.ln_premlp = nn.LayerNorm(q_dim)
        self.mlp = PerceiverMLP(q_dim, widening_factor=widening_factor)

    def forward(self, x, kv=None, return_kv=False):
        xres = x

        B_, N, D = x.shape

        x = self.attn(x, kv=kv, return_kv=return_kv) # ln, attn
        if return_kv:
            x, kv = x
        x = self.projection(x) # dense projection
        x = x + xres

        xres = x

        x = self.ln_premlp(x)
        x = self.mlp(x)
        x = x + xres

        if return_kv:
            return x, kv

        return x


class ReadLayer(nn.Module):
    def __init__(self, q_dim, kv_dim, num_heads, widening_factor=1):
        super().__init__()
        self.attn = PerceiverSelfAttention(
            attention_probs_dropout_prob=0,
            is_cross_attention=True,
            num_heads=num_heads,
            q_dim=q_dim,
            kv_dim=kv_dim,
        )
        self.projection = nn.Linear(q_dim, q_dim)
        self.ln_premlp = nn.LayerNorm(q_dim)
        self.mlp = PerceiverMLP(q_dim, widening_factor=widening_factor)

    def forward(self, xlr, xhr):
        xres = xlr

        B, C, H, W = xhr.shape
        B_, N, D = xlr.shape
        assert B==B_

        windows = int(np.sqrt(N))
        assert windows**2 == N, (N, windows)

        assert H%windows == 0 and W%windows == 0, (H, W, windows)

        # convert hr to sequence
        xhr = rearrange(xhr, "b c (winh pixh) (winw pixw) -> (b winh winw) (pixh pixw) c",
                        b=B, c=C, winh=windows, pixh=H//windows, winw=windows, pixw=W//windows)
        xlr = rearrange(xlr, "b n d -> (b n) 1 d",
                        b=B, n=N, d=D)
        xlr = self.attn(xlr, xhr) # lr, attn
        xlr = rearrange(xlr, "(b n) 1 d -> b n d", b=B, n=N)

        xlr = self.projection(xlr) # dense projection
        xlr = xlr + xres

        xres = xlr

        xlr = self.ln_premlp(xlr)
        xlr = self.mlp(xlr)
        xlr = xlr + xres

        return xlr


def sparse_mixture_2d(logits, values, k=None, return_attn=False):
    B, C, H, W = logits.shape
    B_, C_, H_, W_ = values.shape
    assert B==B_ and H==H_ and W==W_

    logits = rearrange(logits, "b c h w -> b c (h w)",
                       b=B, c=C, h=H, w=W)
    values = rearrange(values, "b c h w -> b (h w) c",
                       b=B, c=C_, h=H, w=W)

    if k is not None:
        with torch.no_grad():
            # top-k mask
            v, _ = torch.topk(logits, k)
            v = v[:, :, -1].unsqueeze(-1)
            m = (logits < v).float()
            m[m>0] = -float("inf")
        logits = logits + m

    if not return_attn:
        return torch.matmul(F.softmax(logits, dim=-1), values)
    else:
        attn = F.softmax(logits, dim=-1)
        return torch.matmul(attn, values), rearrange(attn, "b c (h w) -> b c h w", h=H, w=W)


def local_mixture_2d(logits, values, winh, winw, k=None, return_attn=False):
    logits = resize_mods(logits, (winh, winw))
    values = resize_mods(values, (winh, winw))

    B, C, H, W = logits.shape
    num_heads = C
    B_, C_, H_, W_ = values.shape
    assert B==B_ and H==H_ and W==W_
    assert C_ % num_heads == 0, (C_, num_heads)

    logits = rearrange(
        logits,
        "b heads (winh pixh) (winw pixw) -> (b heads winh winw) 1 (pixh pixw)",
        b=B, heads=num_heads, winh=winh, pixh=H//winh, winw=winw, pixw=W//winw,
    )
    values = rearrange(
        values,
        "b (heads c) (winh pixh) (winw pixw) -> (b heads winh winw) (pixh pixw) c",
        b=B, heads=num_heads, c=C_//num_heads, winh=winh, pixh=H//winh, winw=winw, pixw=W//winw,
    )
    attn = F.softmax(logits, dim=-1)
    out = torch.matmul(attn, values)
    out = rearrange(out, "(b heads winh winw) 1 c -> b (winh winw) (heads c)",
                    b=B, heads=num_heads, c=C_//num_heads, winh=winh, winw=winw)

    if not return_attn:
        return out
    else:
        return out, rearrange(
            attn, "(b heads winh winw) 1 (pixh pixw) -> b heads (winh pixh) (winw pixw)",
            b=B, heads=num_heads, winh=winh, pixh=H//winh, winw=winw, pixw=W//winw,
        )


def sparse_mixture_2d_transposed(logits, values, k=None, return_attn=False):
    assert k is None

    B, C, H, W = logits.shape
    B_, N, C_ = values.shape
    assert B==B_ and C==N

    logits = rearrange(logits, "b c h w -> b (h w) c",
                       b=B, c=N, h=H, w=W)

    out = torch.matmul(F.softmax(logits, dim=-1), values)
    out = rearrange(out, "b (h w) c -> b c h w",
                    b=B, h=H, w=W, c=C_)
    if not return_attn:
        return out
    else:
        attn = F.softmax(logits, dim=-1)
        return out, rearrange(attn, "b (h w) c -> b c h w", h=H, w=W)


def local_mixture_2d_transposed(logits, values, winh, winw, k=None, return_attn=False):
    assert k is None

    _, _, oH, oW = logits.shape
    logits = resize_mods(logits, (winh, winw))
    B, C, H, W = logits.shape
    num_heads = C
    B_, N, C_ = values.shape
    assert B==B_
    assert C_ % num_heads == 0, (C_, num_heads)

    logits = rearrange(
        logits,
        "b heads (winh pixh) (winw pixw) -> (b heads winh winw) (pixh pixw) 1",
        b=B, heads=num_heads, winh=winh, pixh=H//winh, winw=winw, pixw=W//winw,
    )
    attn = torch.sigmoid(logits)
    values = rearrange(
        values,
        "b (winh winw) (heads c) -> (b heads winh winw) 1 c",
        b=B, heads=num_heads, winh=winh, winw=winw, c=C_//num_heads,
    )

    out = torch.matmul(attn, values)
    out = rearrange(
        out,
        "(b heads winh winw) (pixh pixw) c -> b (heads c) (winh pixh) (winw pixw)",
        b=B, heads=num_heads, winh=winh, pixh=H//winh, winw=winw, pixw=W//winw,
        c=C_//num_heads,
    )

    if oH != H or oW != W:
        out = torch.nn.functional.interpolate(
            out, size=(oH, oW), mode="bilinear", align_corners=True
        )
    if not return_attn:
        return out
    else:
        return out, rearrange(
            attn,
            "(b heads winh winw) (pixh pixw) c -> b (heads c) (winh pixh) (winw pixw)",
            b=B, heads=num_heads, winh=winh, pixh=H//winh, winw=winw,
            pixw=W//winw, c=1,
        )


class FastReadLayer(nn.Module):
    def __init__(self, q_dim, kv_dim, num_heads, num_latents, top_k=None,
                 widening_factor=1):
        super().__init__()
        self.top_k = top_k
        self.ln1 = nn.LayerNorm(2*q_dim)
        self.ln2 = nn.LayerNorm(q_dim)
        self.mlp = PerceiverMLP(q_dim, widening_factor=widening_factor)

        self.to_logits = nn.Conv2d(kv_dim, num_latents, 1)
        self.to_values = nn.Conv2d(kv_dim, q_dim, 1)
        self.projection = nn.Linear(2*q_dim, q_dim)

    def forward(self, xlr, xhr, return_attn=False):
        read = sparse_mixture_2d(
            self.to_logits(xhr),
            self.to_values(xhr),
            k=self.top_k,
            return_attn=return_attn,
        )
        if return_attn:
            read, attn = read

        xlr = xlr + self.projection(self.ln1(torch.cat((xlr, read), dim=-1)))
        xlr = xlr + self.mlp(self.ln2(xlr))

        if not return_attn:
            return xlr
        else:
            return xlr, attn


class LocalReadLayer(nn.Module):
    def __init__(self, q_dim, kv_dim, num_heads, num_latents, top_k=None,
                 read_heads=8, widening_factor=1):
        super().__init__()
        self.num_latents_sqrt = int(np.sqrt(num_latents))
        assert self.num_latents_sqrt**2 == num_latents, num_latents

        self.ln1 = nn.LayerNorm(q_dim)
        self.ln2 = nn.LayerNorm(q_dim)
        self.mlp = PerceiverMLP(q_dim, widening_factor=widening_factor)

        self.to_logits = nn.Conv2d(kv_dim, read_heads, 1)
        self.to_values = nn.Conv2d(kv_dim, q_dim, 1)
        self.projection = nn.Linear(q_dim, q_dim)

    def forward(self, xlr, xhr, return_attn=False):
        read = local_mixture_2d(
            self.to_logits(xhr),
            self.to_values(xhr),
            return_attn=return_attn,
            winh=self.num_latents_sqrt,
            winw=self.num_latents_sqrt,
        )
        if return_attn:
            read, attn = read

        xlr = xlr + self.projection(self.ln1(read))
        xlr = xlr + self.mlp(self.ln2(xlr))

        if not return_attn:
            return xlr
        else:
            return xlr, attn


class LocalWriteLayer(nn.Module):
    def __init__(self, q_dim, kv_dim, num_heads, num_latents, top_k=None,
                 write_heads=8, widening_factor=1):
        super().__init__()
        self.num_latents_sqrt = int(np.sqrt(num_latents))
        assert self.num_latents_sqrt**2 == num_latents, num_latents

        self.top_k = top_k
        self.ln1 = nn.BatchNorm2d(q_dim)
        self.ln2 = nn.BatchNorm2d(q_dim)
        self.mlp = PerceiverMLPConv(q_dim, widening_factor=widening_factor)

        self.to_logits = nn.Conv2d(kv_dim, write_heads, 1)
        self.to_values = nn.Linear(kv_dim, q_dim)
        self.projection = nn.Conv2d(q_dim, q_dim, 1)

    def forward(self, xhr, xlr, return_attn=False):
        read = local_mixture_2d_transposed(
            self.to_logits(xhr),
            self.to_values(xlr),
            return_attn=return_attn,
            winh=self.num_latents_sqrt,
            winw=self.num_latents_sqrt,
        )
        if return_attn:
            read, attn = read

        xhr = xhr + self.projection(self.ln1(read))
        xhr = xhr + self.mlp(self.ln2(xhr))

        if not return_attn:
            return xhr
        else:
            return xhr, attn


class FastWriteLayer(nn.Module):
    def __init__(self, q_dim, kv_dim, num_heads, num_latents, top_k=None,
                 widening_factor=1):
        super().__init__()
        self.top_k = top_k
        self.ln1 = nn.BatchNorm2d(2*q_dim)
        self.ln2 = nn.BatchNorm2d(q_dim)
        self.mlp = PerceiverMLPConv(q_dim, widening_factor=widening_factor)

        self.to_logits = nn.Conv2d(kv_dim, num_latents, 1)
        self.to_values = nn.Linear(kv_dim, q_dim)
        self.projection = nn.Conv2d(2*q_dim, q_dim, 1)

    def forward(self, xhr, xlr, return_attn=False):
        read = sparse_mixture_2d_transposed(
            self.to_logits(xhr),
            self.to_values(xlr),
            return_attn=return_attn,
        )
        if return_attn:
            read, attn = read

        xhr = xhr + self.projection(self.ln1(torch.cat((xhr, read), dim=1)))
        xhr = xhr + self.mlp(self.ln2(xhr))

        if not return_attn:
            return xhr
        else:
            return xhr, attn


class WriteLayer(nn.Module):
    def __init__(self, q_dim, kv_dim, num_heads, widening_factor=1):
        super().__init__()
        self.attn = PerceiverSelfAttention(
            attention_probs_dropout_prob=0,
            is_cross_attention=True,
            num_heads=num_heads,
            q_dim=q_dim,
            kv_dim=kv_dim,
        )
        self.expand = nn.Linear(q_dim, 2*q_dim)
        self.projection = nn.Conv2d(q_dim, q_dim, 1)
        self.ln_premlp = nn.BatchNorm2d(q_dim)
        self.mlp = PerceiverMLPConv(q_dim, widening_factor=widening_factor)

    def forward(self, xhr, xlr):
        xres = xhr

        B, C, H, W = xhr.shape
        B_, N, D = xlr.shape
        assert B==B_

        windows = int(np.sqrt(N))
        assert windows**2 == N, (N, windows)

        assert H%windows == 0 and W%windows == 0, (H, W, windows)

        # convert hr to sequence
        xhr = rearrange(xhr, "b c (winh pixh) (winw pixw) -> (b winh winw) (pixh pixw) c",
                        b=B, c=C, winh=windows, pixh=H//windows, winw=windows, pixw=W//windows)
        xlr = rearrange(self.expand(xlr), "b n (x d) -> (b n) x d",
                        b=B, n=N, x=2, d=D)

        xhr = self.attn(xhr, xlr)

        xhr = rearrange(xhr, "(b winh winw) (pixh pixw) c -> b c (winh pixh) (winw pixw)",
                        b=B, winh=windows, pixh=H//windows, winw=windows, pixw=W//windows)
        xhr = self.projection(xhr)
        xhr = xhr + xres

        xres = xhr
        xhr = self.ln_premlp(xhr)
        xhr = self.mlp(xhr)
        xhr = xhr + xres

        return xhr


class DeformableWriteLayer(nn.Module):
    def __init__(self, in_chn):
        super().__init__()
        self.q = nn.Conv2d(2*in_chn+1, in_chn, 1)
        self.v = nn.Conv2d(in_chn+1, in_chn, 1)


    def forward(self, x, srcs, flows, flow_errs):
        B, T, C, H, W = srcs.shape
        B_, C_, H_, W_ = x.shape
        assert B==B_ and C==C_ and H==H_ and W==W_
        B_, T_, _, _, _ = flows.shape
        assert B==B_ and T==T_
        B_, T_, _, _, _ = flow_errs.shape
        assert B==B_ and T==T_

        flows = rearrange(flows, "b t c h w -> (b t) c h w")
        flow_errs = rearrange(flow_errs, "b t c h w -> (b t) c h w")
        srcs = rearrange(srcs, "b t c h w -> (b t) c h w")
        srcs = resized_flow_sample(srcs, flows, align_corners=True)

        xr = repeat(x, "b c h w -> (b t) c h w", b=B, t=T, h=H, w=W)
        flow_errs = torch.nn.functional.interpolate(
            flow_errs, size=(H, W), mode="bilinear", align_corners=True
        )
        srcs = torch.cat((srcs, flow_errs, xr), dim=1)

        q = self.q(srcs)
        v = self.v(srcs[:,:C+1])
        q = rearrange(q, "(b t) c h w -> b t c h w", b=B, t=T)
        v = rearrange(v, "(b t) c h w -> b t c h w", b=B, t=T)

        x = x + torch.sum(
            torch.softmax(q, dim=1)*v,
            dim=1,
        )
        return x


def generate_fourier_features(pos, num_bands, max_resolution=(224, 224), concat_pos=True, sine_only=False):
    """
    Generate a Fourier frequency position encoding with linear spacing.

    Args:
      pos (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`):
        The Tensor containing the position of n points in d dimensional space.
      num_bands (`int`):
        The number of frequency bands (K) to use.
      max_resolution (`Tuple[int]`, *optional*, defaults to (224, 224)):
        The maximum resolution (i.e. the number of pixels per dim). A tuple representing resolution for each dimension.
      concat_pos (`bool`, *optional*, defaults to `True`):
        Whether to concatenate the input position encoding to the Fourier features.
      sine_only (`bool`, *optional*, defaults to `False`):
        Whether to use a single phase (sin) or two (sin/cos) for each frequency band.

    Returns:
      `torch.FloatTensor` of shape `(batch_size, sequence_length, n_channels)`: The Fourier position embeddings. If
      `concat_pos` is `True` and `sine_only` is `False`, output dimensions are ordered as: [dim_1, dim_2, ..., dim_d,
      sin(pi*f_1*dim_1), ..., sin(pi*f_K*dim_1), ..., sin(pi*f_1*dim_d), ..., sin(pi*f_K*dim_d), cos(pi*f_1*dim_1),
      ..., cos(pi*f_K*dim_1), ..., cos(pi*f_1*dim_d), ..., cos(pi*f_K*dim_d)], where dim_i is pos[:, i] and f_k is the
      kth frequency band.
    """

    batch_size = pos.shape[0]

    min_freq = 1.0
    # Nyquist frequency at the target resolution:
    freq_bands = torch.stack(
        [torch.linspace(start=min_freq, end=res / 2, steps=num_bands) for res in max_resolution], dim=0
    )

    # Get frequency bands for each spatial dimension.
    # Output is size [n, d * num_bands]
    per_pos_features = pos[0, :, :][:, :, None] * freq_bands[None, :, :]
    per_pos_features = torch.reshape(per_pos_features, [-1, np.prod(per_pos_features.shape[1:])])

    if sine_only:
        # Output is size [n, d * num_bands]
        per_pos_features = torch.sin(np.pi * (per_pos_features))
    else:
        # Output is size [n, 2 * d * num_bands]
        per_pos_features = torch.cat(
            [torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)], dim=-1
        )
    # Concatenate the raw input positions.
    if concat_pos:
        # Adds d bands to the encoding.
        per_pos_features = torch.cat([pos, per_pos_features.expand(batch_size, -1, -1)], dim=-1)
    return per_pos_features


def build_linear_positions(index_dims, output_range=(-1.0, 1.0)):
    """
    Generate an array of position indices for an N-D input array.

    Args:
      index_dims (`List[int]`):
        The shape of the index dimensions of the input array.
      output_range (`Tuple[float]`, *optional*, defaults to `(-1.0, 1.0)`):
        The min and max values taken by each input index dimension.

    Returns:
      `torch.FloatTensor` of shape `(index_dims[0], index_dims[1], .., index_dims[-1], N)`.
    """

    def _linspace(n_xels_per_dim):
        return torch.linspace(start=output_range[0], end=output_range[1], steps=n_xels_per_dim, dtype=torch.float32)

    dim_ranges = [_linspace(n_xels_per_dim) for n_xels_per_dim in index_dims]
    array_index_grid = torch.meshgrid(*dim_ranges)

    return torch.stack(array_index_grid, dim=-1)


def _check_or_build_spatial_positions(pos, index_dims, batch_size):
    """
    Checks or builds spatial position features (x, y, ...).

    Args:
      pos (`torch.FloatTensor`):
        None, or an array of position features. If None, position features are built. Otherwise, their size is checked.
      index_dims (`List[int]`):
        An iterable giving the spatial/index size of the data to be featurized.
      batch_size (`int`):
        The batch size of the data to be featurized.

    Returns:
        `torch.FloatTensor` of shape `(batch_size, prod(index_dims))` an array of position features.
    """
    if pos is None:
        pos = build_linear_positions(index_dims)
        pos = torch.broadcast_to(pos[None], (batch_size,) + pos.shape)
        pos = torch.reshape(pos, [batch_size, np.prod(index_dims), -1])
    else:
        # Just a warning label: you probably don't want your spatial features to
        # have a different spatial layout than your pos coordinate system.
        # But feel free to override if you think it'll work!
        if pos.shape[-1] != len(index_dims):
            raise ValueError("Spatial features have the wrong number of dimensions.")
    return pos


class PerceiverFourierPositionEncoding(nn.Module):
    """Fourier (Sinusoidal) position encoding."""

    def __init__(self, num_bands, max_resolution, concat_pos=True, sine_only=False):
        super().__init__()
        self.num_bands = num_bands
        self.max_resolution = max_resolution
        self.concat_pos = concat_pos
        self.sine_only = sine_only

    @property
    def num_dimensions(self) -> int:
        return len(self.max_resolution)

    def output_size(self):
        """Returns size of positional encodings last dimension."""
        num_dims = len(self.max_resolution)
        encoding_size = self.num_bands * num_dims
        if not self.sine_only:
            encoding_size *= 2
        if self.concat_pos:
            encoding_size += self.num_dimensions

        return encoding_size

    def forward(self, index_dims, batch_size, device, pos=None):
        pos = _check_or_build_spatial_positions(pos, index_dims, batch_size)
        fourier_pos_enc = generate_fourier_features(
            pos,
            num_bands=self.num_bands,
            max_resolution=self.max_resolution,
            concat_pos=self.concat_pos,
            sine_only=self.sine_only,
        ).to(device)
        return fourier_pos_enc



class GIBlock(nn.Module):
    def __init__(self, q_dim, kv_dim, num_heads, num_latents,
                 read_write_heads, block_cls, block_kwargs):
        super().__init__()
        self.read = LocalReadLayer(q_dim, kv_dim, num_heads,
                                   num_latents=num_latents,
                                   read_heads=read_write_heads)
        self.write = LocalWriteLayer(kv_dim, q_dim, num_heads,
                                     num_latents=num_latents,
                                     write_heads=read_write_heads)
        self.attn1 = AttentionLayer(q_dim, num_heads)
        self.attn2 = AttentionLayer(q_dim, num_heads)
        self.block1 = block_cls(**block_kwargs)
        self.block2 = block_cls(**block_kwargs)

    def forward(self, x, return_attn=False, kv=None, return_kv=False):
        assert not (return_attn and return_kv)
        assert isinstance(x, tuple) and len(x)==2
        xhr = x[0]
        B, C, H, W = xhr.shape
        xlr = x[1]
        B_, N, D = xlr.shape

        xlr = self.read(xlr, xhr, return_attn=return_attn)
        if return_attn:
            xlr, read_attn = xlr
        xlr = self.attn1(xlr, kv=kv, return_kv=return_kv)
        if return_kv:
            xlr, kv = xlr
        xlr = self.attn2(xlr)

        xhr = self.block1(xhr)
        xhr = self.block2(xhr)

        xhr = self.write(xhr, xlr, return_attn=return_attn)
        if return_attn:
            xhr, write_attn = xhr

        if return_attn:
            return (xhr, xlr), (read_attn, write_attn)

        if return_kv:
            return (xhr, xlr), kv

        return xhr, xlr


class GIResNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 resnet_conv_kwargs={},
                 add_out_act=True, max_features=1024, num_latents_sqrt=16,
                 num_groups=4, pre_lama=0, post_lama=0, read_write_heads=8,
                 block_type="conv", num_heads=8, is_deformable=False,
                 is_crossattending=False, n_cross=1000):
        assert (n_blocks >= 0)
        super().__init__()
        self.num_latents_sqrt = num_latents_sqrt

        self.encode_modules = list()
        self.bottleneck_modules = list()
        self.decode_modules = list()

        model = [nn.ReflectionPad2d(3),
                 CONV_NORM_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                               activation_layer=activation_layer)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [CONV_NORM_ACT(min(max_features, ngf * mult),
                                    min(max_features, ngf * mult * 2),
                                    kernel_size=3, stride=2, padding=1,
                                    norm_layer=norm_layer,
                                    activation_layer=activation_layer)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### Pre-Lama blocks
        for i in range(pre_lama):
            cur_resblock = FFCResnetBlock(
                feats_num_bottleneck,
                padding_type=padding_type,
                activation_layer=activation_layer,
                norm_layer=norm_layer, inline=True,
                **resnet_conv_kwargs)
            model += [cur_resblock]

        # jinx
        num_latents_sqrt = self.num_latents_sqrt
        num_latents = num_latents_sqrt**2
        model += [LatentInitLayer(in_channels=feats_num_bottleneck,
                                  num_latents_sqrt=num_latents_sqrt, d_latents=512)]

        # configure inner conv blocks
        if block_type == "conv":
            block_cls = RESIDUAL_CONV_NORM_ACT
            block_kwargs = dict(
                in_channels=feats_num_bottleneck,
                out_channels=feats_num_bottleneck,
                kernel_size=3, padding=1,
                padding_type=padding_type,
                activation_layer=activation_layer,
                norm_layer=norm_layer,
                groups=num_groups,
            )
        elif block_type == "ffc":
            block_cls = FFCResnetBlock
            block_kwargs = dict(
                dim=feats_num_bottleneck,
                padding_type=padding_type,
                activation_layer=activation_layer,
                norm_layer=norm_layer, inline=True,
                **resnet_conv_kwargs
            )
        else:
            raise ValueError(block_type)

        self.encode_modules += model
        bottleneck_start = len(model)

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = GIBlock(
                q_dim=feats_num_bottleneck,
                kv_dim=feats_num_bottleneck,
                num_heads=num_heads,
                num_latents=num_latents,
                read_write_heads=read_write_heads,
                block_cls=block_cls,
                block_kwargs=block_kwargs,
            )
            model += [cur_resblock]

        self.bottleneck_modules = model[bottleneck_start:]
        decode_start = len(model)

        model += [DropLatentLayer()]

        ### Post-Lama blocks
        for i in range(post_lama):
            cur_resblock = FFCResnetBlock(
                feats_num_bottleneck,
                padding_type=padding_type,
                activation_layer=activation_layer,
                norm_layer=norm_layer, inline=True,
                **resnet_conv_kwargs)
            model += [cur_resblock]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))

        self.decode_modules = model[decode_start:]

        self.model = nn.Sequential(*model)

        self.is_crossattending = is_crossattending
        self.is_deformable = is_deformable
        self.n_cross = n_cross
        if self.is_deformable:
            self.deformable_writes = nn.ModuleList()
            for i in range(n_blocks):
                self.deformable_writes.append(DeformableWriteLayer(feats_num_bottleneck))
                if i + 1 >= self.n_cross:
                    break

        self.apply(self._init_weights)


    def forward(self, input, srcs, return_attn=False, flows=None,
                flow_errs=None):
        B, T, C, H, W = srcs.shape
        B_, C_, H_, W_ = input.shape
        assert B==B_ and C==C_ and H==H_ and W==W_

        if self.is_deformable:
            assert flows is not None and flow_errs is not None

        input = torch.cat((input.unsqueeze(1), srcs), dim=1)
        input = rearrange(input, "b t c h w -> (b t) c h w")

        if return_attn:
            attns = list()

        for module in self.encode_modules:
            input = module(input)

        a, b = input
        a = rearrange(a, "(b t) c h w -> b t c h w", b=B, t=T+1)
        b = rearrange(b, "(b t) n d -> b t n d", b=B, t=T+1)
        input = a[:, 0], b[:, 0]
        a = a[:, 1:]
        a = rearrange(a, "b t c h w -> (b t) c h w")
        b = b[:, 1:]
        b = rearrange(b, "b t n d -> (b t) n d")
        input_srcs = a, b

        latent_kv = list()
        src_lifs = list()
        for i, module in enumerate(self.bottleneck_modules):
            input_srcs, kv = module(input_srcs, return_attn=False,
                                    return_kv=True)
            kv = [rearrange(kv_el, "(b t) heads n d -> b heads (t n) d", b=B, t=T)
                  for kv_el in kv]
            latent_kv.append(kv)
            src_lif = input_srcs[0]
            src_lif = rearrange(src_lif, "(b t) c h w -> b t c h w", b=B, t=T)
            src_lifs.append(src_lif)
            if i+1 >= self.n_cross:
                break
            

        for i, module in enumerate(self.bottleneck_modules):
            if self.is_crossattending and (i<self.n_cross):
                kv = latent_kv.pop(0)
            else:
                kv = None

            input = module(input, return_attn=return_attn, kv=kv)

            if return_attn:
                input, attn = input
                attns.append(attn)

            if self.is_deformable and (i<self.n_cross):
                dst_lif, dst_gif = input
                dst_lif = self.deformable_writes[i](dst_lif, src_lifs.pop(0),
                                                    flows, flow_errs)
                input = dst_lif, dst_gif


        for module in self.decode_modules:
            input = module(input)

        if not return_attn:
            return input
        else:
            return input, attns


    def _init_weights(self, module):
        """Initialize the weights"""
        initializer_range = 0.02
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif hasattr(module, "latents"):
            module.latents.data.normal_(mean=0.0, std=initializer_range)
        elif hasattr(module, "position_embeddings") and isinstance(module, PerceiverTrainablePositionEncoding):
            module.position_embeddings.data.normal_(mean=0.0, std=initializer_range)
        elif isinstance(module, nn.ParameterDict):
            for modality in module.keys():
                module[modality].data.normal_(mean=0.0, std=initializer_range)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class FFCNLayerDiscriminator(BaseDiscriminator):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, max_features=512,
                 init_conv_kwargs={}, conv_kwargs={}):
        super().__init__()
        self.n_layers = n_layers

        def _act_ctor(inplace=True):
            return nn.LeakyReLU(negative_slope=0.2, inplace=inplace)

        kw = 3
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[FFC_BN_ACT(input_nc, ndf, kernel_size=kw, padding=padw, norm_layer=norm_layer,
                                activation_layer=_act_ctor, **init_conv_kwargs)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, max_features)

            cur_model = [
                FFC_BN_ACT(nf_prev, nf,
                           kernel_size=kw, stride=2, padding=padw,
                           norm_layer=norm_layer,
                           activation_layer=_act_ctor,
                           **conv_kwargs)
            ]
            sequence.append(cur_model)

        nf_prev = nf
        nf = min(nf * 2, 512)

        cur_model = [
            FFC_BN_ACT(nf_prev, nf,
                       kernel_size=kw, stride=1, padding=padw,
                       norm_layer=norm_layer,
                       activation_layer=lambda *args, **kwargs: nn.LeakyReLU(*args, negative_slope=0.2, **kwargs),
                       **conv_kwargs),
            ConcatTupleLayer()
        ]
        sequence.append(cur_model)

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))

    def get_all_activations(self, x):
        res = [x]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]

    def forward(self, x):
        act = self.get_all_activations(x)
        feats = []
        for out in act[:-1]:
            if isinstance(out, tuple):
                if torch.is_tensor(out[1]):
                    out = torch.cat(out, dim=1)
                else:
                    out = out[0]
            feats.append(out)
        return act[-1], feats

