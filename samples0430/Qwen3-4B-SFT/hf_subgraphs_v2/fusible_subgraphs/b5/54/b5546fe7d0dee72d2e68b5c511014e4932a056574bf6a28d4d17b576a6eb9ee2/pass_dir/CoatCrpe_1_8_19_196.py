"""
Pass: coat_crpe_1_8_19_196
Fuses: cat -> reshape(1,8,19,H*W) -> transpose -> mul -> pad -> scale+add -> transpose -> reshape(1,H+1,8*C)
h_base=38, C=57, H=14, W=14
"""
import torch
import triton
from pass_dir.coat_crpe_shared_kernel import coat_crpe_fused_kernel


def pattern(cat_out, in_4, in_6, scale):
    tmp_4 = cat_out.reshape(1, 8, 19, 196)
    tmp_5 = tmp_4.transpose(-1, -2)
    tmp_6 = in_6 * tmp_5
    tmp_7 = torch.nn.functional.pad(tmp_6, (0, 0, 1, 0, 0, 0), 'constant', None)
    tmp_8 = scale * in_4
    tmp_9 = tmp_8 + tmp_7
    tmp_10 = tmp_9.transpose(1, 2)
    tmp_11 = tmp_10.reshape(1, 197, 152)
    return tmp_11


def replacement_args(cat_out, in_4, in_6, scale):
    return (cat_out, in_4, in_6, scale)


@torch.fx.wrap
def _crpe_fused_1_8_19_196(cat_out, in_4, in_6, scale):
    H = 14
    W = 14
    C_in2 = 38   # h_base (= N1 channels in in_2)
    C_in3 = 57   # channels in in_3
    C_conv = 57  # channels in conv_out
    C_out  = 152  # = 38 + 57 + 57
    n_elements = 1 * (H + 1) * C_out   # 24336

    # Steal N1/N2/N3 from cat_out shape: [1, N1+N2+C_conv, H, W]
    N1 = cat_out.shape[1] - C_conv   # 38
    N2 = cat_out.shape[1] - C_conv - N1   # 57

    in_2   = cat_out[:, :N1]           # [1, N1, H, W]
    in_3   = cat_out[:, N1:N1+N2]      # [1, N2, H, W]
    conv   = cat_out[:, N1+N2:]        # [1, C_conv, H, W]

    output = torch.empty(1, H + 1, C_out, dtype=in_4.dtype, device=in_4.device)

    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    coat_crpe_fused_kernel[grid](
        in_2, in_3, conv, scale, in_4, in_6, output,
        N1, N2, C_conv, C_out, H, W, C_in2,
        n_elements,
    )
    return output


def replacement_func():
    return _crpe_fused_1_8_19_196