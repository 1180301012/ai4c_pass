import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.sigmoid()
    tmp_4 = in_2 * tmp_3
    return tmp_4

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_conv_sigmoid_mul_kernel(
    in_3_ptr,
    weights_ptr,
    bias_ptr,
    in_2_ptr,
    out_ptr,
    B, C_in, C_out, H, W,
    BLOCK_SIZE: tl.constexpr
):
    block_idx = tl.program_id(0)
    batch = block_idx // C_out
    c_out = block_idx % C_out

    bias_val = tl.load(bias_ptr + c_out)
    weights = tl.load(weights_ptr + c_out * C_in, mask=tl.arange(0, C_in) < C_in)
    in_3 = tl.load(in_3_ptr + batch * C_in, mask=tl.arange(0, C_in) < C_in)

    conv_val = bias_val
    for c_in in range(C_in):
        conv_val += in_3[c_in] * weights[c_in]
    sig_val = tl.sigmoid(conv_val)

    tid = tl.thread_idx(0)
    if tid < H * W:
        h = tid // W
        w = tid % W
        if h < H and w < W:
            idx = batch * C_out * H * W + c_out * H * W + h * W + w
            in_2_val = tl.load(in_2_ptr + idx)
            out_val = in_2_val * sig_val
            tl.store(out_ptr + idx, out_val)

@torch.fx.wrap
def fused_conv_sigmoid_mul(in_0, in_1, in_2, in_3):
    B = in_3.shape[0]
    C_in = in_3.shape[1]
    C_out = in_1.shape[0]
    H = in_2.shape[2]
    W = in_2.shape[3]
    
    weights = in_1.contiguous()
    bias = in_0.contiguous()
    in_3_contig = in_3.contiguous()
    in_2_contig = in_2.contiguous()
    
    out = torch.empty_like(in_2)
    
    grid = (B * C_out, )
    BLOCK_SIZE = 64
    fused_conv_sigmoid_mul_kernel[grid](
        in_3_contig, weights, bias, in_2_contig, out, B, C_in, C_out, H, W, BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return fused_conv_sigmoid_mul