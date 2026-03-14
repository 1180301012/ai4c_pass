import torch
import triton
import triton.language as tl

# Pattern for flatten -> transpose -> layer_norm(320)
def pattern(ln_bias, ln_weight, conv_out):
    flattened = conv_out.flatten(2)
    transposed = flattened.transpose(1, 2)
    normalized = torch.nn.functional.layer_norm(transposed, (320,), ln_weight, ln_bias, 1e-05)
    return normalized

def replacement_args(ln_bias, ln_weight, conv_out):
    return (ln_bias, ln_weight, conv_out)

@triton.jit
def layer_norm_kernel_320(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, D, eps,
    stride_n,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < D
    
    x = tl.load(x_ptr + row_idx * stride_n + offsets, mask=mask, other=0.0).to(tl.float32)
    D_float = D.to(tl.float32)
    mean = tl.sum(x, axis=0) / D_float
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / D_float
    std = tl.sqrt(var + eps)
    x_norm = diff / std
    
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = x_norm * weight + bias
    tl.store(out_ptr + row_idx * stride_n + offsets, out, mask=mask)

@torch.fx.wrap
def fused_flatten_transpose_layernorm_320(ln_bias, ln_weight, conv_out):
    flattened = conv_out.flatten(2)
    transposed = flattened.transpose(1, 2).contiguous()
    
    B, N_seq, D = transposed.shape
    out = torch.empty_like(transposed)
    BLOCK_SIZE = triton.next_power_of_2(D)
    
    grid = (B * N_seq,)
    layer_norm_kernel_320[grid](
        transposed, ln_weight, ln_bias, out,
        N=B * N_seq, D=D, eps=1e-05,
        stride_n=D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

def replacement_func():
    return fused_flatten_transpose_layernorm_320