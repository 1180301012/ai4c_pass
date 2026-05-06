import torch
import triton
import triton.language as tl

def pattern(a, b, c, d):
    return torch.nn.functional.layer_norm(a, b, c, d, 1e-05)

def replacement_args(a, b, c, d):
    return a, b, c, d

@triton.jit
def layer_norm_triton_kernel(
    input_ptr,
    output_ptr,
    weight_ptr,
    bias_ptr,
    normalized_shape,
    N,
    H_W,
    C,
    eps: tl.float32 = 1e-05,
    BLOCK_SIZE: tl.constexpr = 256,
):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    output_vals = input_vals * tl.load(weight_ptr) + tl.load(bias_ptr)
    tl.store(output_ptr + offsets, output_vals, mask=mask)

@torch.fx.wrap
def layer_norm_kernel(a, b, c, d):
    N, H_W, C = a.shape
    output = torch.empty_like(a)
    layer_norm_triton_kernel[(tl.cdiv(N, 256),)](
        input_ptr=a,
        output_ptr=output,
        weight_ptr=c,
        bias_ptr=d,
        normalized_shape=(C,),
        N=N,
        H_W=H_W,
        C=C,
    )
    return output

def replacement_func():
    return layer_norm_kernel