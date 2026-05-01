import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    gelu_out = torch.nn.functional.gelu(in_0, approximate='none')
    mul_out = gelu_out * in_1
    drop_out = torch.nn.functional.dropout(mul_out, 0.1, False, False)
    return drop_out

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_gelu_mul_dropout_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    in0 = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
    in1 = tl.load(in1_ptr + offsets, mask=mask, other=0.0)

    # Compute GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    sqrt2 = 1.41421356237
    erf_val = tl.math.erf(in0 / sqrt2)
    gelu_val = 0.5 * in0 * (1.0 + erf_val)

    # Multiply by in1 and by 0.9 (for dropout p=0.1)
    out = 0.9 * gelu_val * in1

    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_gelu_mul_dropout(in0, in1):
    n = in0.numel()
    BLOCK_SIZE = 1024
    grid_size = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(in0)
    fused_gelu_mul_dropout_kernel[(grid_size,)](
        in0_ptr=in0,
        in1_ptr=in1,
        out_ptr=out,
        n_elements=n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def replacement_func():
    return fused_gelu_mul_dropout