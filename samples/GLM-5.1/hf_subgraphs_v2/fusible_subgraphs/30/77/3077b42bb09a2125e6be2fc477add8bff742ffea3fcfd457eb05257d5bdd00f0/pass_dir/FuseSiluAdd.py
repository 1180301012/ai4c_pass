import torch
import triton
import triton.language as tl


def pattern(in_0: torch.Tensor, in_1: torch.Tensor):
    tmp_0 = torch.nn.functional.silu(in_1, inplace=True)
    tmp_1 = tmp_0 + in_0
    return (tmp_1,)


def replacement_args(in_0: torch.Tensor, in_1: torch.Tensor):
    return (in_0, in_1)


@triton.jit
def silu_add_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs
    in0 = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
    in1 = tl.load(in1_ptr + offsets, mask=mask, other=0.0)

    # Compute fused silu(in1) + in0
    # silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    sigmoid_in1 = tl.sigmoid(in1)
    silu_val = in1 * sigmoid_in1
    out = silu_val + in0

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def silu_add_kernel_fast(
    in0_ptr,
    in1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs
    in0 = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
    in1 = tl.load(in1_ptr + offsets, mask=mask, other=0.0)

    # Compute fused silu(in1) + in0 using exp for faster sigmoid
    # silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    # = x * exp(x) / (1 + exp(x))
    neg_in1 = -in1
    exp_neg_in1 = tl.exp(neg_in1)
    sigmoid_in1 = 1.0 / (1.0 + exp_neg_in1)
    silu_val = in1 * sigmoid_in1
    out = silu_val + in0

    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_silu_add(in_0: torch.Tensor, in_1: torch.Tensor):
    out = torch.empty_like(in_0)
    n_elements = in_0.numel()
    # Use larger block size for better occupancy on larger tensors
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    silu_add_kernel[grid](
        in0_ptr=in_0,
        in1_ptr=in_1,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def replacement_func():
    return fused_silu_add