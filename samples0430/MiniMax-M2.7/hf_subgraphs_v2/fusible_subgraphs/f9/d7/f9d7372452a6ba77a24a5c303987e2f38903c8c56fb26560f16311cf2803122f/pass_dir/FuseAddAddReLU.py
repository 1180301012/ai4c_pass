import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_add_relu_kernel(
    in_0_ptr, in_2_ptr, in_3_ptr, out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all three inputs
    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_2 = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    in_3 = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)
    
    # Fused: in_3 + in_0 + in_2
    tmp = in_3 + in_0 + in_2
    
    # Apply ReLU (in-place semantics)
    result = tl.where(tmp > 0, tmp, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_add_add_relu(in_0, in_2, in_3):
    """
    Fused kernel that computes: relu(in_3 + in_0 + in_2)
    Replaces the original pattern:
      in_3 += in_0
      in_4 = in_3
      in_4 += in_2
      tmp_2 = relu(tmp_0)
    """
    N = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(in_0)

    fused_add_add_relu_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_2_ptr=in_2,
        in_3_ptr=in_3,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern:
    in_3 += in_0; in_4 = in_3; in_3 = in_0 = None
    in_4 += in_2; tmp_0 = in_4; in_4 = in_2 = None
    tmp_2 = relu(tmp_0, inplace=True); tmp_0 = None
    tmp_3 = in_1.view(1, 32, -1); in_1 = None
    tmp_4 = tmp_3.permute(0, 2, 1); tmp_3 = None
    return (tmp_2, tmp_4)
    """
    # First add: in_4 = in_3 + in_0
    in_4 = in_3 + in_0
    
    # Second add: tmp_0 = in_4 + in_2
    tmp_0 = in_4 + in_2
    
    # ReLU
    tmp_2 = torch.nn.functional.relu(tmp_0, inplace=True)
    
    # View and permute
    tmp_3 = in_1.view(1, 32, -1)
    tmp_4 = tmp_3.permute(0, 2, 1)
    
    return (tmp_2, tmp_4)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_add_add_relu