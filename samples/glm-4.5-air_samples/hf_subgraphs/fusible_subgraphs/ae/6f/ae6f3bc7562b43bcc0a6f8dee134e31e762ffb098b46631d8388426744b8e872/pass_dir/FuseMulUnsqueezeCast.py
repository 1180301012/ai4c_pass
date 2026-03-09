import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def fused_mul_unsqueeze_kernel(
    input_ptr,
    mask_ptr,
    output_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: unsqueeze(-1) + mul + to(float32)
    
    This kernel fuses:
    1. mask.unsqueeze(-1) - expands mask from [batch, seq] to [batch, seq, 1]
    2. input * mask - element-wise multiplication with broadcasting
    3. to(float32) - redundant, input is already float32
    
    Each program processes a contiguous block of hidden_dim elements
    for one (batch, seq) position.
    """
    # Calculate position in the flattened tensor
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load input tensor (tmp_7 after add)
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Load mask value - mask is 2D [batch, seq], need to compute the index
    # For the flattened view, each "row" in the flattened tensor corresponds to
    # one (batch_idx, seq_idx) position. We load the same mask value for all
    # hidden dimension elements at this position.
    mask_val = tl.load(mask_ptr + pid)  # Load once per (batch, seq) position
    
    # Compute multiplication with broadcast - the key fusion
    # This replaces: mask.unsqueeze(-1); result = input * mask
    output_vals = input_vals * mask_val
    
    # Store result (replaces both tmp_9 and tmp_10 since cast is no-op)
    tl.store(output_ptr + offsets, output_vals, mask=mask)


@torch.fx.wrap
def fused_mul_unsqueeze_wrapper(tmp_7, in_0):
    """
    Optimized wrapper that fuses unsqueeze + mul + redundant cast.
    
    Input:
      tmp_7: tensor of shape [batch, seq, hidden_dim] - result of (div_result + embedding)
      in_0: mask tensor of shape [batch, seq]
    
    Output:
      tmp_10: tensor of shape [batch, seq, hidden_dim] - result after mul + cast
    """
    # Get shapes
    batch_size = tmp_7.shape[0]
    seq_len = tmp_7.shape[1]
    hidden_dim = tmp_7.shape[2]
    
    # Flatten: [batch, seq, hidden] -> [batch * seq, hidden]
    tmp_7_flat = tmp_7.view(-1, hidden_dim)
    
    # mask stays 2D: [batch, seq] - we'll index directly
    # Actually we need to flatten it too for the kernel
    in_0_flat = in_0.view(-1)
    
    # Allocate output
    output_flat = torch.empty_like(tmp_7_flat)
    
    # Launch kernel: one program per (batch, seq) position
    N = hidden_dim
    grid = (batch_size * seq_len,)
    
    fused_mul_unsqueeze_kernel[grid](
        input_ptr=tmp_7_flat,
        mask_ptr=in_0_flat,
        output_ptr=output_flat,
        N=N,
    )
    
    # Reshape back to [batch, seq, hidden]
    return output_flat.view(batch_size, seq_len, hidden_dim)


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Match the full computation pattern and return both outputs.
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3
    tmp_4 = in_5 / in_4
    tmp_5 = tmp_4.to(torch.float32)
    tmp_4 = None
    tmp_6 = torch.nn.functional.embedding(in_6, tmp_1, 1, None, 2.0, False, False)
    tmp_1 = None
    tmp_7 = tmp_5 + tmp_6
    tmp_5 = tmp_6 = None
    tmp_8 = tmp_0.unsqueeze(-1)
    tmp_0 = None
    tmp_9 = tmp_7 * tmp_8
    tmp_7 = tmp_8 = None
    tmp_10 = tmp_9.to(torch.float32)
    tmp_9 = None
    tmp_11 = torch.nn.functional.layer_norm(tmp_10, (1280,), tmp_3, tmp_2, 1e-12)
    tmp_3 = tmp_2 = None
    return (tmp_10, tmp_11)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Extract the arguments needed for the replacement.
    
    We need:
    - tmp_7 (result of div + embedding + add)
    - in_0 (the mask)
    
    These will be computed in the original graph, and our replacement
    will optimize the mul + unsqueeze + cast part.
    """
    # Compute intermediate values needed for replacement
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3
    tmp_4 = in_5 / in_4
    tmp_5 = tmp_4.to(torch.float32)
    tmp_4 = None
    tmp_6 = torch.nn.functional.embedding(in_6, tmp_1, 1, None, 2.0, False, False)
    tmp_1 = None
    tmp_7 = tmp_5 + tmp_6
    
    # Return (tmp_7, in_0) for the fused kernel
    return (tmp_7, in_0)


def replacement_func():
    """
    Returns the replacement function that fuses the operations.
    """
    return fused_mul_unsqueeze_wrapper