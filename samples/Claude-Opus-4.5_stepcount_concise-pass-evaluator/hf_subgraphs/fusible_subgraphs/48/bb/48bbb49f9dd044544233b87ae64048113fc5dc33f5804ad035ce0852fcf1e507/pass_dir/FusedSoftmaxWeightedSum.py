import torch
import triton
import triton.language as tl


# Pattern matching function - matches the computation in model.py
def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.softmax(in_2, dim=2)
    tmp_3 = tmp_2.reshape(-1, 17, 64, 64)
    tmp_4 = tmp_3.mul(in_0)
    tmp_5 = tmp_4.reshape(256, 17, -1)
    tmp_6 = torch.sum(tmp_5, dim=2, keepdim=True)
    tmp_7 = tmp_3.mul(in_1)
    tmp_8 = tmp_7.reshape(256, 17, -1)
    tmp_9 = torch.sum(tmp_8, dim=2, keepdim=True)
    tmp_10 = torch.cat([tmp_6, tmp_9], dim=-1)
    return (tmp_3, tmp_10)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_softmax_weighted_sum_kernel(
    in_ptr,           # Input logits [batch, 17, 4096]
    x_weights_ptr,    # X weights [64]
    y_weights_ptr,    # Y weights [64]
    out_softmax_ptr,  # Output softmax [batch, 17, 64, 64]
    out_coords_ptr,   # Output coords [batch, 17, 2]
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, keypoint) pair
    pid = tl.program_id(0)
    
    # Calculate offsets
    in_offset = pid * 4096
    
    # First pass: find max for numerical stability
    max_val = tl.full([], float('-inf'), dtype=tl.float32)
    for block_start in range(0, 4096, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < 4096
        x = tl.load(in_ptr + in_offset + offsets, mask=mask, other=float('-inf'))
        block_max = tl.max(x, axis=0)
        max_val = tl.where(block_max > max_val, block_max, max_val)
    
    # Second pass: compute sum of exp(x - max)
    sum_exp = tl.full([], 0.0, dtype=tl.float32)
    for block_start in range(0, 4096, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < 4096
        x = tl.load(in_ptr + in_offset + offsets, mask=mask, other=float('-inf'))
        exp_x = tl.exp(x - max_val)
        masked_exp = tl.where(mask, exp_x, 0.0)
        sum_exp = sum_exp + tl.sum(masked_exp, axis=0)
    
    # Third pass: compute softmax, weighted sums, and store
    sum_x = tl.full([], 0.0, dtype=tl.float32)
    sum_y = tl.full([], 0.0, dtype=tl.float32)
    
    for block_start in range(0, 4096, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < 4096
        x = tl.load(in_ptr + in_offset + offsets, mask=mask, other=float('-inf'))
        
        # Compute softmax values
        softmax_vals = tl.exp(x - max_val) / sum_exp
        softmax_vals = tl.where(mask, softmax_vals, 0.0)
        
        # Store softmax values
        tl.store(out_softmax_ptr + in_offset + offsets, softmax_vals, mask=mask)
        
        # Compute weighted sums
        # For position i in [0, 4096): row = i // 64, col = i % 64
        col_idx = offsets % 64  # x index
        row_idx = offsets // 64  # y index
        
        x_weight = tl.load(x_weights_ptr + col_idx, mask=mask, other=0.0)
        y_weight = tl.load(y_weights_ptr + row_idx, mask=mask, other=0.0)
        
        sum_x = sum_x + tl.sum(softmax_vals * x_weight, axis=0)
        sum_y = sum_y + tl.sum(softmax_vals * y_weight, axis=0)
    
    # Store weighted sums (coords output is [batch*17, 2])
    out_coords_offset = pid * 2
    tl.store(out_coords_ptr + out_coords_offset, sum_x)
    tl.store(out_coords_ptr + out_coords_offset + 1, sum_y)


@torch.fx.wrap
def fused_softmax_weighted_sum(in_0, in_1, in_2):
    """
    Fused softmax + weighted sum computation.
    
    Args:
        in_0: X weights [1, 1, 1, 64]
        in_1: Y weights [1, 1, 64, 1]
        in_2: Input logits [batch, 17, 4096]
    
    Returns:
        out_softmax: Softmax output [batch, 17, 64, 64]
        out_coords: Weighted sum coordinates [batch, 17, 1, 2]
    """
    batch_size = in_2.shape[0]
    n_keypoints = in_2.shape[1]
    
    # Flatten weights to 1D for easier indexing
    x_weights = in_0.reshape(-1).contiguous()  # [64]
    y_weights = in_1.reshape(-1).contiguous()  # [64]
    
    # Ensure input is contiguous
    in_2_contig = in_2.contiguous()
    
    # Allocate outputs
    out_softmax = torch.empty(batch_size, n_keypoints, 64, 64, 
                               dtype=in_2.dtype, device=in_2.device)
    out_coords = torch.empty(batch_size * n_keypoints, 2, 
                              dtype=in_2.dtype, device=in_2.device)
    
    # Launch kernel - one block per (batch, keypoint) pair
    n_programs = batch_size * n_keypoints
    
    fused_softmax_weighted_sum_kernel[(n_programs,)](
        in_2_contig,
        x_weights,
        y_weights,
        out_softmax,
        out_coords,
        n_elements=4096,
    )
    
    # Reshape coords to match expected output shape [batch, 17, 1, 2]
    out_coords = out_coords.reshape(batch_size, n_keypoints, 1, 2)
    
    return (out_softmax, out_coords)


def replacement_func():
    return fused_softmax_weighted_sum