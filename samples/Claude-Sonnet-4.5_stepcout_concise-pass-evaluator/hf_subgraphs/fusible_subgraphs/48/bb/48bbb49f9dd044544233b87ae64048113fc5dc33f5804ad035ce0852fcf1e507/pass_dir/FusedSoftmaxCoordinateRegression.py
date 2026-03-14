import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """
    Pattern matching the entire coordinate regression computation:
    - Softmax on heatmaps
    - Reshape to spatial format
    - Weighted sum with x and y coordinates
    - Concatenate results
    """
    tmp_2 = torch.nn.functional.softmax(in_2, dim=2)
    tmp_3 = tmp_2.reshape(-1, 17, 64, 64)
    tmp_4 = tmp_3.mul(in_0)
    tmp_5 = tmp_4.reshape(tmp_4.shape[0], 17, -1)
    tmp_6 = torch.sum(tmp_5, dim=2, keepdim=True)
    tmp_7 = tmp_3.mul(in_1)
    tmp_8 = tmp_7.reshape(tmp_7.shape[0], 17, -1)
    tmp_9 = torch.sum(tmp_8, dim=2, keepdim=True)
    tmp_10 = torch.cat([tmp_6, tmp_9], dim=-1)
    return (tmp_3, tmp_10)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def fused_softmax_coord_kernel(
    input_ptr,
    linspace_x_ptr,
    linspace_y_ptr,
    softmax_out_ptr,
    coord_out_ptr,
    batch_size,
    num_keypoints,
    N,  # 4096
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes:
    1. Softmax along spatial dimension (N=4096)
    2. Weighted sum with x and y coordinates
    3. Output both softmax result and coordinates
    """
    # Each program handles one (batch, keypoint) pair
    pid = tl.program_id(0)
    batch_idx = pid // num_keypoints
    kp_idx = pid % num_keypoints
    
    # Calculate input/output offsets
    input_offset = (batch_idx * num_keypoints + kp_idx) * N
    softmax_offset = (batch_idx * num_keypoints + kp_idx) * N
    coord_offset = (batch_idx * num_keypoints + kp_idx) * 2
    
    # Load linspace_x and linspace_y (broadcasted to 4096 elements)
    # linspace_x: [1, 1, 1, 64] -> broadcast to [64, 64] -> flatten to [4096]
    # linspace_y: [1, 1, 64, 1] -> broadcast to [64, 64] -> flatten to [4096]
    
    # Step 1: Compute softmax
    # First pass: find max
    max_val = float('-inf')
    for offset in range(0, N, BLOCK_SIZE):
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        vals = tl.load(input_ptr + input_offset + offsets, mask=mask, other=float('-inf'))
        max_val = tl.maximum(max_val, tl.max(vals, axis=0))
    
    # Second pass: compute exp sum
    exp_sum = 0.0
    for offset in range(0, N, BLOCK_SIZE):
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        vals = tl.load(input_ptr + input_offset + offsets, mask=mask, other=0.0)
        exp_vals = tl.exp(vals - max_val)
        exp_sum += tl.sum(exp_vals, axis=0)
    
    # Third pass: normalize, store softmax, and compute weighted coordinates
    weighted_x_sum = 0.0
    weighted_y_sum = 0.0
    
    for offset in range(0, N, BLOCK_SIZE):
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        # Load input and compute softmax
        vals = tl.load(input_ptr + input_offset + offsets, mask=mask, other=0.0)
        softmax_vals = tl.exp(vals - max_val) / exp_sum
        
        # Store softmax result
        tl.store(softmax_out_ptr + softmax_offset + offsets, softmax_vals, mask=mask)
        
        # Load coordinates
        # For linspace_x: repeat pattern [0, 1, 2, ..., 63] 64 times
        # For linspace_y: repeat each value 64 times
        spatial_offsets = offsets  # 0 to 4095
        x_indices = spatial_offsets % 64  # column index
        y_indices = spatial_offsets // 64  # row index
        
        x_coords = tl.load(linspace_x_ptr + x_indices, mask=mask, other=0.0)
        y_coords = tl.load(linspace_y_ptr + y_indices * 64, mask=mask, other=0.0)
        
        # Accumulate weighted sums
        weighted_x_sum += tl.sum(softmax_vals * x_coords, axis=0)
        weighted_y_sum += tl.sum(softmax_vals * y_coords, axis=0)
    
    # Store coordinate results
    if tl.program_id(0) == pid:  # Only the first thread stores
        tl.store(coord_out_ptr + coord_offset, weighted_x_sum)
        tl.store(coord_out_ptr + coord_offset + 1, weighted_y_sum)

@torch.fx.wrap
def fused_softmax_coord_regression(in_0, in_1, in_2):
    """
    Wrapper function for the fused kernel.
    
    Args:
        in_0: linspace_x with shape [1, 1, 1, 64]
        in_1: linspace_y with shape [1, 1, 64, 1]
        in_2: input heatmaps with shape [batch, 17, 4096]
    
    Returns:
        tmp_3: softmax reshaped to [batch, 17, 64, 64]
        tmp_10: coordinates with shape [batch, 17, 2]
    """
    batch_size = in_2.shape[0]
    num_keypoints = 17
    N = 4096
    
    # Allocate output tensors
    softmax_out = torch.empty((batch_size, num_keypoints, N), dtype=torch.float32, device=in_2.device)
    coord_out = torch.empty((batch_size, num_keypoints, 2), dtype=torch.float32, device=in_2.device)
    
    # Flatten linspace_x and linspace_y for easier access
    # in_0: [1, 1, 1, 64] -> [64]
    # in_1: [1, 1, 64, 1] -> [64]
    linspace_x = in_0.view(64)
    linspace_y = in_1.view(64)
    
    # Launch kernel
    grid = (batch_size * num_keypoints,)
    fused_softmax_coord_kernel[grid](
        in_2,
        linspace_x,
        linspace_y,
        softmax_out,
        coord_out,
        batch_size,
        num_keypoints,
        N,
    )
    
    # Reshape outputs to match expected format
    tmp_3 = softmax_out.reshape(batch_size, 17, 64, 64)
    tmp_10 = coord_out
    
    return (tmp_3, tmp_10)

def replacement_func():
    return fused_softmax_coord_regression