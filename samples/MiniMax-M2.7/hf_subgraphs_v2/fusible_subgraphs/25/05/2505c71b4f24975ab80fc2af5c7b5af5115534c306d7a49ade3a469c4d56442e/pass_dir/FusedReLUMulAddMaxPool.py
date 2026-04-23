import torch
import triton
import triton.language as tl

# Pattern variant 1: Match the computation graph using in_1 and in_0 directly
# tmp_2 = relu(in_2), tmp_3 = in_1 * tmp_2, tmp_4 = tmp_3 + in_0
# tmp_5 = max_pool2d(in_3) with ceil_mode, tmp_6 = cat([tmp_5, tmp_4], dim=1)
def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    tmp_5 = torch.nn.functional.max_pool2d(in_3, 2, 1, 0, 1, ceil_mode=True, return_indices=False)
    tmp_6 = torch.cat([tmp_5, tmp_4], dim=1)
    return tmp_6

# Extract arguments needed for replacement
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Optimized fused kernel that combines:
# 1. relu -> multiply -> add (for channels 0 to C1-1)
# 2. max_pool2d with ceil_mode (for channels C1 to C1+C2-1)
# 3. concatenation along channel dimension
@triton.jit
def fused_relu_mul_add_maxpool_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr, out_ptr,
    N, C1, C2, H, W, in_3_H, in_3_W, out_H, out_W
):
    # Position mapping
    total_pid = tl.program_id(0)
    batch_idx = total_pid // (out_H * out_W)
    rem = total_pid % (out_H * out_W)
    out_h = rem // out_W
    out_w = rem % out_W
    
    # Load broadcast scalars
    in_0_val = tl.load(in_0_ptr)
    in_1_val = tl.load(in_1_ptr)
    
    out_base_idx = batch_idx * (C1 + C2) * out_H * out_W + out_h * out_W + out_w
    
    # Process channels 0 to C1-1: relu, multiply, add
    for c in range(C1):
        in_2_idx = batch_idx * C1 * H * W + c * H * W + out_h * W + out_w
        x = tl.load(in_2_ptr + in_2_idx)
        # Fuse: relu(x) * in_1 + in_0
        x = tl.maximum(x, 0)  # relu
        x = x * in_1_val  # multiply by scale
        x = x + in_0_val  # add bias
        out_idx = out_base_idx + c * out_H * out_W
        tl.store(out_ptr + out_idx, x)
    
    # Process channels C1 to C1+C2-1: max_pool2d with ceil_mode
    for c in range(C2):
        # Ceil mode pooling: h_start = oh * 2 - 1
        h_start = out_h * 2 - 1
        w_start = out_w * 2 - 1
        
        max_val = float('-inf')
        for kh in range(2):
            for kw in range(2):
                h = h_start + kh
                w = w_start + kw
                # Boundary check with ceil_mode padding
                if h >= 0 and h < in_3_H and w >= 0 and w < in_3_W:
                    in_3_idx = batch_idx * C2 * in_3_H * in_3_W + c * in_3_H * in_3_W + h * in_3_W + w
                    val = tl.load(in_3_ptr + in_3_idx)
                    max_val = tl.maximum(max_val, val)
        
        out_idx = out_base_idx + (C1 + c) * out_H * out_W
        tl.store(out_ptr + out_idx, max_val)

@triton.autotune(
    key=['N', 'out_H', 'out_W'],
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=1),
    ]
)
@triton.jit
def fused_relu_mul_add_maxpool_kernel_autotuned(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr, out_ptr,
    N, C1, C2, H, W, in_3_H, in_3_W, out_H, out_W,
    BLOCK_SIZE: tl.constexpr
):
    fused_relu_mul_add_maxpool_kernel(
        in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr, out_ptr,
        N, C1, C2, H, W, in_3_H, in_3_W, out_H, out_W
    )

@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1, in_2, in_3):
    N, C1, H, W = in_2.shape
    _, C2, in_3_H, in_3_W = in_3.shape
    
    # Calculate output spatial dimensions for ceil_mode pooling with k=2, s=1, p=1
    out_H = (in_3_H + 1) if ((in_3_H + 1) % 2 == 1) else in_3_H // 2 + 1
    out_W = (in_3_W + 1) if ((in_3_W + 1) % 2 == 1) else in_3_W // 2 + 1
    
    # Output shape: [N, C1+C2, out_H, out_W]
    output = torch.empty((N, C1 + C2, out_H, out_W), dtype=in_2.dtype, device=in_2.device)
    
    # Grid: one program per output spatial position per batch
    grid = (N * out_H * out_W,)
    
    fused_relu_mul_add_maxpool_kernel_autotuned[grid](
        in_0, in_1, in_2, in_3, output,
        N, C1, C2, H, W, in_3_H, in_3_W, out_H, out_W,
        BLOCK_SIZE=256
    )
    
    return output

def replacement_func():
    return fused_kernel_wrapper