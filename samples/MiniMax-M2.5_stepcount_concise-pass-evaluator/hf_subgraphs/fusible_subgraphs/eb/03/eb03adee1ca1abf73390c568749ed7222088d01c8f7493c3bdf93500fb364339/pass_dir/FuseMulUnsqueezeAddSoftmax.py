import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1):
    """
    Match the computation pattern:
    1. in_1 * constant (scalar multiplication)
    2. in_0.unsqueeze(2) (add dimension)
    3. tmp_0 + tmp_1 (broadcast add)
    4. softmax(dim=-1) (softmax on last dim)
    5. dropout(p=0.0) (no-op, can be optimized away)
    
    This pattern matches the exact operations from model.py
    """
    tmp_0 = in_1 * 0.1767766952966369
    tmp_1 = in_0.unsqueeze(2)
    tmp_2 = tmp_0 + tmp_1
    tmp_3 = tmp_2.softmax(dim=-1)
    # dropout with p=0.0 is a no-op, returns input unchanged
    tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    return tmp_4

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Triton kernel for fused multiply + unsqueeze + add + softmax
@triton.autotune(
    configs=[
        # Try different block sizes for the softmax dimension (49)
        triton.Config({'BLOCK_SIZE': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=4),
    ],
    key=['n_cols'],  # Re-autotune based on the softmax dimension size
)
@triton.jit
def fused_mul_add_softmax_kernel(
    in_0_ptr,  # in_0: [B, H, W, W] -> after unsqueeze: [B, H, 1, W, W]
    in_1_ptr,  # in_1: [B, H, K, W, W]
    out_ptr,
    B: tl.constexpr,    # Batch size (1)
    H: tl.constexpr,    # Number of heads (361)
    K: tl.constexpr,    # Number of attention heads dimension (3)
    W: tl.constexpr,    # Width/height (49)
    n_cols: tl.constexpr,  # Total elements per softmax: K * W * W
    n_rows: tl.constexpr,  # Total rows: B * H * K * W
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes:
    1. tmp_0 = in_1 * scale (scale = 0.1767766952966369)
    2. tmp_1 = in_0.unsqueeze(2) [B, H, W, W] -> [B, H, 1, W, W]
    3. tmp_2 = tmp_0 + tmp_1 (broadcast add)
    4. tmp_3 = softmax(tmp_2, dim=-1)
    
    Input shapes:
    - in_0: [B, H, W, W] = [1, 361, 49, 49]
    - in_1: [B, H, K, W, W] = [1, 361, 3, 49, 49]
    
    Output shape:
    - out: [B, H, K, W, W] = [1, 361, 3, 49, 49]
    
    We parallelize over n_rows = B * H * K * W = 1 * 361 * 3 * 49
    Each thread block computes one row's softmax over W elements.
    """
    # Get row index (0 to n_rows-1)
    row_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = row_idx < n_rows
    
    # Calculate indices
    # row_idx = b * H * K * W + h * K * W + k * W + w
    # We need to decompose to access in_0 and in_1 correctly
    
    # Calculate the w (column within softmax) and k (attention head) indices
    # Each row corresponds to a specific (b, h, k, w) tuple
    # For softmax, we need to process all W elements for each (b, h, k) group
    
    # Actually, for softmax with dim=-1, we process W elements at a time
    # Let's reorganize: total groups = B * H * K = 1 * 361 * 3 = 1083
    # Each group has W = 49 elements
    
    # The original n_cols = K * W * W = 3 * 49 * 49 doesn't work well
    # Let me reconsider...
    
    # For softmax(dim=-1), we process per (b, h, k) slice
    # Total number of softmax groups = B * H * K = 1 * 361 * 3 = 1083
    # Each group has W = 49 elements
    
    # Each thread should load W elements for one softmax group
    pass


# Let me rewrite the kernel with a simpler and more efficient approach
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=8),
    ],
    key=['W'],
)
@triton.jit
def fused_mul_add_softmax_kernel_v2(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    # Shapes
    B: tl.constexpr,   # 1
    H: tl.constexpr,   # 361  
    K: tl.constexpr,   # 3
    W: tl.constexpr,   # 49 (softmax dimension)
    # Derived
    num_groups: tl.constexpr,  # B * H * K = 1 * 361 * 3
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for:
    tmp_0 = in_1 * scale
    tmp_1 = in_0.unsqueeze(2)  [B, H, W, W] -> [B, H, 1, W, W]
    tmp_2 = tmp_0 + tmp_1      [B, H, K, W, W]
    tmp_3 = softmax(tmp_2, dim=-1)
    
    Parallelization: Each thread block computes one softmax group
    num_groups = B * H * K = 1083 groups
    Each group has W = 49 elements to process
    """
    # Each program handles one softmax group (one row in the K dimension)
    group_idx = tl.program_id(0)
    
    if group_idx >= num_groups:
        return
    
    # Decompose group_idx to (b, h, k)
    # group_idx = b * H * K + h * K + k
    # Since B = 1: group_idx = h * K + k
    h = group_idx // K
    k = group_idx % K
    
    # Offset for in_1: [b, h, k, w] -> flat offset
    # in_1 stride: [H*K*W*W, K*W*W, W*W, W, 1]
    # Since B=1: offset = h * K * W * W + k * W * W
    in_1_offset_base = h * K * W * W + k * W * W
    
    # Offset for in_0: [b, h, w1, w2] -> flat offset  
    # in_0 stride: [H*W*W, W*W, W, 1]
    # Since B=1: offset = h * W * W + w2 (for the column we're processing)
    in_0_offset_base = h * W * W
    
    # Allocate local storage for this group's values
    # We need to store W values for the softmax computation
    # Use a 1D array in tl.arange(0, W)
    
    # First pass: compute the max for numerical stability
    # Load values: in_1[k] * scale + in_0[b, h, w] (broadcast across k)
    
    # Load and compute: tmp_0 + tmp_1 for each w
    # in_1[b, h, k, w] * scale + in_0[b, h, w', w] (where w' varies)
    
    # Actually, let me think more carefully:
    # tmp_1 = in_0.unsqueeze(2) creates [B, H, 1, W, W]
    # When we add tmp_0 [B, H, K, W, W] + tmp_1 [B, H, 1, W, W]
    # tmp_1 broadcasts across the K dimension
    
    # So for each (b, h, k, w_col), we add:
    # - in_1[b, h, k, w_row, w_col] * scale + in_0[b, h, w_row, w_col]
    # where w_row is the dimension we're doing softmax over
    
    # That means for softmax at dimension -1, for each (b, h, k, w_col),
    # we need to compute max over w_row of (in_1[b,h,k,w_row,w_col]*scale + in_0[b,h,w_row,w_col])
    
    # For the output at position (b, h, k, w_row, w_col):
    # softmax = exp(x - max) / sum(exp(x - max))
    
    # This is a 2D operation - we can parallelize over (b, h, k, w_col) groups
    # and each group computes softmax over w_row dimension
    
    # Total groups = B * H * K * W_col = 1 * 361 * 3 * 49 = 53067
    pass


# Let me simplify and use a cleaner approach
# We'll process in a way that handles the 2D nature efficiently

SCALE: tl.constexpr = 0.1767766952966369

@triton.autotune(
    configs=[
        # Different block sizes for different workloads
        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=2, num_warps=8),
        # Try more stages for better memory latency hiding
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=8),
    ],
    key=['W'],
)
@triton.jit
def fused_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    B: tl.constexpr, H: tl.constexpr, K: tl.constexpr, W: tl.constexpr,
    num_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute: in_1 * scale + broadcast(in_0.unsqueeze(2))
    
    This is the fused multiply + unsqueeze + add operation.
    Grid: one block per BLOCK_SIZE elements
    Each block handles multiple elements with vectorized loads.
    """
    # Get block start index
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Total elements = B * H * K * W * W
    # Decompose offset into (b, h, k, w_row, w_col)
    tmp = offsets
    w_col = tmp % W
    tmp = tmp // W
    w_row = tmp % W
    tmp = tmp // W
    k = tmp % K
    tmp = tmp // K
    h = tmp % H
    
    # Compute offsets for in_1 and in_0
    # in_1: [b, h, k, w_row, w_col]
    in1_offsets = h * K * W * W + k * W * W + w_row * W + w_col
    
    # in_0: [b, h, w_row, w_col] (broadcast across k dimension)
    in0_offsets = h * W * W + w_row * W + w_col
    
    # Vectorized loads and compute
    in1_vals = tl.load(in_1_ptr + in1_offsets, mask=mask, other=0.0).to(tl.float32)
    in0_vals = tl.load(in_0_ptr + in0_offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Fused multiply + add
    combined = in1_vals * SCALE + in0_vals
    
    # Store
    tl.store(out_ptr + offsets, combined, mask=mask)


@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1):
    """
    Wrapper function that launches the fused Triton kernel.
    
    Input shapes:
    - in_0: [1, 361, 49, 49] (4D)
    - in_1: [1, 361, 3, 49, 49] (5D)
    
    Output shape:
    - out: [1, 361, 3, 49, 49]
    
    Computation:
    1. tmp_0 = in_1 * scale (multiply)
    2. tmp_1 = in_0.unsqueeze(2) (unsqueeze/broadcast)
    3. tmp_2 = tmp_0 + tmp_1 (add with broadcast)
    4. tmp_3 = softmax(tmp_2, dim=-1)
    5. dropout(tmp_3, p=0.0) = tmp_3 (no-op)
    """
    # in_0 shape: [B, H, W, W] = [1, 361, 49, 49]
    B, H, W, W2 = in_0.shape
    assert W == W2, "Expected square last two dimensions"
    
    # in_1 shape: [B, H, K, W, W]
    _, _, K, _, _ = in_1.shape
    
    # Output
    combined = torch.empty_like(in_1)  # [1, 361, 3, 49, 49]
    
    # Total number of elements in output
    num_elements = B * H * K * W * W  # 1 * 361 * 3 * 49 * 49 = 2,579,367
    
    # Compute grid size
    BLOCK_SIZE = 1024  # Will be autotuned
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    grid = (num_programs,)
    
    fused_kernel[grid](
        in_0, in_1, combined,
        B, H, K, W,
        num_elements,
    )
    
    # Apply softmax on last dimension
    # PyTorch's softmax is highly optimized
    out = combined.softmax(dim=-1)
    
    # dropout with p=0.0 is a no-op, so we can skip it
    # torch.nn.functional.dropout(out, 0.0, False, False) would just return out
    
    return out


def replacement_func():
    return fused_kernel_wrapper