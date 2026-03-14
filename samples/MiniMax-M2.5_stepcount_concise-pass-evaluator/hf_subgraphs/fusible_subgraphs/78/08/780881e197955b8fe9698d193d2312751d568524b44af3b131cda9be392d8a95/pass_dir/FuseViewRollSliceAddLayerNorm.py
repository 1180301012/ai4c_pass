import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Try matching a simpler pattern - just the final layer_norm operation.
    This tests if the framework can handle multi-value returns.
    
    Case 1: C=96
    """
    tmp_7 = in_3.contiguous().view(-1, 133, 133, 96)
    tmp_8 = in_2 + tmp_7
    tmp_9 = torch.nn.functional.layer_norm(tmp_8, (96,), in_1, in_0, 1e-05)
    return tmp_8, tmp_9


def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments needed for the replacement kernel."""
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        # Different block sizes for different hidden dimensions
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def fused_kernel(
    in_3_ptr, in_2_ptr, in_1_ptr, in_0_ptr,
    out_8_ptr, out_9_ptr,
    N: tl.constexpr, C: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
    R: tl.constexpr, S: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Load and reshape in_3 (view operation)
    2. Apply roll with shifts=(3, 3)
    3. Slice to get valid region
    4. Add with in_2
    5. Apply layer_norm
    """
    # Each program handles one row of the output
    pid = tl.program_id(0)
    
    # Calculate which row this program handles
    row_offset = pid * BLOCK_SIZE
    rows = row_offset + tl.arange(0, BLOCK_SIZE)
    mask = rows < N
    
    # Calculate the (h, w) position in the reshaped tensor
    h = rows // W
    w = rows % W
    
    # Apply roll: shift by (3, 3) in (h, w) dims
    # After roll: new_h = (h + 3) % H, new_w = (w + 3) % W
    rolled_h = (h + R) % H
    rolled_w = (w + S) % W
    
    # Calculate original flat index in in_3 (after view)
    # Original shape after view: [1, H, W, C] where H=133, W=133 for case 1
    flat_idx = rolled_h * W + rolled_w
    
    # Load in_3 data: the view operation means we access in_3 as [batch, h, w, c]
    # But in_3 has shape [1, a, b, c, d, C] = [1, p, q, r, s, C]
    # After view(-1, H, W, C), it becomes [1, H*q/ratio, H, W, C] - need to compute properly
    # For input [1, 19, 7, 19, 7, 96], view(-1, 133, 133, 96) -> [1, 133, 133, 96]
    # So we can just treat in_3 as [1, H*q, r, s, C] and select properly
    
    # Actually, let's compute the channel offset
    c_offset = tl.arange(0, C)
    channel_idx = c_offset  # [0, 1, 2, ..., C-1]
    
    # Load from in_3 with roll applied
    # The in_3 tensor after view becomes [1, H*factor, W*factor, C]
    # We need to map the rolled (h, w) back to the original in_3 indices
    
    # For the first graph: in_3 shape [1, 19, 7, 19, 7, 96] = [1, 19*7, 19*7, 96]
    # After view(-1, 133, 133, 96): [1, 133, 133, 96]
    # The roll shifts by (3, 3) so we use (rolled_h, rolled_w)
    
    # Load in_3 data
    in_3_offset = rolled_h * W * C + rolled_w * C + channel_idx
    x = tl.load(in_3_ptr + in_3_offset, mask=mask, other=0.0)
    
    # Load in_2 data (shape: [1, N, C])
    in_2_offset = rows * C + channel_idx
    y = tl.load(in_2_ptr + in_2_offset, mask=mask, other=0.0)
    
    # Add
    z = x + y
    
    # Now apply layer_norm
    # Compute mean
    mean = tl.sum(z, axis=0) / C
    # Compute variance
    var = tl.sum((z - mean) * (z - mean), axis=0) / C
    # Normalize
    eps = 1e-05
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Load weight and bias for layer_norm
    weight = tl.load(in_1_ptr + channel_idx)
    bias = tl.load(in_0_ptr + channel_idx)
    
    # Apply layer_norm: (z - mean) * weight * inv_std + bias
    normalized = (z - mean) * inv_std
    normalized = normalized * weight + bias
    
    # Store outputs
    tl.store(out_8_ptr + in_2_offset, z, mask=mask)
    tl.store(out_9_ptr + in_2_offset, normalized, mask=mask)


@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1, in_2, in_3):
    """
    Wrapper function that launches the fused Triton kernel.
    
    Expected tensor shapes:
    - in_0: [C] (layer_norm bias)
    - in_1: [C] (layer_norm weight)
    - in_2: [1, N, C] (layer output to add)
    - in_3: [1, a, b, c, d, C] (permuted tensor, e.g., [1, 19, 7, 19, 7, 96])
    
    Returns:
    - out_8: [1, N, C] (add result)
    - out_9: [1, N, C] (layer_norm result)
    """
    # Determine dimensions from input shapes
    # in_3 shape: [1, p, q, r, s, C] -> view to [p*q, r*s, C]
    in_3_shape = in_3.shape
    C = in_3_shape[-1]  # channel dimension
    
    # Compute the view dimensions: in_3.shape = [1, a, b, c, d, C]
    # After view(-1, H, W, C): H = a * b, W = c * d (for the first case: 19*7=133, 19*7=133)
    H = in_3_shape[1] * in_3_shape[2]
    W = in_3_shape[3] * in_3_shape[4]
    
    # After roll and slice: we get [slice_H, slice_W, C]
    # For case 1: H=133, W=133, slice to 128x128 -> N = 128*128 = 16384
    # For case 2: H=70, W=70, slice to 64x64 -> N = 64*64 = 4096
    # For case 3: H=35, W=35, slice to 32x32 -> N = 32*32 = 1024
    
    # The roll shifts by (3, 3), and we slice from 0 to H-5 (or 128 for case 1)
    # Actually looking at the pattern: slice(None, 128, None) means from start to 128
    # So the output N is 128 * 128 = 16384 for case 1
    
    # Determine N from in_2 shape (which is the target)
    N = in_2.numel() // C  # Should be 16384, 4096, or 1024
    
    # Calculate the actual slice size from N
    slice_size = int(N ** 0.5)  # sqrt(N)
    
    # Roll parameters
    R = 3  # shift in height
    S = 3  # shift in width
    
    # Allocate output tensors
    out_8 = torch.empty_like(in_2)
    out_9 = torch.empty_like(in_2)
    
    # Calculate grid
    BLOCK_SIZE = 1024
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Ensure contiguous
    in_3_contig = in_3.contiguous()
    
    # Launch kernel
    fused_kernel[grid](
        in_3_ptr=in_3_contig,
        in_2_ptr=in_2,
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        out_8_ptr=out_8,
        out_9_ptr=out_9,
        N=N,
        C=C,
        H=H,
        W=W,
        R=R,
        S=S,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_8, out_9


def replacement_func():
    return fused_kernel_wrapper