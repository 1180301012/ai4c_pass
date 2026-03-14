import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Match the computation pattern:
    conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1) -> view -> softmax -> unsqueeze
    
    This pattern is common in attention mechanisms (conv mask).
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_1 = tmp_0 = None
    tmp_3 = tmp_2.view(64, 1, 192)
    tmp_2 = None
    tmp_4 = torch.nn.functional.softmax(tmp_3, 2, _stacklevel=5)
    tmp_3 = None
    tmp_5 = tmp_4.unsqueeze(-1)
    tmp_4 = None
    return (tmp_5,)


def pattern_0(in_0, in_1, in_2):
    """Pattern for batch=1, shape (1, 1, 48)"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_1 = tmp_0 = None
    tmp_3 = tmp_2.view(1, 1, 48)
    tmp_2 = None
    tmp_4 = torch.nn.functional.softmax(tmp_3, 2, _stacklevel=5)
    tmp_3 = None
    tmp_5 = tmp_4.unsqueeze(-1)
    tmp_4 = None
    return (tmp_5,)


def pattern_1(in_0, in_1, in_2):
    """Pattern for batch=256, shape (256, 1, 192)"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_1 = tmp_0 = None
    tmp_3 = tmp_2.view(256, 1, 192)
    tmp_2 = None
    tmp_4 = torch.nn.functional.softmax(tmp_3, 2, _stacklevel=5)
    tmp_3 = None
    tmp_5 = tmp_4.unsqueeze(-1)
    tmp_4 = None
    return (tmp_5,)


def pattern_2(in_0, in_1, in_2):
    """Pattern for batch=64, shape (64, 1, 3072)"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_1 = tmp_0 = None
    tmp_3 = tmp_2.view(64, 1, 3072)
    tmp_2 = None
    tmp_4 = torch.nn.functional.softmax(tmp_3, 2, _stacklevel=5)
    tmp_3 = None
    tmp_5 = tmp_4.unsqueeze(-1)
    tmp_4 = None
    return (tmp_5,)


def pattern_3(in_0, in_1, in_2):
    """Pattern for batch=256, shape (256, 1, 3072)"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_1 = tmp_0 = None
    tmp_3 = tmp_2.view(256, 1, 3072)
    tmp_2 = None
    tmp_4 = torch.nn.functional.softmax(tmp_3, 2, _stacklevel=5)
    tmp_3 = None
    tmp_5 = tmp_4.unsqueeze(-1)
    tmp_4 = None
    return (tmp_5,)


def pattern_4(in_0, in_1, in_2):
    """Pattern for batch=1, shape (1, 1, 4096)"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_1 = tmp_0 = None
    tmp_3 = tmp_2.view(1, 1, 4096)
    tmp_2 = None
    tmp_4 = torch.nn.functional.softmax(tmp_3, 2, _stacklevel=5)
    tmp_3 = None
    tmp_5 = tmp_4.unsqueeze(-1)
    tmp_4 = None
    return (tmp_5,)


def pattern_5(in_0, in_1, in_2):
    """Pattern for batch=32, shape (32, 1, 4096)"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_1 = tmp_0 = None
    tmp_3 = tmp_2.view(32, 1, 4096)
    tmp_2 = None
    tmp_4 = torch.nn.functional.softmax(tmp_3, 2, _stacklevel=5)
    tmp_3 = None
    tmp_5 = tmp_4.unsqueeze(-1)
    tmp_4 = None
    return (tmp_5,)


def pattern_6(in_0, in_1, in_2):
    """Pattern for batch=64, shape (64, 1, 48)"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_1 = tmp_0 = None
    tmp_3 = tmp_2.view(64, 1, 48)
    tmp_2 = None
    tmp_4 = torch.nn.functional.softmax(tmp_3, 2, _stacklevel=5)
    tmp_3 = None
    tmp_5 = tmp_4.unsqueeze(-1)
    tmp_4 = None
    return (tmp_5,)


def pattern_7(in_0, in_1, in_2):
    """Pattern for batch=1, shape (1, 1, 192)"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_1 = tmp_0 = None
    tmp_3 = tmp_2.view(1, 1, 192)
    tmp_2 = None
    tmp_4 = torch.nn.functional.softmax(tmp_3, 2, _stacklevel=5)
    tmp_3 = None
    tmp_5 = tmp_4.unsqueeze(-1)
    tmp_4 = None
    return (tmp_5,)


def pattern_8(in_0, in_1, in_2):
    """Pattern for batch=256, shape (256, 1, 48)"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_1 = tmp_0 = None
    tmp_3 = tmp_2.view(256, 1, 48)
    tmp_2 = None
    tmp_4 = torch.nn.functional.softmax(tmp_3, 2, _stacklevel=5)
    tmp_3 = None
    tmp_5 = tmp_4.unsqueeze(-1)
    tmp_4 = None
    return (tmp_5,)


# Autotune configurations for different input sizes
@triton.autotune(
    configs=[
        # Small sizes
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        # Medium sizes
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
    ],
    key=['N'],
)
@triton.jit
def fused_view_softmax_unsqueeze_kernel(
    input_ptr,
    output_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for:
    1. View: reshape (B, C, H, W) -> (B, 1, H*W) 
    2. Softmax on dim=2
    3. Unsqueeze: add dimension at end -> (B, 1, H*W, 1)
    
    This kernel fuses all three operations into one for better performance.
    """
    # Get program ID for batch processing
    pid = tl.program_id(0)
    n_elements = N
    
    # Calculate the offset for this batch
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Compute softmax: exp(x - max(x)) / sum(exp(x - max(x)))
    # First, get the max value for numerical stability
    # For each "row" (pid), we need to compute max
    # Since we're processing one row at a time, we compute max across all elements
    
    # Actually, for proper softmax we need to know the max across ALL elements in the row
    # But we're processing in blocks, so we need a two-pass approach or reduction
    
    # Simplified: compute exp and sum in one pass (less numerically stable but faster)
    # Better: use block-wise approach
    
    # For now, let's use a simpler approach that loads all data needed
    # In practice, we'd use tl.reduce for the max, but let's do a simpler version
    
    # Compute exp(x) - this is the key operation
    # Note: for proper softmax we need to subtract max first
    # This is a simplified version that should work for most cases
    
    # Actually, let's implement a two-phase kernel for correctness
    # Phase 1: compute max
    # Phase 2: compute exp and sum, then normalize
    
    # For simplicity and to ensure correctness, we use a single pass with exp
    # In production, you'd want the two-pass version
    exp_x = tl.exp(x - tl.max(x, axis=0))
    sum_exp_x = tl.sum(exp_x, axis=0)
    softmax = exp_x / sum_exp_x
    
    # Store result (unsqueeze is implicit - we just write to the output with +1 dimension)
    tl.store(output_ptr + offsets, softmax, mask=mask)


@triton.jit
def fused_view_softmax_unsqueeze_kernel_v2(
    input_ptr,
    output_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized fused kernel for View + Softmax + Unsqueeze.
    Uses block-wise reduction for numerical stability.
    """
    pid = tl.program_id(0)
    n_elements = N
    
    # Create offsets
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Softmax computation with numerical stability
    # Step 1: Find max
    max_val = tl.max(x, axis=0)
    
    # Step 2: Compute exp(x - max)
    exp_vals = tl.exp(x - max_val)
    
    # Step 3: Sum exp values
    sum_exp = tl.sum(exp_vals, axis=0)
    
    # Step 4: Normalize
    result = exp_vals / sum_exp
    
    # Store (output shape will be same as input but represents unsqueezed)
    tl.store(output_ptr + offsets, result, mask=mask)


def compute_output_shape(input_tensor, view_shape):
    """
    Compute output shape based on view operation.
    View transforms (B, C, H, W) to (B', 1, H*W) where B' = B or 1
    Then unsqueeze(-1) adds a dimension at the end.
    """
    # The output shape is (view_shape[0], view_shape[1], view_shape[2], 1)
    # But our kernel processes one "row" at a time
    return view_shape


def replacement_args(in_0, in_1, in_2):
    """Extract arguments needed for the replacement function."""
    return (in_0, in_1, in_2)


def get_view_params(view_node):
    """Extract view shape parameters from the view node."""
    # The view node has args with the target shape
    # In this case, it's constructed in the pattern
    # We need to extract the shape from the pattern call
    pass


def create_optimized_kernel(view_shape):
    """
    Create a customized Triton kernel for the specific view shape.
    This allows better optimization for each specific size.
    """
    B = view_shape[0]  # batch size (or 1, 64, 256, etc)
    seq_len = view_shape[2]  # spatial size (48, 192, 3072, 4096)
    
    # Calculate grid based on batch size
    grid = (B,)
    
    return grid, seq_len


@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1, in_2):
    """
    Wrapper function that implements the fused Conv2D + View + Softmax + Unsqueeze.
    
    The original pattern:
    1. Conv2D: (B, C, H, W) -> (B, C, H, W)
    2. View: (B, C, H, W) -> (B, 1, H*W)  
    3. Softmax: (B, 1, H*W) -> (B, 1, H*W)
    4. Unsqueeze: (B, 1, H*W) -> (B, 1, H*W, 1)
    
    Our fused implementation leverages cuDNN for Conv2D (already optimized)
    and fuses View + Softmax + Unsqueeze into a single Triton kernel.
    """
    # Step 1: Conv2D (use cuDNN - already highly optimized)
    conv_out = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    
    # Get shapes to determine kernel config
    B, C, H, W = conv_out.shape
    seq_len = H * W
    
    # Reshape for softmax: (B, C, H, W) -> (B, 1, H*W) 
    # But actually we need (B, H*W) for the kernel
    # The original view uses (B', 1, seq_len) where B' may be B or a divisor
    # Looking at the patterns:
    # - view(64, 1, 192): B=64, seq_len=192 -> input was (64, 304, 16, 12) -> 64*304*16*12 = 64*192 = 12288, wait that's not right
    # Let me recalculate: 64*304*16*12 = 64*304*192 = 64*58368 = 3735552
    # view to (64, 1, 192) - but that's only 64*1*192 = 12288 elements
    # Ah! The view is on the result AFTER conv2d, which has shape (64, 304, 16, 12) = 64*304*16*12
    # But view to (64, 1, 192) = 64*1*192 = 12288, that's not the same!
    
    # Wait, let me check the weight_meta:
    # in_2 shape: [64, 304, 16, 12] - so conv output is [64, 304, 16, 12]
    # view to (64, 1, 192) = 64 * 1 * 192 = 12288
    # But 64 * 304 * 16 * 12 = 3735552
    
    # There's something wrong. Let me re-examine...
    # Actually wait - the conv2d with groups=1 should preserve the channel dimension
    # Weight is [1, 304, 1, 1], so output channels = 1
    # So conv_out would be [64, 1, 16, 12]
    # Then view(64, 1, 192) = 64 * 1 * 192 = 12288
    # And 64 * 1 * 16 * 12 = 64 * 192 = 12288 ✓
    
    # Great! So the output channels from conv2d is 1 (because weight has out_channels=1)
    
    # After conv2d: (B, 1, H, W)
    # After view: (B, 1, H*W)
    # After softmax: (B, 1, H*W) 
    # After unsqueeze: (B, 1, H*W, 1)
    
    # For the fused kernel, we process each batch element
    # Input: (B, 1, H*W) - we treat this as B rows of length H*W
    
    # Actually, let me simplify: we reshape to 2D first, then apply softmax row-wise
    conv_reshaped = conv_out.view(B, seq_len)  # (B, H*W)
    
    # Allocate output
    output = torch.empty((B, seq_len), device=conv_out.device, dtype=conv_out.dtype)
    
    # Launch kernel - one program per batch element
    BLOCK_SIZE = 4096
    if seq_len <= 128:
        BLOCK_SIZE = 128
    elif seq_len <= 256:
        BLOCK_SIZE = 256
    elif seq_len <= 512:
        BLOCK_SIZE = 512
    elif seq_len <= 1024:
        BLOCK_SIZE = 1024
    elif seq_len <= 2048:
        BLOCK_SIZE = 2048
    
    grid = (B,)
    
    fused_view_softmax_unsqueeze_kernel_v2[grid](
        conv_reshaped,
        output,
        seq_len,
        BLOCK_SIZE,
    )
    
    # Reshape to match original output: (B, H*W) -> (B, 1, H*W, 1)
    # Original: view to (B, 1, H*W), softmax on dim=2, unsqueeze(-1) -> (B, 1, H*W, 1)
    output = output.view(B, 1, seq_len)  # (B, 1, H*W)
    output = output.unsqueeze(-1)  # (B, 1, H*W, 1)
    
    return output


def replacement_func():
    """Return the fused kernel wrapper function."""
    return fused_kernel_wrapper