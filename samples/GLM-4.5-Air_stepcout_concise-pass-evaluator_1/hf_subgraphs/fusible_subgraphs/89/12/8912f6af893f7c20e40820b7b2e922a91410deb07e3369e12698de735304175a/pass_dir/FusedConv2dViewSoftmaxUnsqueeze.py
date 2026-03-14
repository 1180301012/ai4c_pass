import torch
import triton
import triton.language as tl


# Pattern matching function - matches conv2d + view + softmax + unsqueeze
def pattern(in_0, in_1, in_2):
    """
    Pattern: conv2d(in_2, in_1, in_0) -> view(B, 1, seq) -> softmax(dim=2) -> unsqueeze(-1)
    This pattern is common in attention mechanisms (e.g., SE-Net, GCNet).
    """
    # Step 1: 1x1 Conv2D - directly use the output
    tmp_conv = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    
    # Step 2: View - the shape will be inferred from the traced graph
    # We need to use reshape instead of view to avoid static shape requirements
    tmp_view = tmp_conv.reshape(tmp_conv.shape[0], 1, -1)
    
    # Step 3: Softmax along dim 2
    tmp_softmax = torch.nn.functional.softmax(tmp_view, 2, _stacklevel=5)
    
    # Step 4: Unsqueeze at dim -1
    tmp_unsqueeze = tmp_softmax.unsqueeze(-1)
    
    # Return only the final output
    return tmp_unsqueeze


def replacement_args(in_0, in_1, in_2):
    """Extract arguments needed for the optimized kernel."""
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 256}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 512}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 1024}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 2048}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 4096}, num_stages=4, num_warps=8),
    ],
    key=['seq_len'],
)
@triton.jit
def fused_conv_softmax_kernel(
    # Input pointers
    input_ptr, weight_ptr, bias_ptr,
    # Output pointer
    output_ptr,
    # Dimensions
    B, C, H, W, seq_len,
    # Strides
    input_stride_b, input_stride_c, input_stride_h, input_stride_w,
    weight_stride_c, weight_stride_h, weight_stride_w,
    # Block size
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused kernel: Conv2D (1x1) + View + Softmax + Unsqueeze
    
    For each batch element and each spatial position:
    1. Apply 1x1 conv (single channel weight contribution)
    2. Apply softmax across all spatial positions in the batch element
    
    Since the weight has shape [1, C, 1, 1] and groups=1, this is essentially:
    output[b, c, h, w] = sum over input channels of (input[b, c', h, w] * weight[0, c', 0, 0]) + bias[0]
    
    Then we apply softmax across the seq_len dimension for each batch element.
    """
    # Get batch index
    pid_b = tl.program_id(0)
    pid_seq = tl.program_id(1)
    
    # Calculate the starting offset for this batch
    # Input: [B, C, H, W]
    # Each batch has C * H * W elements
    batch_offset = pid_b * C * H * W
    
    # Each program handles a block of sequence elements
    # For softmax, we need to compute exp(x - max(x)) for all elements, then sum
    # We'll use a reduction pattern
    
    # Load bias (broadcast to all positions)
    bias = tl.load(bias_ptr)
    
    # Create offsets for input channels
    # For each spatial position in the block, we need to compute conv output
    # Conv with 1x1 kernel: out[b, c, h, w] = sum_c(in[b, c, h, w] * weight[0, c, 0, 0]) + bias
    
    # Calculate which spatial positions this block handles
    seq_start = pid_seq * BLOCK_SIZE_N
    seq_end = min(seq_start + BLOCK_SIZE_N, seq_len)
    
    # For each spatial position, compute the conv output
    # Then apply softmax
    
    # First pass: compute conv outputs for all elements in the block
    # Also track the max for numerical stability
    
    # Offsets for spatial position
    # seq_idx = h * W + w
    # h = seq_idx // W, w = seq_idx % W
    
    # Storage for this block's values and exp sum
    offsets = seq_start + tl.arange(0, BLOCK_SIZE_N)
    mask = offsets < seq_len
    
    # Compute h, w from seq index
    h_coords = offsets // W
    w_coords = offsets % W
    
    # For each element in the block, compute conv output
    # output[b, seq] = sum_c(in[b, c, h, w] * weight[0, c, 0, 0]) + bias
    
    # We need to compute: exp(out[b, seq]) / sum_exp(out[b, :])
    # For numerical stability: exp(out - max_out)
    
    # First, compute all conv outputs and find max
    max_val = tl.full((BLOCK_SIZE_N,), float('-inf'), tl.float32)
    conv_outputs = tl.zeros((BLOCK_SIZE_N,), tl.float32)
    
    # Loop over channels
    for c in range(C):
        # Calculate input offset: [b, c, h, w]
        # offset = b * C * H * W + c * H * W + h * W + w
        input_offset = batch_offset + c * H * W + h_coords * W + w_coords
        
        # Load input value
        inp = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
        
        # Load weight [0, c, 0, 0]
        weight_offset = c * weight_stride_c
        w = tl.load(weight_ptr + weight_offset)
        
        # Accumulate conv output
        conv_outputs = conv_outputs + inp * w
    
    # Add bias
    conv_outputs = conv_outputs + bias
    
    # Update max for numerical stability
    max_val = tl.max(conv_outputs, axis=0)
    
    # Second pass: compute exp(x - max) and sum
    exp_vals = tl.exp(conv_outputs - max_val)
    exp_sum = tl.sum(exp_vals, axis=0)
    
    # Third pass: compute softmax and store
    softmax_vals = exp_vals / exp_sum
    
    # Store output: [B, 1, seq_len, 1] -> [B * seq_len]
    output_offset = pid_b * seq_len + offsets
    tl.store(output_ptr + output_offset, softmax_vals, mask=mask)


def fused_conv_softmax_kernel_v2(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, C, H, W, seq_len,
    input_stride_b, input_stride_c, input_stride_h, input_stride_w,
    weight_stride_c, weight_stride_h, weight_stride_w,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Simplified fused kernel with better memory access pattern.
    
    Process each batch element independently:
    1. Compute all conv outputs for the batch
    2. Apply softmax across sequence dimension
    """
    pid_b = tl.program_id(0)
    pid_seq = tl.program_id(1)
    
    batch_offset = pid_b * C * H * W
    seq_start = pid_seq * BLOCK_SIZE_N
    seq_end = min(seq_start + BLOCK_SIZE_N, seq_len)
    
    # Load bias once
    bias = tl.load(bias_ptr)
    
    # Compute offsets
    offsets = seq_start + tl.arange(0, BLOCK_SIZE_N)
    mask = offsets < seq_len
    h_coords = offsets // W
    w_coords = offsets % W
    
    # Compute conv outputs
    conv_out = tl.zeros((BLOCK_SIZE_N,), tl.float32)
    
    for c in range(C):
        input_offset = batch_offset + c * H * W + h_coords * W + w_coords
        inp = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
        w = tl.load(weight_ptr + c * weight_stride_c)
        conv_out = conv_out + inp * w
    
    conv_out = conv_out + bias
    
    # Softmax with numerical stability
    max_val = tl.max(conv_out, axis=0)
    exp_vals = tl.exp(conv_out - max_val)
    exp_sum = tl.sum(exp_vals, axis=0)
    softmax_vals = exp_vals / exp_sum
    
    # Store [B, 1, seq_len, 1]
    output_offset = pid_b * seq_len + offsets
    tl.store(output_ptr + output_offset, softmax_vals, mask=mask)


@torch.fx.wrap
def conv_softmax_kernel_wrapper(in_0, in_1, in_2):
    """
    Wrapper function that launches the fused Conv2D + Softmax kernel.
    
    Args:
        in_0: bias tensor, shape [1]
        in_1: weight tensor, shape [1, C, 1, 1]
        in_2: input tensor, shape [B, C, H, W]
    
    Returns:
        output tensor, shape [B, 1, seq_len, 1] where seq_len = H * W
    """
    B = in_2.shape[0]
    C = in_1.shape[1]
    H = in_2.shape[2]
    W = in_2.shape[3]
    seq_len = H * W
    
    # Allocate output
    output = torch.empty((B, 1, seq_len, 1), device=in_2.device, dtype=in_2.dtype)
    
    # Define block size based on seq_len
    # Use power of 2 for efficiency
    if seq_len <= 48:
        BLOCK_SIZE_N = 64
    elif seq_len <= 192:
        BLOCK_SIZE_N = 256
    elif seq_len <= 768:
        BLOCK_SIZE_N = 512
    elif seq_len <= 2048:
        BLOCK_SIZE_N = 1024
    else:
        BLOCK_SIZE_N = 2048
    
    # Calculate grid
    # Grid: (B, ceil(seq_len / BLOCK_SIZE_N))
    num_seq_blocks = (seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    fused_conv_softmax_kernel[(B, num_seq_blocks)](
        in_2, in_1, in_0,
        output,
        B, C, H, W, seq_len,
        in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
        in_1.stride(0), in_1.stride(1), in_1.stride(2),
        BLOCK_SIZE_N,
    )
    
    return output


def replacement_func():
    """Return the optimized kernel wrapper."""
    return conv_softmax_kernel_wrapper