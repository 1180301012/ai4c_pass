import torch
import triton
import triton.language as tl

@triton.jit
def fused_unfold_kernel(
    input_ptr,
    output_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H_padded: tl.constexpr,
    W_padded: tl.constexpr,
    H_out_1: tl.constexpr,
    W_out_1: tl.constexpr,
    H_out_2: tl.constexpr,
    W_out_2: tl.constexpr,
    window_size: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Improved fused kernel that performs two unfold operations.
    This implementation handles multiple window positions correctly.
    """
    pid = tl.program_id(0)
    
    # Calculate total output positions including window elements
    # Each "position" now includes all window elements
    output_positions = N * C * H_out_1 * W_out_1 * H_out_2
    start_offset = pid * BLOCK_SIZE
    offsets = start_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_positions
    
    if tl.sum(mask) > 0:
        # Convert offset to coordinates: [N, C, H_out_1, W_out_1, H_out_2]
        stride_N = C * H_out_1 * W_out_1 * H_out_2
        stride_C = H_out_1 * W_out_1 * H_out_2
        stride_H1 = W_out_1 * H_out_2
        stride_W1 = H_out_2
        
        n = offsets // stride_N
        rem = offsets % stride_N
        c = rem // stride_C
        rem = rem % stride_C
        h1 = rem // stride_H1
        rem = rem % stride_H1
        w1 = rem // stride_W1
        rem = rem % stride_W1
        
        # H_out_2 represents the flat window index (0 to window_size*window_size-1)
        window_idx = rem
        
        # Convert flat window index to 2D coordinates within the window
        h_win = window_idx // window_size
        w_win = window_idx % window_size
        
        # Convert output coordinates to input coordinates for the sliding window
        # h1, w1 are the starting positions of the windows in the padded tensor
        h_padded = h1 * stride + h_win
        w_padded = w1 * stride + w_win
        
        # Create input offset for this window position
        input_offset = n * C * H_padded * W_padded + c * H_padded * W_padded + h_padded * W_padded + w_padded
        
        # Load input data and store to output
        output_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
        tl.store(output_ptr + offsets, output_val, mask=mask)

@torch.fx.wrap
def fused_unfold_operations(input_tensor):
    """
    Fused implementation of two unfold operations:
    unfold(2, 12, 8) followed by unfold(3, 12, 8)
    """
    # Get input dimensions (input is already padded)
    N, C, H_padded, W_padded = input_tensor.shape
    
    # Calculate output dimensions for both unfold operations
    window_size = 12
    stride = 8
    
    # First unfold operation (along dimension 2 - height)
    H_out_1 = (H_padded - window_size) // stride + 1
    
    # Second unfold operation (along dimension 3 - width) 
    W_out_1 = (W_padded - window_size) // stride + 1
    
    # In the improved kernel, H_out_2 represents flat window indices
    # We need to set H_out_2 to window_size*window_size to cover all positions
    H_out_2 = window_size * window_size
    W_out_2 = 1  # Not used in kernel logic, but kept for compatibility
    
    # Calculate total output elements
    total_elements = N * C * H_out_1 * W_out_1 * H_out_2
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with correct final shape
    final_output_shape = (N, C, H_out_1, W_out_1, window_size, window_size)
    output = torch.empty(final_output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    fused_unfold_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        N=N,
        C=C,
        H_padded=H_padded,
        W_padded=W_padded,
        H_out_1=H_out_1,
        W_out_1=W_out_1,
        H_out_2=H_out_2,
        W_out_2=W_out_2,
        window_size=window_size,
        stride=stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def pattern(tmp_2):
    """
    Pattern matching two consecutive unfold operations:
    tmp_3 = tmp_2.unfold(2, 12, 8)
    tmp_4 = tmp_3.unfold(3, 12, 8)
    """
    # First unfold along dimension 2
    tmp_3 = tmp_2.unfold(2, 12, 8)
    
    # Second unfold along dimension 3
    tmp_4 = tmp_3.unfold(3, 12, 8)
    
    return tmp_4

def replacement_args(tmp_2):
    return (tmp_2,)

def replacement_func():
    return fused_unfold_operations