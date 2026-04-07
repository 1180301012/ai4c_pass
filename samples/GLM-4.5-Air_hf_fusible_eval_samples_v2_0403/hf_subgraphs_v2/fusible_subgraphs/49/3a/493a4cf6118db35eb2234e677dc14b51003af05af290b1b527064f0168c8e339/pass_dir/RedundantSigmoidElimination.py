import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """
    Pattern matches the computation graph with redundant sigmoid computation.
    The pattern identifies duplicate sigmoid operations on the same tensor.
    """
    tmp_1 = in_2.softmax(dim=-1)
    tmp_2 = in_0.view(1, -1, 1, 1)
    tmp_3 = torch.sigmoid(tmp_2)
    tmp_4 = 1.0 - tmp_3
    tmp_5 = tmp_4 * in_1
    tmp_6 = torch.sigmoid(tmp_2)  # Redundant - same as tmp_3
    tmp_7 = tmp_6 * tmp_1
    tmp_8 = tmp_5 + tmp_7
    return (tmp_8,)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def optimized_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr,
    out_ptr,
    in_0_size, batch_size, channels, height, width,
    is_fp32: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel that eliminates redundant sigmoid computation.
    Uses mathematical identity: 1 - sigmoid(x) = sigmoid(-x)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Calculate indices for the 4D output tensor [batch, channels, height, width]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * channels * height * width)
    
    # Convert linear index to 4D coordinates
    idx = offsets
    h_idx = idx // (channels * width) % height
    w_idx = idx % width
    c_idx = (idx // width) % channels
    b_idx = idx // (channels * height * width)
    
    # Load gating parameter and cast appropriately
    gating_val = tl.load(in_0_ptr + c_idx)
    if is_fp32:
        gating_fp32 = gating_val
    else:
        gating_fp32 = tl.cast(gating_val, tl.float32)  # Cast to fp32 for sigmoid
    
    # Compute sigmoid once (instead of twice) in fp32
    sigmoid_fp32 = tl.sigmoid(gating_fp32)
    inv_sigmoid_fp32 = tl.sigmoid(-gating_fp32)  # 1 - sigmoid(x) = sigmoid(-x)
    
    # Cast back to original dtype
    if is_fp32:
        sigmoid_val = sigmoid_fp32
        inv_sigmoid_val = inv_sigmoid_fp32
    else:
        sigmoid_val = tl.cast(sigmoid_fp32, tl.float16)
        inv_sigmoid_val = tl.cast(inv_sigmoid_fp32, tl.float16)
    
    # Load input tensors for broadcasting
    softmax_val = tl.load(in_1_ptr + offsets, mask=mask)
    
    # Compute the two branches in fp32 for precision, then cast back
    if is_fp32:
        sigmoid_bcast = sigmoid_val
        inv_sigmoid_bcast = inv_sigmoid_val
    else:
        sigmoid_bcast = tl.cast(sigmoid_val, tl.float32)
        inv_sigmoid_bcast = tl.cast(inv_sigmoid_val, tl.float32)
    
    # Perform the computation
    branch1 = inv_sigmoid_bcast * softmax_val
    branch2 = sigmoid_bcast * softmax_val
    result = branch1 + branch2
    
    # Cast result back to original dtype if needed
    if not is_fp32:
        result = tl.cast(result, tl.float16)
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_kernel_wrapper(in_0, in_1, in_2):
    """
    Wrapper function to launch the optimized kernel.
    Handles different input sizes and launches appropriate number of programs.
    """
    # Get input shapes
    in_1_shape = in_1.shape  # [1, 16, 196, 196]
    batch_size, channels, height, width = in_1_shape
    
    # Determine if we're using fp32
    is_fp32 = in_1.dtype == torch.float32
    
    # Calculate total elements
    total_elements = batch_size * channels * height * width
    
    # Optimize block size based on data type and tensor size
    if is_fp32:
        BLOCK_SIZE = 1024  # Larger block for fp32
    else:
        BLOCK_SIZE = 2048  # Larger block for fp16/bfloat16 for better occupancy
    
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(in_1)
    
    # Launch kernel with appropriate grid size
    optimized_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        out_ptr=out,
        in_0_size=in_0.size(0),
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        is_fp32=is_fp32,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_kernel_wrapper