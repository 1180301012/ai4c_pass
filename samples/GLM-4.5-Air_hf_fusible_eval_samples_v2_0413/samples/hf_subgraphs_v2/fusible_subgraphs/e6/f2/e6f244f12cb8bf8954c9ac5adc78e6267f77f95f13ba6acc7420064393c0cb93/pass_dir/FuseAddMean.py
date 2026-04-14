import torch
import triton
import triton.language as tl

def pattern(in_0=None, in_1=None, in_2=None):
    """
    Matches element-wise addition followed by mean computation patterns.
    Handles variants:
    - Single input with zeros: 0 + in_0; tmp_0 += 0
    - Two inputs: 0 + in_1; tmp_0 += in_0  
    - Three inputs: in_1 + in_2; tmp_0 += in_0
    """
    # Pattern to cover all three variants using exact operations from original code
    if in_1 is None and in_2 is None:
        # Single input variant: 0 + in_0; tmp_0 += 0
        tmp_0 = 0 + in_0
        tmp_0 += 0
        tmp_1 = tmp_0
    elif in_2 is None:
        # Two input variant: 0 + in_1; tmp_0 += in_0
        tmp_0 = 0 + in_1
        tmp_0 += in_0
        tmp_1 = tmp_0
    else:
        # Three input variant: in_1 + in_2; tmp_0 += in_0
        tmp_0 = in_1 + in_2
        tmp_0 += in_0
        tmp_1 = tmp_0
    
    # Compute mean on the sum result
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    
    return (tmp_1, tmp_2)

def replacement_args(in_0=None, in_1=None, in_2=None):
    """
    Extract arguments and route information for kernel dispatch.
    Route string determines which variant to compute in the kernel.
    """
    if in_1 is None and in_2 is None:
        return (in_0, "single_input")
    elif in_2 is None:
        return (in_0, in_1, "two_input")
    else:
        return (in_0, in_1, in_2, "three_input")

@triton.jit
def fused_add_mean_kernel_single_input(
    input_ptr,
    sum_ptr,
    mean_ptr,
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel for single input variant: 0 + input + 0"""
    pid = tl.program_id(0)
    grid_size = tl.program_count(0)
    
    # Each program processes one element in mean tensor shape (batch_size, channels, 1, 1)
    batch_idx = pid // channels
    channel_idx = pid % channels
    
    if batch_idx >= batch_size or channel_idx >= channels:
        return
    
    # Load all spatial elements for this batch and channel
    spatial_size = height * width
    sum_val = 0.0
    input_values = []
    
    for spatial_offset in range(spatial_size):
        element_offset = batch_idx * channels * spatial_size + channel_idx * spatial_size + spatial_offset
        x = tl.load(input_ptr + element_offset, other=0.0)
        input_values.append(x)
        sum_val += x
    
    # Store sum result (all elements - same as input for single input case)
    for spatial_offset, x_val in enumerate(input_values):
        element_offset = batch_idx * channels * spatial_size + channel_idx * spatial_size + spatial_offset
        tl.store(sum_ptr + element_offset, x_val)
    
    # Store mean result
    mean_val = sum_val / (height * width)
    mean_offset = batch_idx * channels + channel_idx
    tl.store(mean_ptr + mean_offset, mean_val)

@triton.jit
def fused_add_mean_kernel_multi_input(
    input0_ptr,
    input1_ptr,
    input2_ptr,
    sum_ptr,
    mean_ptr,
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel for multiple input variant: input0 + input1 + input2"""
    pid = tl.program_id(0)
    grid_size = tl.program_count(0)
    
    # Each program processes one element in mean tensor shape (batch_size, channels, 1, 1)
    batch_idx = pid // channels
    channel_idx = pid % channels
    
    if batch_idx >= batch_size or channel_idx >= channels:
        return
    
    # Load all spatial elements for this batch and channel
    spatial_size = height * width
    sum_val = 0.0
    local_sum = 0.0
    
    for spatial_offset in range(spatial_size):
        element_offset = batch_idx * channels * spatial_size + channel_idx * spatial_size + spatial_offset
        
        # Load all inputs
        x0 = tl.load(input0_ptr + element_offset, other=0.0)
        x1 = tl.load(input1_ptr + element_offset, other=0.0)
        x2 = tl.load(input2_ptr + element_offset, other=0.0)
        
        # Compute sum for this spatial location
        spatial_sum = x0 + x1 + x2
        
        # Store sum result (all spatial elements)
        tl.store(sum_ptr + element_offset, spatial_sum)
        
        # Accumulate for mean
        local_sum += spatial_sum
    
    # Store mean result
    mean_val = local_sum / (height * width)
    mean_offset = batch_idx * channels + channel_idx
    tl.store(mean_ptr + mean_offset, mean_val)

@torch.fx.wrap
def fused_add_mean_dispatch(*args, **kwargs):
    """
    Unified dispatch function that routes to appropriate kernel based on input pattern.
    This wrapper handles all variants in a single function.
    """
    # Extract arguments based on route string
    if len(args) == 2 and isinstance(args[1], str):
        # Two args: input and route string
        in_0, route_info = args
        in_1 = None
        in_2 = None
        if route_info == "single_input":
            input_tensor = in_0
        else:
            raise ValueError(f"Unknown route: {route_info}")
    elif len(args) == 3 and isinstance(args[2], str):
        # Three args: inputs and route string  
        in_0, in_1, route_info = args
        in_2 = None
        if route_info == "two_input":
            input_tensor = in_0 + in_1
        else:
            raise ValueError(f"Unknown route: {route_info}")
    elif len(args) == 4 and isinstance(args[3], str):
        # Four args: inputs and route string
        in_0, in_1, in_2, route_info = args
        if route_info == "three_input":
            input_tensor = in_0 + in_1 + in_2
        else:
            raise ValueError(f"Unknown route: {route_info}")
    else:
        raise ValueError(f"Invalid arguments: {args}")
    
    # Get tensor properties
    batch_size, channels, height, width = input_tensor.shape
    n_elements = batch_size * channels * height * width
    mean_elements = batch_size * channels  # (batch_size, channels, 1, 1)
    
    # Allocate output tensors
    sum_result = torch.empty_like(input_tensor)
    mean_result = torch.empty((batch_size, channels), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch appropriate kernel based on route string
    BLOCK_SIZE = 1024
    
    # Determine kernel variant from number of args
    if len(args) == 2:  # single input
        grid_size = mean_elements
        fused_add_mean_kernel_single_input[(grid_size,)](
            input_ptr=input_tensor,
            sum_ptr=sum_result,
            mean_ptr=mean_result,
            batch_size=batch_size,
            channels=channels,
            height=height,
            width=width,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    elif len(args) == 3:  # two inputs
        grid_size = mean_elements
        fused_add_mean_kernel_multi_input[(grid_size,)](
            input0_ptr=in_0,
            input1_ptr=in_1,
            input2_ptr=None,
            sum_ptr=sum_result,
            mean_ptr=mean_result,
            batch_size=batch_size,
            channels=channels,
            height=height,
            width=width,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:  # three inputs
        grid_size = mean_elements
        fused_add_mean_kernel_multi_input[(grid_size,)](
            input0_ptr=in_0,
            input1_ptr=in_1,
            input2_ptr=in_2,
            sum_ptr=sum_result,
            mean_ptr=mean_result,
            batch_size=batch_size,
            channels=channels,
            height=height,
            width=width,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    # Reshape mean result to match keepdim=True format
    mean_result = mean_result.reshape(batch_size, channels, 1, 1)
    
    return (sum_result, mean_result)

def replacement_func():
    """Return the dispatch function that handles all variants."""
    return fused_add_mean_dispatch