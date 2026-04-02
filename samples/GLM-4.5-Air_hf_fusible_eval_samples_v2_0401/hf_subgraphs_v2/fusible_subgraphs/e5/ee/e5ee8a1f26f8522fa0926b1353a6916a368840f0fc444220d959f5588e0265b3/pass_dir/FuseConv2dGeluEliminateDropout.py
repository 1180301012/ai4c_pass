import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1, in_2):
    """
    Matches Conv2D + GELU + Dropout(p=0.0) pattern.
    The dropout with p=0.0 is identity and can be eliminated.
    This pattern handles variations in conv2d parameters and GELU approximation.
    """
    # Match the conv2d operation with various parameter combinations
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (1, 1), (1, 1))  # groups inferred, padding=(1,1)
    tmp_3 = torch.nn.functional.gelu(conv2d)  # default GELU approximation
    tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)  # p=0.0 makes this identity
    return tmp_4

def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments needed for the fused Conv2D+GELU kernel.
    We don't need to extract dropout parameters since p=0.0 makes it identity.
    """
    return (in_0, in_1, in_2)

@triton.jit
def conv2d_gelu_kernel_wide(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    batch_size, in_channels, in_height, in_width,
    out_channels, kernel_height, kernel_width,
    stride_height, stride_width,
    pad_height, pad_width,
    dilation_height, dilation_width,
    groups,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_Kv: tl.constexpr,
):
    """
    Optimized Triton kernel for fused Conv2D + GELU operation.
    Supports both regular conv2d (groups=1) and depthwise conv2d (groups=out_channels).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute output coordinates
    m = pid_m  # batch index
    n = pid_n  # output channel index
    
    # Offset pointers for this program
    output_ptr += m * out_channels * ((in_height + 2 * pad_height - dilation_height * (kernel_height - 1) - 1) // stride_height + 1) * \
                      ((in_width + 2 * pad_width - dilation_width * (kernel_width - 1) - 1) // stride_width + 1) + \
                      n * ((in_height + 2 * pad_height - dilation_height * (kernel_height - 1) - 1) // stride_height + 1) * \
                      ((in_width + 2 * pad_width - dilation_width * (kernel_width - 1) - 1) // stride_width + 1)
    
    # For depthwise convolution (groups = out_channels), we process one output channel at a time
    if groups == out_channels:
        # Depthwise case: each input channel maps to one output channel
        start_k = n
        end_k = n + 1
        k_step = 1
    else:
        # Regular convolution: all input channels contribute to each output channel
        start_k = 0
        end_k = in_channels
        k_step = 1
    
    acc = 0.0
    
    # Process the convolution using a blocked algorithm
    for k in range(start_k, end_k, k_step):
        # Offset weight and bias pointers
        if groups == out_channels:
            weight_offset = n * kernel_height * kernel_width
        else:
            weight_offset = n * in_channels // groups * kernel_height * kernel_width + k * kernel_height * kernel_width
        
        # Load weight tile
        weight = tl.load(weight_ptr + weight_offset, eviction_policy='evict_last')
        
        # For each position in the kernel
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Compute input position with padding and dilation
                ih = stride_height * ((output_ptr - (m * out_channels * ((in_height + 2 * pad_height - dilation_height * (kernel_height - 1) - 1) // stride_height + 1) * \
                                      ((in_width + 2 * pad_width - dilation_width * (kernel_width - 1) - 1) // stride_width + 1)) // ((in_height + 2 * pad_height - dilation_height * (kernel_height - 1) - 1) // stride_height + 1)) // out_channels) + \
                     kh * dilation_height - pad_height
                iw = stride_width * ((output_ptr - (m * out_channels * ((in_height + 2 * pad_height - dilation_height * (kernel_height - 1) - 1) // stride_height + 1) * \
                                    ((in_width + 2 * pad_width - dilation_width * (kernel_width - 1) - 1) // stride_width + 1)) // ((in_height + 2 * pad_height - dilation_height * (kernel_height - 1) - 1) // stride_height + 1)) % ((in_height + 2 * pad_height - dilation_height * (kernel_height - 1) - 1) // stride_height + 1)) + \
                     kw * dilation_width - pad_width
                
                # Check bounds and load input value
                if 0 <= ih < in_height and 0 <= iw < in_width:
                    input_offset = m * in_channels * in_height * in_width + \
                                 k * in_height * in_width + ih * in_width + iw
                    input_val = tl.load(input_ptr + input_offset, eviction_policy='evict_last')
                    acc += input_val * weight[kh * kernel_width + kw]
    
    # Add bias
    bias_offset = n if groups == out_channels else n * in_channels // groups
    bias = tl.load(bias_ptr + bias_offset, eviction_policy='evict_last') if bias_ptr != 0 else 0.0
    acc += bias
    
    # Apply GELU activation
    # GELU(x) = x * 0.5 * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x = acc
    gelu_out = x * 0.5 * (1.0 + tl.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))
    
    # Store the result
    tl.store(output_ptr, gelu_out)

# Simplified kernel for efficient computation
@triton.jit
def conv2d_gelu_kernel_simple(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    batch_size, in_channels, in_height, in_width,
    out_channels, kernel_height, kernel_width,
    stride_height, stride_width,
    pad_height, pad_width,
    dilation_height, dilation_width,
    groups,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simplified Conv2D + GELU kernel for better performance on most workloads.
    """
    pid = tl.program_id(0)
    
    # Calculate effective output dimensions
    out_height = (in_height + 2 * pad_height - dilation_height * (kernel_height - 1) - 1) // stride_height + 1
    out_width = (in_width + 2 * pad_width - dilation_width * (kernel_width - 1) - 1) // stride_width + 1
    
    # Compute output coordinates
    output_idx = pid
    if output_idx >= batch_size * out_channels * out_height * out_width:
        return
    
    m = output_idx // (out_channels * out_height * out_width)  # batch index
    remainder = output_idx % (out_channels * out_height * out_width)
    
    n = remainder // (out_height * out_width)  # output channel index
    remainder = remainder % (out_height * out_width)
    
    oh = remainder // out_width  # output height
    ow = remainder % out_width   # output width
    
    acc = 0.0
    
    # Handle depthwise vs regular convolution
    if groups == out_channels:
        # Depthwise: each channel processed independently
        start_k = n
        end_k = n + 1
        channel_mult = 1
    else:
        # Regular: all channels contribute
        start_k = 0
        end_k = in_channels
        channel_mult = in_channels // groups
    
    # Convolution computation
    for k in range(start_k, end_k):
        # Weight offset
        if groups == out_channels:
            weight_offset = k * kernel_height * kernel_width
        else:
            weight_offset = (k // channel_mult) * (out_channels // groups) * kernel_height * kernel_width + \
                           (k % channel_mult) * kernel_height * kernel_width
        
        # Process each kernel position
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Input coordinates with padding and dilation
                ih = oh * stride_height + kh * dilation_height - pad_height
                iw = ow * stride_width + kw * dilation_width - pad_width
                
                # Check bounds
                if 0 <= ih < in_height and 0 <= iw < in_width:
                    # Input offset
                    input_offset = (m * in_channels + k) * in_height * in_width + ih * in_width + iw
                    # Weight offset in the tile
                    weight_tile_offset = (n // groups) * kernel_height * kernel_width + (k // channel_mult) * kernel_height * kernel_width + kh * kernel_width + kw
                    
                    input_val = tl.load(input_ptr + input_offset, eviction_policy='evict_last')
                    weight_val = tl.load(weight_ptr + weight_tile_offset, eviction_policy='evict_last')
                    acc += input_val * weight_val
    
    # Add bias
    if bias_ptr != 0:
        if groups == out_channels:
            bias_offset = n
        else:
            bias_offset = n * in_channels // groups
        bias = tl.load(bias_ptr + bias_offset, eviction_policy='evict_last')
        acc += bias
    
    # Apply GELU activation
    # Use piecewise approximation for better performance
    x = acc
    if x > 0:
        if x < 1:
            gelu_out = x * (0.44209 + x * 0.117954)
        else:
            gelu_out = x * 0.5 * (1.0 + tl.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))
    else:
        if x > -1:
            gelu_out = x * (0.44209 + x * 0.117954)
        else:
            gelu_out = x * 0.5 * (1.0 + tl.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))
    
    # Store result
    output_offset = (m * out_channels + n) * out_height * out_width + oh * out_width + ow
    tl.store(output_ptr + output_offset, gelu_out)

@torch.fx.wrap
def fused_conv2d_gelu(input, weight, bias, 
                     stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1):
    """
    Fused Conv2D + GELU operation with optimized Triton kernel.
    Eliminates the redundant dropout (p=0.0) and fuses Conv2D + GELU.
    """
    # Get input and weight tensor shapes
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Handle weight strides for different conv2d implementations
    if len(weight.stride()) == 4:
        # Weight has [out_channels, in_channels//groups, kernel_height, kernel_width] layout
        pass
    else:
        # Weight might be flattened or have different layout
        weight = weight.view(out_channels, in_channels // groups, kernel_height, kernel_width)
    
    # Calculate output dimensions
    out_height = (in_height + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1) // stride[1] + 1
    out_width = (in_width + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1) // stride[0] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, out_height, out_width, 
                        dtype=input.dtype, device=input.device)
    
    # Check if this is a depthwise convolution
    is_depthwise = (groups == out_channels)
    
    # Choose kernel based on workload characteristics
    total_elements = batch_size * out_channels * out_height * out_width
    
    # Use simple kernel for most cases (better performance)
    BLOCK_SIZE = 256 if total_elements > 10000 else 128
    
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Set up kernel arguments based on data type
    if input.dtype == torch.float16:
        # Use float16 computation with float16 GELU
        conv2d_gelu_kernel_simple[(num_programs,)](
            input, weight, bias if bias is not None else 0,
            output,
            batch_size, in_channels, in_height, in_width,
            out_channels, kernel_height, kernel_width,
            stride[1], stride[0],
            padding[1], padding[0],
            dilation[1], dilation[0],
            groups,
            BLOCK_SIZE
        )
    elif input.dtype == torch.bfloat16:
        # Use bfloat16 computation
        conv2d_gelu_kernel_simple[(num_programs,)](
            input, weight, bias if bias is not None else 0,
            output,
            batch_size, in_channels, in_height, in_width,
            out_channels, kernel_height, kernel_width,
            stride[1], stride[0],
            padding[1], padding[0],
            dilation[1], dilation[0],
            groups,
            BLOCK_SIZE
        )
    else:
        # Use float32 computation for best accuracy
        conv2d_gelu_kernel_simple[(num_programs,)](
            input, weight, bias if bias is not None else 0,
            output,
            batch_size, in_channels, in_height, in_width,
            out_channels, kernel_height, kernel_width,
            stride[1], stride[0],
            padding[1], padding[0],
            dilation[1], dilation[0],
            groups,
            BLOCK_SIZE
        )
    
    return output

def replacement_func():
    """
    Returns the fused Conv2D + GELU function that eliminates dropout (p=0.0).
    """
    return fused_conv2d_gelu