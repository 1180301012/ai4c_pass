import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    # Pattern matching - only include operations, no cleanup statements
    tmp_0 = torch.cat([in_0, in_1, in_2], dim=1)
    tmp_1 = tmp_0.reshape(1, 8, -1, -1)  # Variable reshape dimensions
    tmp_2 = tmp_1.transpose(-1, -2)
    tmp_3 = in_3 * tmp_2
    tmp_4 = torch.nn.functional.pad(tmp_3, (0, 0, 1, 0, 0, 0), 'constant', None)
    return (tmp_4,)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Optimized kernel using Triton
@triton.jit
def fused_kernel(
    in0_ptr, in1_ptr, in2_ptr, in3_ptr,
    out_ptr,
    in0_offset, in1_offset, in2_offset, in3_offset,
    in0_stride_0, in0_stride_1, in0_stride_2, in0_stride_3,
    in1_stride_0, in1_stride_1, in1_stride_2, in1_stride_3,
    in2_stride_0, in2_stride_1, in2_stride_2, in2_stride_3,
    in3_stride_0, in3_stride_1, in3_stride_2, in3_stride_3,
    out_stride_0, out_stride_1, out_stride_2, out_stride_3,
    cat_channels_0, cat_channels_1, cat_channels_2,
    kernel_height, kernel_width,
    in3_height, in3_width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program id and grid setup
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Block offsets
    m_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for bounds checking
    m_mask = m_offset < kernel_height
    n_mask = n_offset < kernel_width
    mask = m_mask[:, None] & n_mask[None, :]
    
    # Input dimensions
    batch_size = 1
    num_features = 8
    
    # Compute source positions for the concatenated inputs
    # Total concatenated channels
    total_channels = cat_channels_0 + cat_channels_1 + cat_channels_2
    
    # For each spatial position, compute the corresponding positions in input tensors
    # and perform the equivalent of concat + reshape + transpose
    out_height = kernel_height + 1  # Accounting for padding
    out_width = kernel_width
    
    # For each output position (padded height, width)
    for h in range(kernel_height + 1):
        for w in range(kernel_width):
            if h >= kernel_height:
                # Padding case - set to zeros
                for c in range(num_features):
                    offset = out_ptr + (c * out_stride_1 + h * out_stride_2 + w * out_stride_3)
                    tl.store(offset, 0.0)
            else:
                # Normal computation
                # Map (h, w) back to source positions in concatenated tensor
                src_h = h
                src_w = w
                
                # Load from input tensors and concatenate in memory
                # Simulate concatenation by computing offsets directly
                # We need to compute which input tensor each feature comes from
                # This is simplified - in a real implementation we'd need more sophisticated indexing
                
                # For now, let's create a more efficient approach that works with the input shapes
                pass

# Simpler and more efficient approach
@triton.jit
def fused_kernel_simple(
    in0_ptr, in1_ptr, in2_ptr, in3_ptr,
    out_ptr,
    in0_stride_0, in0_stride_1, in0_stride_2, in0_stride_3,
    in1_stride_0, in1_stride_1, in1_stride_2, in1_stride_3,
    in2_stride_0, in2_stride_1, in2_stride_2, in2_stride_3,
    in3_stride_0, in3_stride_1, in3_stride_2, in3_stride_3,
    out_stride_0, out_stride_1, out_stride_2, out_stride_3,
    cat_ch0, cat_ch1, cat_ch2,
    src_height, src_width,
    in3_height, in3_width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Linear index for the output tensor (padded height * width)
    total_elements = (src_height + 1) * src_width
    linear_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = linear_idx < total_elements
    
    # Convert linear index to (h, w) coordinates
    h = linear_idx // src_width
    w = linear_idx % src_width
    
    for idx, h_val, w_val in zip(tl.arange(0, BLOCK_SIZE), h, w):
        if mask[idx]:
            if h_val < src_height:
                # For each feature channel
                for c in range(8):
                    # Compute source position in concatenated tensor
                    src_flat_idx = c * (src_height * src_width) + h_val * src_width + w_val
                    
                    # Determine which input tensor this comes from
                    remaining_idx = src_flat_idx
                    if remaining_idx < cat_ch0 * src_height * src_width:
                        # From in_0
                        ch_idx = remaining_idx // (src_height * src_width)
                        spatial_idx = remaining_idx % (src_height * src_width)
                        h_src = spatial_idx // src_width
                        w_src = spatial_idx % src_width
                        val = tl.load(in0_ptr + ch_idx * in0_stride_1 + h_src * in0_stride_2 + w_src * in0_stride_3)
                    elif remaining_idx < (cat_ch0 + cat_ch1) * src_height * src_width:
                        # From in_1
                        remaining_idx -= cat_ch0 * src_height * src_width
                        ch_idx = remaining_idx // (src_height * src_width)
                        spatial_idx = remaining_idx % (src_height * src_width)
                        h_src = spatial_idx // src_width
                        w_src = spatial_idx % src_width
                        val = tl.load(in1_ptr + ch_idx * in1_stride_1 + h_src * in1_stride_2 + w_src * in1_stride_3)
                    else:
                        # From in_2
                        remaining_idx -= (cat_ch0 + cat_ch1) * src_height * src_width
                        ch_idx = remaining_idx // (src_height * src_width)
                        spatial_idx = remaining_idx % (src_height * src_width)
                        h_src = spatial_idx // src_width
                        w_src = spatial_idx % src_width
                        val = tl.load(in2_ptr + ch_idx * in2_stride_1 + h_src * in2_stride_2 + w_src * in2_stride_3)
                    
                    # After transpose operation (w, h -> h, w)
                    # Multiply with in_3
                    in3_val = tl.load(in3_ptr + c * in3_stride_1 + w_val * in3_stride_2 + h_val * in3_stride_3)
                    result = val * in3_val
                    
                    # Store result with padding
                    out_offset = out_ptr + c * out_stride_1 + h_val * out_stride_2 + w_val * out_stride_3
                    tl.store(out_offset, result)
            else:
                # Padding row - set to zeros
                for c in range(8):
                    out_offset = out_ptr + c * out_stride_1 + h_val * out_stride_2 + w_val * out_stride_3
                    tl.store(out_offset, 0.0)

@torch.fx.wrap
def fused_operation(in_0, in_1, in_2, in_3):
    # Get input tensor properties
    device = in_0.device
    
    # Calculate output shape
    total_cat_channels = in_0.shape[1] + in_1.shape[1] + in_2.shape[1]
    src_height = in_0.shape[2]
    src_width = in_0.shape[3]
    
    # Output shape after padding: (1, 8, src_height + 1, src_width)
    out_shape = (1, 8, src_height + 1, src_width)
    output = torch.empty(out_shape, dtype=torch.float32, device=device)
    
    # Get tensor strides
    in0_stride = list(in_0.stride())
    in1_stride = list(in_1.stride())
    in2_stride = list(in_2.stride())
    in3_stride = list(in_3.stride())
    out_stride = list(output.stride())
    
    # Pad strides to 4D (even if some are 1)
    while len(in0_stride) < 4:
        in0_stride.insert(0, 1)
    while len(in1_stride) < 4:
        in1_stride.insert(0, 1)
    while len(in2_stride) < 4:
        in2_stride.insert(0, 1)
    while len(in3_stride) < 4:
        in3_stride.insert(0, 1)
    
    # Calculate total elements for grid
    total_elements = (src_height + 1) * src_width
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_kernel_simple[(num_programs,)](
        in0_ptr=in_0,
        in1_ptr=in_1,
        in2_ptr=in_2,
        in3_ptr=in_3,
        out_ptr=output,
        in0_stride_0=in0_stride[0], in0_stride_1=in0_stride[1], in0_stride_2=in0_stride[2], in0_stride_3=in0_stride[3],
        in1_stride_0=in1_stride[0], in1_stride_1=in1_stride[1], in1_stride_2=in1_stride[2], in1_stride_3=in1_stride[3],
        in2_stride_0=in2_stride[0], in2_stride_1=in2_stride[1], in2_stride_2=in2_stride[2], in2_stride_3=in2_stride[3],
        in3_stride_0=in3_stride[0], in3_stride_1=in3_stride[1], in3_stride_2=in3_stride[2], in3_stride_3=in3_stride[3],
        out_stride_0=out_stride[0], out_stride_1=out_stride[1], out_stride_2=out_stride[2], out_stride_3=out_stride[3],
        cat_ch0=in_0.shape[1], cat_ch1=in_1.shape[1], cat_ch2=in_2.shape[1],
        src_height=src_height, src_width=src_width,
        in3_height=in_3.shape[2], in3_width=in_3.shape[3],
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_operation