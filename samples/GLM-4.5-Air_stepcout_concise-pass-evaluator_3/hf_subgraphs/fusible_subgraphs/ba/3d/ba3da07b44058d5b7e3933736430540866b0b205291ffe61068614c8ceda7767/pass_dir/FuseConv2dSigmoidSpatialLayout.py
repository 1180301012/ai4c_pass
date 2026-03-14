import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # Pattern matching the computation:
    # Conv2D + Permute + Reshape + Sigmoid
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2.permute(0, 2, 3, 1)
    tmp_4 = tmp_3.reshape(24, -1, 36)
    tmp_5 = torch.nn.functional.sigmoid(tmp_4)
    return tmp_5

def replacement_args(in_0, in_1, in_2):
    # Extract arguments needed for the fused kernel
    return (in_0, in_1, in_2)

@triton.jit
def fused_conv_sigmoid_kernel(
    input_ptr,
    weight_ptr, 
    bias_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Get program indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute range for this program
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    
    # Create masks for bounds checking
    m_mask = m_offset < batch_size * height * width
    n_mask = n_offset < out_channels
    
    # Compute linear indices for output [batch, height*width, out_channels]
    global_idx = m_offset * out_channels + n_offset
    m_idx = tl.arange(0, BLOCK_SIZE_M)
    n_idx = tl.arange(0, BLOCK_SIZE_N)
    m_mask_local = m_idx < (batch_size * height * width - m_offset)
    n_mask_local = n_idx < (out_channels - n_offset)
    
    # Load bias for output channels
    bias = tl.load(bias_ptr + n_offset, mask=n_mask_local, other=0.0)
    bias = tl.broadcast_to(bias, [BLOCK_SIZE_M, BLOCK_SIZE_N])
    
    # Compute input pointer offset for this spatial position
    spatial_idx = m_offset // out_channels
    batch_idx = spatial_idx // (height * width)
    spatial_remainder = spatial_idx % (height * width)
    h_idx = spatial_remainder // width
    w_idx = spatial_remainder % width
    
    # Load weights for this output channel (1x1 conv -> out_channels x in_channels)
    weight_ptrs = weight_ptr + tl.arange(0, BLOCK_SIZE_K)[:, None] * out_channels + tl.arange(0, BLOCK_SIZE_K)[None, :] * in_channels * out_channels + n_offset
    weight = tl.load(weight_ptrs, mask=n_mask_local[:, None], other=0.0)
    
    # Load input patch for this spatial position (1x1 conv single position)
    input_ptrs = input_ptr + batch_idx * in_channels * height * width + h_idx * in_channels * width + w_idx * in_channels + tl.arange(0, BLOCK_SIZE_K)[None, :]
    input_val = tl.load(input_ptrs, mask=m_mask_local[:, None], other=0.0)
    
    # Matrix multiplication (1x1 conv is essentially GEMM)
    output = tl.dot(input_val, weight.to(tl.float32))
    output = output + bias
    
    # Apply sigmoid activation
    output = 1.0 / (1.0 + tl.exp(-output))
    
    # Store result
    output_ptrs = output_ptr + global_idx
    tl.store(output_ptrs, output, mask=tl.outer(m_mask_local, n_mask_local))

@torch.fx.wrap
def fused_conv_sigmoid(in_0, in_1, in_2):
    # Input shapes
    batch_size, in_channels, height, width = in_2.shape
    out_channels = in_0.shape[0]
    
    # Output shape [batch_size, height*width, out_channels]
    output_shape = (batch_size, height * width, out_channels)
    output = torch.empty(output_shape, device=in_2.device, dtype=torch.float32)
    
    # Set up block sizes for GPU optimization
    BLOCK_SIZE_M = 64  # spatial positions per thread block
    BLOCK_SIZE_N = 32  # output channels per thread block  
    BLOCK_SIZE_K = 256  # input channels per thread block
    
    # Calculate grid size
    spatial_positions = batch_size * height * width
    num_programs_m = (spatial_positions + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_programs_n = (out_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch Triton kernel
    fused_conv_sigmoid_kernel[(num_programs_m, num_programs_n)](
        input_ptr=in_2,
        weight_ptr=in_1,
        bias_ptr=in_0,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output

def replacement_func():
    return fused_conv_sigmoid