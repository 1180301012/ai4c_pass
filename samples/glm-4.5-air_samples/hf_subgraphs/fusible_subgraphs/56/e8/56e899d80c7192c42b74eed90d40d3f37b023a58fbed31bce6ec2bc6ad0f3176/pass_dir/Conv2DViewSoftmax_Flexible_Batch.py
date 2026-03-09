import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Load chunks of data
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability (simple max without mask)
    max_val = tl.max(x)
    tl.debug_barrier()
    
    # Compute exponentials, handling mask by using 0.0 for out-of-bounds elements
    exp_x = tl.where(mask, tl.exp(x - max_val), 0.0)
    
    # Compute sum (simple sum without mask, since masked elements are 0.0)
    sum_exp = tl.sum(exp_x)
    tl.debug_barrier()
    
    # Avoid division by zero
    sum_exp = tl.where(sum_exp == 0.0, 1.0, sum_exp)
    
    # Compute softmax, handling mask by using 0.0 for out-of-bounds elements
    softmax_out = tl.where(mask, exp_x / sum_exp, 0.0)
    
    # Store results
    tl.store(output_ptr + offsets, softmax_out, mask=mask)

@triton.jit
def fused_conv2d_view_softmax_kernel(
    x_ptr,  # input [B, 512, 64, 64]
    w_ptr,  # weight [1, 512, 1, 1] 
    b_ptr,  # bias [1]
    z_ptr,  # output [B, 1, 4096] - flattened spatial dimension
    B: tl.constexpr,  # batch size
    IC: tl.constexpr,  # input channels = 512  
    OC: tl.constexpr,  # output channels = 1
    H: tl.constexpr,  # height = 64
    W: tl.constexpr,  # width = 64
    BLOCK_SIZE: tl.constexpr,
):
    # Each program computes one spatial position for one batch element
    pid = tl.program_id(0)
    spatial_size = H * W
    
    if pid >= B * spatial_size:
        return
    
    # Get batch index and spatial coordinates
    batch_idx = pid // spatial_size
    spatial_idx = pid % spatial_size
    
    # Compute 1x1 convolution for this batch and spatial position
    conv_val = tl.load(b_ptr)
    
    # Load weight for this spatial position and all channels
    # Weight is [1, 512, 1, 1] -> flatten to [512]
    for ic in range(IC):
        weight_idx = ic
        weight_val = tl.load(w_ptr + weight_idx)
        
        # Compute input index: [batch, channel, h, w] -> flattened
        # Input: [B, 512, 64, 64] -> [B*512*64*64]
        input_idx = batch_idx * (IC * spatial_size) + ic * spatial_size + spatial_idx
        input_val = tl.load(x_ptr + input_idx)
        
        conv_val += input_val * weight_val
    
    # Store result in flattened output tensor [B, 1, 4096]
    tl.store(z_ptr + pid, conv_val)

# Pattern for flexible batch size: conv 512->1, view to flatten, softmax
def pattern(in_0, in_1, in_2):
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2.view(-1, -1)  # Flexible view: [batch, channels, height, width] -> [batch, channels*height*width]
    tmp_4 = tmp_3.softmax(dim=-1)
    return tmp_4

def replacement_args(input_tensor, weight_tensor, bias_tensor):
    return (input_tensor, weight_tensor, bias_tensor)

@torch.fx.wrap
def fused_conv2d_view_softmax_flexible(input, weight, bias):
    B, C_in, H, W = input.shape
    C_out = weight.shape[0]
    
    # Support flexible batch sizes while keeping constraints
    assert C_in == 512, f"This kernel only supports 512 input channels, got {C_in}"
    assert C_out == 1, f"This kernel only supports 1 output channel, got {C_out}"
    assert H == W == 64, f"This kernel only supports 64x64 spatial dimensions, got {H}x{W}"
    
    # Flattened spatial size per batch element
    spatial_size = H * W
    
    # Create output buffer for convolution [B, 1, 4096]
    conv_output = torch.empty((B, 1, spatial_size), dtype=input.dtype, device=input.device)
    
    # Launch Triton convolution kernel
    BLOCK_SIZE = 1024
    total_elements = B * spatial_size
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_conv2d_view_softmax_kernel[(num_programs,)](
        x_ptr=input,
        w_ptr=weight,
        b_ptr=bias,
        z_ptr=conv_output,
        B=B,
        IC=C_in,
        OC=C_out,
        H=H,
        W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Apply softmax using Triton kernel on each batch element [B, 1, 4096]
    softmax_output = torch.empty((B, 1, spatial_size), dtype=input.dtype, device=input.device)
    
    # Launch softmax kernel for each batch element separately
    for i in range(B):
        element_conv_output = conv_output[i, 0, :]  # [4096]
        element_softmax_output = torch.empty(spatial_size, dtype=input.dtype, device=input.device)
        
        num_softmax_programs = (spatial_size + BLOCK_SIZE - 1) // BLOCK_SIZE
        softmax_kernel[(num_softmax_programs,)](
            input_ptr=element_conv_output,
            output_ptr=element_softmax_output,
            n_elements=spatial_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        softmax_output[i, 0, :] = element_softmax_output
    
    return softmax_output

def replacement_func():
    return fused_conv2d_view_softmax_flexible