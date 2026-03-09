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
def fused_conv2d_view_softmax_kernel_b1(
    x_ptr,  # input [1, 512, 64, 64]
    w_ptr,  # weight [1, 512, 1, 1] 
    b_ptr,  # bias [1]
    z_ptr,  # output [1, 1, 4096] - flattened spatial dimension
    M: tl.constexpr,  # batch size = 1
    IC: tl.constexpr,  # input channels = 512  
    OC: tl.constexpr,  # output channels = 1
    H: tl.constexpr,  # height = 64
    W: tl.constexpr,  # width = 64
    BLOCK_SIZE: tl.constexpr,
):
    # Each program computes one spatial position and flattens it
    pid = tl.program_id(0)
    spatial_size = H * W
    if pid >= spatial_size:
        return
    
    # Get spatial coordinates
    h = pid // W
    w = pid % W
    
    # Compute 1x1 convolution for this spatial position
    conv_val = tl.load(b_ptr)
    for ic in range(IC):
        # Input: [1, 512, 64, 64] -> flatten to [1, 512*64*64]
        input_idx = ic * spatial_size + pid
        input_val = tl.load(x_ptr + input_idx)
        
        # Weight: [1, 512, 1, 1] -> flatten to [1*512] 
        weight_idx = ic
        weight_val = tl.load(w_ptr + weight_idx)
        
        conv_val += input_val * weight_val
    
    # Store result in flattened output tensor [1, 1, 4096]
    tl.store(z_ptr + pid, conv_val)

# Pattern for Model 0: batch size 1, conv 512->1, view to [1,1,4096], softmax
def pattern(input_tensor, weight_tensor, bias_tensor):
    tmp_0 = bias_tensor  # in_0 (bias)
    tmp_1 = weight_tensor  # in_1 (weight)
    tmp_2 = torch.conv2d(input_tensor, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2.view(1, 1, -1)
    tmp_4 = tmp_3.softmax(dim=-1)
    return tmp_4

def replacement_args(input_tensor, weight_tensor, bias_tensor):
    return (input_tensor, weight_tensor, bias_tensor)

@torch.fx.wrap
def fused_conv2d_view_softmax_b1(input, weight, bias):
    B, C_in, H, W = input.shape
    C_out = weight.shape[0]
    
    assert B == 1, "This kernel only supports batch size 1"
    assert C_out == 1, "This kernel only supports 1 output channel"
    assert H == W == 64, "This kernel only supports 64x64 spatial dimensions"
    
    # Flattened spatial size
    spatial_size = H * W
    
    # Create output buffer for convolution [1, 1, 4096]
    conv_output = torch.empty((B, 1, spatial_size), dtype=input.dtype, device=input.device)
    
    # Launch Triton convolution kernel
    BLOCK_SIZE = 1024
    num_programs = (spatial_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_conv2d_view_softmax_kernel_b1[(num_programs,)](
        x_ptr=input,
        w_ptr=weight,
        b_ptr=bias,
        z_ptr=conv_output,
        M=B,
        IC=C_in,
        OC=C_out,
        H=H,
        W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Apply softmax using Triton kernel on the flattened tensor [1, 1, 4096]
    softmax_output = torch.empty((B, 1, spatial_size), dtype=input.dtype, device=input.device)
    
    num_softmax_programs = (spatial_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    softmax_kernel[(num_softmax_programs,)](
        input_ptr=conv_output,
        output_ptr=softmax_output,
        n_elements=spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return softmax_output

def replacement_func():
    return fused_conv2d_view_softmax_b1