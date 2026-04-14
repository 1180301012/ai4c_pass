import torch
import triton
import triton.language as tl

# Pattern matching function - optimized reshape operation
def pattern(input_tensor):
    """
    Match the final reshape operation with optimization potential
    The reshape operation converts unfolded tensor [1, 512, 256] to [1, 128, 4, 256]
    We can optimize this by using more efficient memory layout
    """
    # The final reshape operation
    result = input_tensor.reshape(1, 128, 4, -1)
    return (result,)

# Argument extraction function
def replacement_args(input_tensor):
    # Our pattern only needs the tensor that gets reshaped
    return (input_tensor,)

# Optimized kernel - fused conv1x1 + unfold
@triton.jit
def fused_conv1x1_unfold_kernel(
    in_ptr,  # [B, C_in, H, W] = [1, 256, 32, 32]
    weight_ptr,  # [C_out, C_in, 1, 1] = [128, 256, 1, 1]
    out_ptr,  # [B, C_out, patch_h, patch_w, num_patches] = [1, 128, 2, 2, 256]
    B: tl.constexpr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride: tl.constexpr,
    patch_h: tl.constexpr,
    patch_w: tl.constexpr,
    num_patches: tl.constexpr
):
    # Each program handles one patch (one spatial location in unfolded output)
    patch_idx = tl.program_id(0)
    
    if patch_idx >= num_patches:
        return
        
    # Calculate spatial position of this patch
    patch_y = patch_idx // (W // stride)
    patch_x = patch_idx % (W // stride)
    
    # Iterate over output channels and patch elements
    for c_out in range(C_out):
        # For each output channel, compute the patch
        for py in range(patch_h):
            for px in range(patch_w):
                # Global coordinates in original conv2d output
                gy = patch_y * stride + py
                gx = patch_x * stride + px
                
                # Compute one element of the patch
                if gy < H and gx < W:
                    # Conv1x1 operation: weighted sum of input channels
                    sum_val = tl.zeros([tl.float32], dtype=tl.float32)
                    for c_in in range(C_in):
                        # Load input and weight values
                        in_val = tl.load(in_ptr + c_in * H * W + gy * W + gx, mask=(gy < H) & (gx < W))
                        weight_val = tl.load(weight_ptr + c_out * C_in + c_in, mask=(c_out < C_out) & (c_in < C_in))
                        sum_val += in_val * weight_val
                    
                    # Store the result in the output patch
                    patch_element_idx = c_out * (patch_h * patch_w) + py * patch_w + px
                    tl.store(out_ptr + patch_idx * (C_out * patch_h * patch_w) + patch_element_idx, sum_val)
                else:
                    # Store zero for out-of-bounds elements
                    patch_element_idx = c_out * (patch_h * patch_w) + py * patch_w + px
                    tl.store(out_ptr + patch_idx * (C_out * patch_h * patch_w) + patch_element_idx, 0.0)

# Triton kernel for reshape operation
@triton.jit
def reshape_kernel(
    input_ptr,
    output_ptr,
    input_size: tl.constexpr,
    output_dims: tl.constexpr,
):
    # Each program handles one element
    pid = tl.program_id(0)
    if pid >= input_size:
        return
    
    # Copy element from input to output
    # We need to calculate the output tensor shape: [1, 128, 4, -1]
    # Let the caller handle the size calculation
    tl.store(output_ptr + pid, tl.load(input_ptr + pid))

# Replacement function for reshape operation using allowed APIs
@torch.fx.wrap
def reshape_with_allowed_apis(input_tensor):
    # Calculate final shape - output should be [1, 128, 4, -1]
    # Since -1 is calculated automatically, we need to determine the last dimension
    total_elements = input_tensor.numel()
    # 1 * 128 * 4 * last_dim = total_elements
    last_dim = total_elements // (1 * 128 * 4)
    
    # Create output tensor with correct shape using allowed API
    output_tensor = torch.empty([1, 128, 4, last_dim], dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate sizes
    input_size = total_elements
    output_dims = (1, 128, 4, last_dim)  # Use tuple instead of list for hashability
    
    # Launch kernel to copy data
    grid = (input_size,)
    reshape_kernel[grid](
        input_tensor,
        output_tensor,
        input_size,
        output_dims
    )
    
    return output_tensor

# Optimized fused kernel - Conv1x1 + Unfold + Reshape
@triton.jit
def fused_optimized_kernel(
    input_ptr,     # [B, C_in, H, W] = [1, 256, 32, 32]
    weight_ptr,    # [C_out, C_in, 1, 1] = [128, 256, 1, 1]  
    output_ptr,    # [1, 128, 4, 256] - final reshaped output
    B: tl.constexpr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr
):
    # Each program outputs one patch at one spatial location
    # We want to output [1, 128, 4, 256] where 4=2*2 (patch size) and 256=num_patches
    patch_idx = tl.program_id(0)  # 0 to 255
    
    if patch_idx >= 256:  # Number of patches: 16x16
        return
    
    # Calculate spatial position of this patch
    patch_y = patch_idx // 16  # 16 patches per dimension
    patch_x = patch_idx % 16
    
    # Output tensor has shape [1, C_out, 4, 256]
    # For each output channel, compute the 2x2 patch
    for c_out in range(C_out):
        for patch_element in range(4):  # 2x2 = 4 elements
            py = patch_element // 2  # y position within patch
            px = patch_element % 2   # x position within patch
            
            # Global coordinate in original conv2d output
            gy = patch_y * 2 + py  # stride=2
            gx = patch_x * 2 + px
            
            if gy < H and gx < W:
                # Compute conv1x1: weighted sum of input channels
                sum_val = 0.0
                for c_in in range(C_in):
                    # Load input value
                    in_val = tl.load(input_ptr + c_in * H * W + gy * W + gx, 
                                   mask=(c_in < C_in) & (gy < H) & (gx < W))
                    # Load weight value  
                    weight_val = tl.load(weight_ptr + c_out * C_in + c_in,
                                       mask=(c_out < C_out) & (c_in < C_in))
                    sum_val += in_val * weight_val
                
                # Store in final output: [1, C_out, 4, 256]
                output_idx = c_out * (4 * 256) + patch_element * 256 + patch_idx
                tl.store(output_ptr + output_idx, sum_val)
            else:
                # Store zero for out-of-bounds elements
                output_idx = c_out * (4 * 256) + patch_element * 256 + patch_idx
                tl.store(output_ptr + output_idx, 0.0)

# Replacement function that implements the fused operation
@torch.fx.wrap
def fused_conv1x1_unfold_optimized(in_1, in_0):
    # Get input shapes
    B, C_in, H, W = in_1.shape      # [1, 256, 32, 32]
    C_out, _, _, _ = in_0.shape     # [128, 256, 1, 1]
    
    # Final output shape: [1, 128, 4, 256]
    final_shape = [1, C_out, 4, 256]
    total_elements = 1 * C_out * 4 * 256
    
    # Create output tensor using allowed API
    output_tensor = torch.empty(total_elements, dtype=in_1.dtype, device=in_1.device)
    
    # Launch the fused kernel
    grid = (256,)  # 16x16 = 256 patches
    fused_optimized_kernel[grid](
        in_1,
        in_0,
        output_tensor,
        B, C_in, C_out, H, W
    )
    
    # Reshape to final output format
    result = output_tensor.reshape(final_shape)
    return result

# Optimized fused kernel - Conv1x1 + Unfold (without reshape)
@triton.jit
def fused_conv1x1_unfold_kernel(
    input_ptr,     # [B, C_in, H, W] = [1, 256, 32, 32]
    weight_ptr,    # [C_out, C_in, 1, 1] = [128, 256, 1, 1]  
    output_ptr,    # [1, 512, 256] - unfolded output
    B: tl.constexpr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr
):
    # Each program computes one patch at one spatial location
    patch_idx = tl.program_id(0)  # 0 to 255
    if patch_idx >= 256:  
        return
    
    # Calculate spatial position of this patch
    patch_y = patch_idx // 16  # 16 patches per dimension  
    patch_x = patch_idx % 16
    
    # For this patch, compute all output channels and patch elements
    for c_out in range(C_out):  # 128 output channels
        for py in range(2):      # 2x2 patch
            for px in range(2):  
                # Global coordinate in conv2d output
                gy = patch_y * 2 + py  # stride=2
                gx = patch_x * 2 + px
                
                if gy < H and gx < W:
                    # Compute conv1x1: weighted sum of input channels
                    sum_val = 0.0
                    for c_in in range(C_in):
                        # Load input and weight values
                        in_val = tl.load(input_ptr + c_in * H * W + gy * W + gx, 
                                       mask=(c_in < C_in) & (gy < H) & (gx < W))
                        weight_val = tl.load(weight_ptr + c_out * C_in + c_in,
                                           mask=(c_out < C_out) & (c_in < C_in))
                        sum_val += in_val * weight_val
                    
                    # Store in unfolded output: [1, 512, 256]
                    patch_element_idx = c_out * 4 + py * 2 + px  # 4 elements per patch
                    unfolded_idx = patch_element_idx * 256 + patch_idx
                    tl.store(output_ptr + unfolded_idx, sum_val)
                else:
                    # Store zero for out-of-bounds
                    patch_element_idx = c_out * 4 + py * 2 + px
                    unfolded_idx = patch_element_idx * 256 + patch_idx
                    tl.store(output_ptr + unfolded_idx, 0.0)

# Replacement function that implements the fused conv2d → unfold
@torch.fx.wrap
def fused_conv1x1_unfold(in_1, in_0):
    # Get input shapes
    B, C_in, H, W = in_1.shape      # [1, 256, 32, 32]
    C_out, _, _, _ = in_0.shape     # [128, 256, 1, 1]
    
    # Unfolded output shape: [1, C_out*2*2, 16*16] = [1, 512, 256]
    unfolded_shape = [1, C_out * 4, 256]  # 4=2*2 elements per patch
    total_elements = 1 * C_out * 4 * 256
    
    # Create output tensor using allowed API
    output_tensor = torch.empty(total_elements, dtype=in_1.dtype, device=in_1.device)
    
    # Launch the fused kernel
    grid = (256,)  # 16x16 = 256 patches
    fused_conv1x1_unfold_kernel[grid](
        in_1,
        in_0,
        output_tensor,
        B, C_in, C_out, H, W
    )
    
    # Reshape to unfolded format [1, 512, 256]
    result = output_tensor.reshape(unfolded_shape)
    return result

# Optimized reshape kernel using larger blocks for efficiency
@triton.jit
def optimized_reshape_kernel(
    input_ptr,
    output_ptr, 
    input_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input_size
    
    # Load block of input data
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Store block of output data (no transformation needed, just memory copy)
    tl.store(output_ptr + offsets, input_data, mask=mask)

# Optimized replacement function
@torch.fx.wrap
def optimized_reshape_replacement(input_tensor):
    # Calculate output shape
    total_elements = input_tensor.numel()
    last_dim = total_elements // (1 * 128 * 4)  # Calculate -1 dimension
    output_shape = [1, 128, 4, last_dim]
    
    # Create output tensor with correct shape using allowed API
    output_tensor = torch.empty(total_elements, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Use efficient block-based copying instead of element-by-element
    BLOCK_SIZE = 1024  # Larger block size for better efficiency
    num_blocks = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch optimized kernel
    optimized_reshape_kernel[(num_blocks,)](
        input_tensor,
        output_tensor,
        total_elements,
        BLOCK_SIZE
    )
    
    # Reshape to final format
    result = output_tensor.reshape(output_shape)
    return result

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_reshape_replacement