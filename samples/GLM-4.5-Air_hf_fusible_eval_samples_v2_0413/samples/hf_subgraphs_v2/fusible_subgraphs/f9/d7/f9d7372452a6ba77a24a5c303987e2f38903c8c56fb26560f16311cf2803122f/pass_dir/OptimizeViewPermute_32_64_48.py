import torch
import triton
import triton.language as tl

@triton.jit
def optimized_view_permute_kernel(
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    features: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    spatial_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel to transform tensor from [B, F, H, W] to [B, H*W, F]"""
    pid = tl.program_id(0)
    
    # Calculate range of elements each program handles
    num_elements = batch_size * features * spatial_size
    elements_per_program = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    start_element = pid * elements_per_program
    end_element = min(start_element + elements_per_program, num_elements)
    
    if start_element >= num_elements:
        return
    
    # Process elements in this program
    for idx in range(start_element, end_element):
        # Calculate indices for output tensor [B, spatial_size, F]
        b = idx // (features * spatial_size)
        remaining = idx % (features * spatial_size)
        f = remaining // spatial_size
        s = remaining % spatial_size  # flattened spatial position
        
        # Calculate source indices in input tensor [B, F, H, W]
        h = s // input_width  # Correct division by actual width (48)
        w = s % input_width   # Remainder gives actual width (48)
        
        # Calculate source offset (row-major order for input tensor)
        src_offset = (b * features * input_height * input_width) + \
                     (f * input_height * input_width) + \
                     (h * input_width) + w
        
        # Calculate destination offset (row-major order for output tensor [B, spatial_size, F])
        dst_offset = (b * features * spatial_size) + \
                     (s * features) + f
        
        # Load and store
        if src_offset < batch_size * features * input_height * input_width:
            value = tl.load(input_ptr + src_offset)
            tl.store(output_ptr + dst_offset, value)

@torch.fx.wrap
def optimized_view_permute(x):
    """Optimized view(1, 32, -1) followed by permute(0, 2, 1) operation"""
    # Input shape: [1, 32, 64, 48] 
    # Target shape: [1, 3072, 32] (view + permute)
    
    batch_size, features, height, width = x.shape  # [1, 32, 64, 48]
    spatial_size = height * width  # 64 * 48 = 3072
    
    # Create output tensor with target shape [1, 3072, 32]
    output_shape = (batch_size, spatial_size, features)
    result = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Calculate total elements and configure kernel launch
    total_elements = batch_size * features * spatial_size
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the custom Triton kernel
    optimized_view_permute_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=result,
        batch_size=batch_size,
        features=features,
        input_height=height,
        input_width=width,
        spatial_size=spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return result

def pattern(in_1):
    """Match the pattern: tmp_3 = in_1.view(1, 32, -1); tmp_4 = tmp_3.permute(0, 2, 1); return tmp_4"""
    # Simulate the exact operations from the model
    tmp_3 = in_1.view(1, 32, -1)
    tmp_4 = tmp_3.permute(0, 2, 1)
    
    # Return tmp_4 as it's the observable output
    return tmp_4

def replacement_args(in_1):
    """Extract the input tensor for the view+permute operation"""
    return (in_1,)

def replacement_func():
    """Return the optimized kernel function"""
    return optimized_view_permute