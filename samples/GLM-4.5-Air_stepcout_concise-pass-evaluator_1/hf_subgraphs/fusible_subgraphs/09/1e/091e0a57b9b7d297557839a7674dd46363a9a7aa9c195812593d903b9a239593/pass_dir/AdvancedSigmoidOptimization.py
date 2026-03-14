import torch
import triton
import triton.language as tl

@triton.jit
def advanced_sigmoid_kernel(
    sigmoid_input_ptr,
    x1_ptr,
    x0_ptr,
    output_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
):
    # Each program handles a spatial location
    pid = tl.program_id(0)
    total_spatial_elements = H * W
    offsets = pid + tl.arange(0, BATCH_SIZE * total_spatial_elements)
    mask = offsets < (BATCH_SIZE * C * H * W)
    
    # Load sigmoid input for spatial broadcasting
    sigmoid_idx = (offsets // total_spatial_elements) % C
    sigmoid_values = tl.load(sigmoid_input_ptr + sigmoid_idx, mask=sigmoid_idx < C)
    
    # Load input tensors
    x1_data = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x0_data = tl.load(x0_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sigmoid for each element
    sigmoid_expanded = tl.where(sigmoid_values >= 0, 
                               1 / (1 + tl.exp(-sigmoid_values)), 
                               tl.exp(sigmoid_values) / (1 + tl.exp(sigmoid_values)))
    
    # Fused operations: sigmoid * x1 + x0
    result = sigmoid_expanded * x1_data + x0_data
    
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def advanced_sigmoid_optimization(x2, x1, x0):
    N, C, H, W = x1.shape
    
    output = torch.empty_like(x1)
    
    # Optimized launch with better grid configuration
    grid = (N * C * H * W,)
    
    advanced_sigmoid_kernel[grid](
        sigmoid_input_ptr=x2.view(-1),  # Flatten [1, 1, C] to [C]
        x1_ptr=x1.view(-1),            # Flatten [N, C, H, W] to [N*C*H*W]
        x0_ptr=x0.view(-1),            # Flatten [N, C, H, W] to [N*C*H*W]
        output_ptr=output.view(-1),    # Flatten for output
        C=C,
        H=H,
        W=W,
        BATCH_SIZE=N,
    )
    
    return output

def pattern(in_2, in_1, in_0):
    """Match sigmoid -> view -> expand -> multiply-add"""
    tmp_0 = in_2.sigmoid()
    tmp_1 = tmp_0.view(1, -1, 1, 1)
    tmp_2 = tmp_1.expand_as(in_1)
    tmp_3 = in_1 * tmp_2
    tmp_3 = tmp_3 + in_0
    return tmp_3

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

def replacement_func():
    return advanced_sigmoid_optimization