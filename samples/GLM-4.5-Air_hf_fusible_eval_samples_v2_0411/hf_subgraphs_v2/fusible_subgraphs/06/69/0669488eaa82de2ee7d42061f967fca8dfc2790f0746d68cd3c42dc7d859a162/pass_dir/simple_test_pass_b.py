import torch
import triton
import triton.language as tl

def pattern(tmp_2, tmp_3):
    """Pattern for: norm * 0.07216878364870322 -> clamp -> divide (type B)"""
    # Match the sequence with hardcoded constant (type B): tmp_4 = tmp_3 * 0.07216878364870322; tmp_3 = None
    tmp_4 = tmp_3 * 0.07216878364870322;  tmp_3 = None
    tmp_5 = tmp_4.clamp(min = 1e-05);  tmp_4 = None  
    tmp_6 = tmp_2 / tmp_5;  tmp_2 = tmp_5 = None
    return tmp_6

def replacement_args(tmp_2, tmp_3):
    return (tmp_2, tmp_3)

@triton.jit
def fused_norm_op_kernel_b(
    input_ptr,      # tmp_2 (flattened tensor after ReLU)
    norm_ptr,       # tmp_3 (norm tensor)
    output_ptr,     # tmp_6 (result)
    n_samples,      # Number of samples
    total_elements, # Total elements per sample
    BLOCK_SIZE: tl.constexpr,
):
    # 1D grid - each program handles one element
    pid = tl.program_id(0)
    stride = tl.num_programs(0)
    
    # Process multiple elements per program for better occupancy
    for i in range(pid, n_samples * total_elements, stride):
        # Determine which sample and which position in that sample
        sample_idx = i // total_elements
        element_idx = i % total_elements
        
        # Load input value and norm - let Triton infer the data type
        x = tl.load(input_ptr + i)
        norm_val = tl.load(norm_ptr + sample_idx)
        
        # Apply fused operations: (x / (norm_val * 0.07216878364870322)).clamp(min=1e-05)
        # Use explicit conversion to match computation precision
        const = tl.constexpr(0.07216878364870322)
        scaled_norm = norm_val * const
        epsilon = tl.where(scaled_norm < 1e-05, tl.constexpr(1e-05), scaled_norm)
        result = x / epsilon
        
        # Store result
        tl.store(output_ptr + i, result)

@torch.fx.wrap  
def fused_norm_operations_b(input_tensor, norm_tensor):
    """Fused: divide by (norm * 0.07216878364870322) with clamping (type B)"""
    
    # Get dimensions - handle both 2D and higher dimensional tensors
    input_shape = input_tensor.shape
    n_samples = input_shape[0]
    total_elements = input_tensor.numel() // n_samples  # Elements per sample
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Launch kernel
    BLOCK_SIZE = 256
    total_elements_all = n_samples * total_elements
    num_programs = (total_elements_all + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    if n_samples > 0 and total_elements > 0:
        fused_norm_op_kernel_b[(num_programs,)](
            input_ptr=input_tensor,
            norm_ptr=norm_tensor,
            output_ptr=output,
            n_samples=n_samples,
            total_elements=total_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return output

def replacement_func():
    return fused_norm_operations_b