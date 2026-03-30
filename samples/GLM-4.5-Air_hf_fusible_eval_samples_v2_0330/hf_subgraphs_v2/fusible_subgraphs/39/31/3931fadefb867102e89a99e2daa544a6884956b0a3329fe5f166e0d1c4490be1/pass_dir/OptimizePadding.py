import torch
import triton
import triton.language as tl

def pattern(x):
    """Match any padding operation for debugging"""
    # Try a more general padding pattern first
    return torch.nn.functional.pad(x, (0, 1), 'constant', 0.0)

def replacement_args(x):
    """Extract input tensor for padding operation"""
    return (x,)

@triton.jit
def pad_kernel(
    input_ptr,
    output_ptr,
    N: tl.constexpr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    F: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for padding (0, 0, 0, 1) - adds 1 element to second last dimension"""
    pid = tl.program_id(0)
    total_elements_out = N * C_out * F
    num_blocks = (total_elements_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    if pid < num_blocks:
        start_idx = pid * BLOCK_SIZE
        end_idx = min(start_idx + BLOCK_SIZE, total_elements_out)
        
        for idx in range(start_idx, end_idx):
            # Convert output index to (n, c, f) coordinates
            n_idx = idx // (C_out * F)
            temp_idx = idx % (C_out * F)
            c_idx = temp_idx // F
            f_idx = temp_idx % F
            
            # Determine if this position should copy input data or be padded
            if c_idx == 0 or c_idx == C_out - 1:
                # Padding positions - set to 0
                tl.store(output_ptr + idx, 0.0)
            else:
                # Copy from input, adjusting for padding
                input_n = n_idx
                input_c = c_idx - 1  # Remove the first padding row
                input_f = f_idx
                input_idx = input_n * (C_in * F) + input_c * F + input_f
                input_val = tl.load(input_ptr + input_idx, other=0.0)
                tl.store(output_ptr + idx, input_val)

@torch.fx.wrap
def optimized_pad(x):
    """Optimized padding for (0, 0, 0, 1) pattern"""
    # Input: [N, C_in, F], Output: [N, C_out, F] where C_out = C_in + 1
    N, C_in, F = x.shape
    C_out = C_in + 1
    
    # Check if we need to pad
    if C_out != C_in + 1:
        # This pass only handles the specific case where we add 1 to second last dimension
        # For other cases, we should raise an error or return input unchanged
        return x  # Return unchanged for unsupported padding patterns
    
    output = torch.empty((N, C_out, F), dtype=x.dtype, device=x.device)
    
    # Configure kernel launch
    BLOCK_SIZE = 1024
    total_elements_out = N * C_out * F
    num_programs = (total_elements_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    pad_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=output,
        N=N,
        C_in=C_in,
        C_out=C_out,
        F=F,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_pad