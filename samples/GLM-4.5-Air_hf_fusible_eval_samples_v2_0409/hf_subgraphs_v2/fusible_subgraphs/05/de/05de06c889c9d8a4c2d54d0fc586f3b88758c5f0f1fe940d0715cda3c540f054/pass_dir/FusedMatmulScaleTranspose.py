import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """
    Pattern matching: matmul + scalar multiply
    This matches the computation: (in_2 @ in_1) * in_0
    """
    matmul = torch.matmul(in_2, in_1)
    result = matmul * in_0
    return result

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_matmul_scale_kernel(
    out_ptr,
    in_0_val,
    in_1_ptr,
    in_1_stride,
    in_2_ptr,
    in_2_stride,
    batch_size: tl.constexpr,
    hidden_dim: tl.constexpr,
):
    """
    Optimized kernel that fuses:
    1. Matmul: [batch_size, hidden_dim] @ [hidden_dim, 1] -> [batch_size, 1]
    2. Scale: multiply by scalar logit_scale
    """
    pid = tl.program_id(0)
    
    # Each program handles one row of the result
    if pid >= batch_size:
        return
    
    # Compute dot product using scalar loads
    matmul_result = 0.0
    for i in range(hidden_dim):
        # Load elements from in_2 row and in_1 column
        val1 = tl.load(in_2_ptr + pid * in_2_stride + i)
        val2 = tl.load(in_1_ptr + i)
        matmul_result += val1 * val2
    
    # Apply scalar scaling
    result = matmul_result * in_0_val
    
    # Store the result: [batch_size, 1]
    tl.store(out_ptr + pid, result)

@torch.fx.wrap
def fused_matmul_scale(in_0, in_1, in_2):
    """
    Wrapper function that launches the fused kernel
    """
    # Get tensor shapes and properties
    batch_size = in_2.shape[0]  # Should be 2
    hidden_dim = in_2.shape[1]  # Should be 512
    
    # Get scalar value from in_0
    in_0_val = in_0.item()
    
    # Output tensor
    out_shape = (batch_size, 1)
    output = torch.empty(out_shape, dtype=in_2.dtype, device=in_2.device)
    
    # Configure grid based on batch size
    grid = (batch_size,)
    
    # Get tensor strides (in bytes)
    in_1_stride = in_1.stride(0) * in_1.element_size()
    in_2_stride = in_2.stride(0) * in_2.element_size()
    
    # Launch kernel
    fused_matmul_scale_kernel[grid](
        out_ptr=output,
        in_0_val=in_0_val,
        in_1_ptr=in_1,
        in_1_stride=in_1_stride,
        in_2_ptr=in_2,
        in_2_stride=in_2_stride,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
    )
    
    return output

def replacement_func():
    return fused_matmul_scale