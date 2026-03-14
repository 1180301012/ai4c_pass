import torch
import triton
import triton.language as tl

# Pattern to match: ReLU -> BatchNorm -> Dropout(p=0)
# The dropout with p=0 is a no-op, so we can eliminate it
# The main fusion opportunity is ReLU + BatchNorm

def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Match the pattern: ReLU -> BatchNorm -> Dropout(p=0.0)
    
    in_0: running mean [128]
    in_1: running var [128]
    in_2: bias [128]
    in_3: weight [128]
    in_4: input tensor [1000, 128]
    
    Returns the output after dropout (which is just the batchnorm output)
    """
    # ReLU activation
    tmp_4 = torch.nn.functional.relu(in_4, inplace=False)
    
    # BatchNorm with running mean/var, weight, bias
    # batch_norm(input, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=1e-05)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    
    # Dropout with p=0.0 - this is a no-op during inference
    tmp_6 = torch.nn.functional.dropout(tmp_5, p=0.0, training=False)
    
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    """
    Extract arguments needed for the replacement kernel.
    
    in_0: running mean
    in_1: running var  
    in_2: bias
    in_3: weight
    in_4: input tensor
    """
    return (in_0, in_1, in_2, in_3, in_4)


@triton.jit
def fused_relu_batch_norm_kernel(
    # Input pointers
    input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    # Output pointer
    output_ptr,
    # Sizes
    N: tl.constexpr,  # feature dimension (128)
    M: tl.constexpr,  # batch dimension (1000)
    # Batch norm parameters
    eps: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused ReLU + BatchNorm kernel.
    
    Computes: relu(x) -> batch_norm(..., running_mean, running_var, weight, bias)
    
    Since BatchNorm with training=False uses running statistics:
    y = (x - running_mean) / sqrt(running_var + eps) * weight + bias
    
    We fuse ReLU with BatchNorm to avoid an extra kernel launch and memory read/write.
    """
    # Each program processes a row (batch element)
    row_idx = tl.program_id(0)
    
    # Calculate row offset
    row_offset = row_idx * N
    
    # Load running statistics (they're small, 128 elements)
    # We'll load all N elements per row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load input row
    input_ptrs = input_ptr + row_offset + offsets
    x = tl.load(input_ptrs, mask=mask, other=0.0)
    
    # Load running mean
    mean_offsets = offsets
    mean = tl.load(running_mean_ptr + mean_offsets, mask=mask, other=0.0)
    
    # Load running var
    var_offsets = offsets
    var = tl.load(running_var_ptr + var_offsets, mask=mask, other=0.0)
    
    # Load weight
    weight_offsets = offsets
    weight = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0)
    
    # Load bias
    bias_offsets = offsets
    bias = tl.load(bias_ptr + bias_offsets, mask=mask, other=0.0)
    
    # Step 1: ReLU activation: max(0, x)
    x = tl.where(x < 0, 0.0, x)
    
    # Step 2: BatchNorm (inference mode with running stats)
    # y = ((x - mean) / sqrt(var + eps)) * weight + bias
    x_norm = (x - mean) / tl.sqrt(var + eps)
    y = x_norm * weight + bias
    
    # Store result
    output_ptrs = output_ptr + row_offset + offsets
    tl.store(output_ptrs, y, mask=mask)


@torch.fx.wrap
def fused_relu_batch_norm_kernel_wrapper(in_0, in_1, in_2, in_3, in_4):
    """
    Wrapper function for the fused ReLU + BatchNorm kernel.
    
    in_0: running mean [128]
    in_1: running var [128]
    in_2: bias [128]
    in_3: weight [128]
    in_4: input tensor [1000, 128]
    
    Returns: output tensor [1000, 128]
    """
    # Get dimensions
    M, N = in_4.shape  # M=1000 (batch), N=128 (features)
    
    # Prepare inputs - ensure they are on the same device
    # Convert running stats to cuda if needed
    device = in_4.device
    running_mean = in_0.to(device=device)
    running_var = in_1.to(device=device)
    weight = in_3.to(device=device)
    bias = in_2.to(device=device)
    
    # Allocate output
    output = torch.empty_like(in_4)
    
    # Configure block size
    BLOCK_SIZE = 128  # Match the feature dimension
    
    # Launch kernel - one program per row
    grid = (M,)
    
    fused_relu_batch_norm_kernel[grid](
        in_4, running_mean, running_var, weight, bias,
        output,
        N,  # feature dimension
        M,  # batch dimension
        1e-05,  # eps
        BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    """
    Return the replacement function.
    This fuses ReLU + BatchNorm into a single kernel and eliminates the no-op dropout.
    """
    return fused_relu_batch_norm_kernel_wrapper