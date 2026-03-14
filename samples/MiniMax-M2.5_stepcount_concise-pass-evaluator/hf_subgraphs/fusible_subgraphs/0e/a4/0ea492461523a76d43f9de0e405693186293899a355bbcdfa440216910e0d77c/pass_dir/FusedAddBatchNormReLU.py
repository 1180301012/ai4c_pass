import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Pattern: Fused Add + BatchNorm + ReLU
    
    The computation:
    1. in_5 += in_4  (element-wise add)
    2. batch_norm(in_5, running_mean, running_var, weight, bias, ...)
    3. relu(batch_norm_output, inplace=True)
    
    Returns (input_after_add, relu_output)
    """
    # Step 1: Element-wise addition
    in_5 += in_4
    tmp_4 = in_5
    
    # Step 2: Batch normalization
    # F.batch_norm(input, running_mean, running_var, weight, bias, 
    #              training=False, momentum=0.1, eps=1e-05)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    
    # Step 3: ReLU activation (inplace)
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=True)
    
    return tmp_4, tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Extract arguments for the replacement kernel.
    - in_0: running_mean (1D, shape [512])
    - in_1: running_var (1D, shape [512]) 
    - in_2: bias (1D, shape [512])
    - in_3: weight (1D, shape [512])
    - in_4: input to add (4D, shape [batch, 512, 8, 8])
    - in_5: input to add (4D, shape [batch, 512, 8, 8])
    """
    return (in_0, in_1, in_2, in_3, in_4, in_5)


# Autotune configurations for different block sizes
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=4, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def fused_add_bn_relu_kernel(
    # Input pointers
    in_5_ptr, in_4_ptr,
    # BatchNorm parameters
    running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    # Output pointers  
    output_ptr, output_add_ptr,
    # Dimensions
    N, C, H, W,
    # Constants
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. out = in_5 + in_4  (element-wise)
    2. out = (out - running_mean) / sqrt(running_var + eps) * weight + bias
    3. out = relu(out)
    
    All in a single kernel launch.
    """
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate offset for this program
    num_elements = N * C * H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Calculate channel index for batchnorm params
    # Each element belongs to a channel
    c = (offsets // (H * W)) % C
    
    # Load inputs (in_5 and in_4)
    in_5 = tl.load(in_5_ptr + offsets, mask=mask, other=0.0)
    in_4 = tl.load(in_4_ptr + offsets, mask=mask, other=0.0)
    
    # Step 1: Element-wise addition
    added = in_5 + in_4
    
    # Load batch norm parameters (indexed by channel)
    # running_mean and running_var are 1D [C]
    running_mean = tl.load(running_mean_ptr + c)
    running_var = tl.load(running_var_ptr + c)
    weight = tl.load(weight_ptr + c)
    bias = tl.load(bias_ptr + c)
    
    # Step 2: Batch normalization (inference mode)
    # output = (input - mean) / sqrt(var + eps) * weight + bias
    normalized = (added - running_mean) / tl.sqrt(running_var + eps)
    bn_out = normalized * weight + bias
    
    # Step 3: ReLU activation (inplace)
    relu_out = tl.where(bn_out > 0, bn_out, 0.0)
    
    # Store results
    # output_ptr gets the relu result (tmp_6)
    # output_add_ptr gets the added result (tmp_4)
    tl.store(output_ptr + offsets, relu_out, mask=mask)
    tl.store(output_add_ptr + offsets, added, mask=mask)


@torch.fx.wrap
def fused_add_bn_relu_wrapper(
    running_mean, running_var, weight, bias, 
    in_4, in_5
):
    """
    Wrapper function that launches the fused Triton kernel.
    
    Args:
        running_mean: Running mean, shape [C]
        running_var: Running variance, shape [C]
        weight: BatchNorm weight (gamma), shape [C]
        bias: BatchNorm bias (beta), shape [C]
        in_4: Input tensor to add, shape [B, C, H, W]
        in_5: Input tensor to add, shape [B, C, H, W]
    
    Returns:
        Tuple of (added_input, relu_output)
        - added_input: in_5 + in_4 (for the model's first return value)
        - relu_output: ReLU(batch_norm(added_input)) (for the model's second return value)
    """
    # Get dimensions
    B, C, H, W = in_5.shape
    N = B * C * H * W
    
    # Allocate output tensors
    output = torch.empty_like(in_5)  # For relu output
    output_add = torch.empty_like(in_5)  # For added input (tmp_4)
    
    # Choose block size based on tensor size
    BLOCK_SIZE = 4096
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_add_bn_relu_kernel[(num_programs,)](
        in_5_ptr=in_5,
        in_4_ptr=in_4,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        output_add_ptr=output_add,
        N=N,
        C=C,
        H=H,
        W=W,
        eps=1e-05,
    )
    
    return output_add, output


def replacement_func():
    """Return the replacement function."""
    return fused_add_bn_relu_wrapper