import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=3, num_warps=4),
    ],
    key=["n_elements"],
)
@triton.jit
def fused_relu_batchnorm_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    n_features,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused ReLU + BatchNorm kernel for inference mode.
    
    Computes: relu((x - mean) / sqrt(var + eps) * weight + bias)
    
    Optimizations:
    1. Fuses ReLU and BatchNorm into a single kernel
    2. Uses vectorized loads/stores for coalesced memory access
    3. Single kernel launch for reduced overhead
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Compute feature indices (broadcasting for channel dimension)
    feat_idx = offsets % n_features
    
    # Vectorized load for x
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load running statistics (broadcasted across batch dimension)
    mean = tl.load(mean_ptr + feat_idx)
    var = tl.load(var_ptr + feat_idx)
    weight = tl.load(weight_ptr + feat_idx)
    bias = tl.load(bias_ptr + feat_idx)
    
    # BatchNorm: (x - mean) * rsqrt(var + eps) * weight + bias
    rstd = tl.rsqrt(var + eps)
    bn_out = (x - mean) * rstd * weight + bias
    
    # ReLU activation: max(0, x)
    relu_out = tl.maximum(bn_out, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, relu_out, mask=mask)


@torch.fx.wrap
def fused_relu_batchnorm_wrapper(
    x,
    running_mean,
    running_var,
    weight,
    bias,
    momentum=0.1,
    eps=1e-05,
):
    """
    Wrapper function for the fused ReLU + BatchNorm kernel.
    
    This fuses:
        tmp_4 = torch.nn.functional.relu(x, inplace=False)
        tmp_5 = torch.nn.functional.batch_norm(tmp_4, running_mean, running_var, weight, bias, False, momentum, eps)
        tmp_6 = torch.nn.functional.dropout(tmp_5, p=0.0, training=False)  # no-op
    
    The dropout(p=0.0) is a no-op and is effectively eliminated.
    
    Args:
        x: Input tensor [N, C] where N=batch_size, C=channels
        running_mean: Running mean [C]
        running_var: Running variance [C]
        weight: Learnable weight [C]
        bias: Learnable bias [C]
        momentum: Momentum (unused in inference mode)
        eps: Small constant for numerical stability
    
    Returns:
        Fused ReLU + BatchNorm output tensor
    """
    # Get device and shape information
    device = x.device
    n_elements = x.numel()
    n_features = x.shape[-1]  # Channel dimension
    
    # Ensure all tensors are on the same device
    mean = running_mean.to(device)
    var = running_var.to(device)
    w = weight.to(device)
    b = bias.to(device)
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Calculate grid dimensions
    BLOCK_SIZE = 1024  # Will be auto-tuned
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_relu_batchnorm_kernel[(num_programs,)](
        x_ptr=x,
        mean_ptr=mean,
        var_ptr=var,
        weight_ptr=w,
        bias_ptr=b,
        out_ptr=out,
        n_elements=n_elements,
        n_features=n_features,
        eps=eps,
    )
    
    return out


def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Pattern to match: ReLU -> BatchNorm -> Dropout(p=0.0)
    
    Matches the complete normalization block used in LINKX model:
    1. ReLU activation
    2. Batch normalization (inference mode)
    3. Dropout with p=0.0 (no-op in inference mode, but must be included
       to avoid "node leakage" in subgraph matching)
    
    The dropout is a no-op for p=0.0 and training=False, so it's
    effectively eliminated in the fused kernel.
    
    Args:
        in_0: running_mean [C]
        in_1: running_var [C]
        in_2: bias [C]
        in_3: weight [C]
        in_4: input tensor [N, C]
    
    Returns:
        The output tensor after all operations (dropout returns input unchanged)
    """
    tmp_relu = torch.nn.functional.relu(in_4, inplace = False)
    tmp_bn = torch.nn.functional.batch_norm(tmp_relu, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_out = torch.nn.functional.dropout(tmp_bn, p = 0.0, training = False)
    return tmp_out


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    """
    Extract arguments for the replacement function.
    
    Reorders from model's (running_mean, running_var, weight, bias, input)
    to kernel's (input, running_mean, running_var, weight, bias, momentum, eps)
    """
    return (in_4, in_0, in_1, in_3, in_2, 0.1, 1e-05)


def replacement_func():
    """
    Returns the replacement function for the matched pattern.
    """
    return fused_relu_batchnorm_wrapper