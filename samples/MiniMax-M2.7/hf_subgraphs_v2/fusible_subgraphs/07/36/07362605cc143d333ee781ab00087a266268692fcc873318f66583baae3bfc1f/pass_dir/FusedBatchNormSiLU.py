import torch
import triton
import triton.language as tl

# Pattern matching function - matches reshape -> batch_norm -> silu pattern for 512 channels (8x8 spatial)
def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Pattern matches:
    1. Reshape tensor to 4D [1, 512, 8, 8]
    2. Batch normalization with running statistics
    3. SiLU activation (in-place)
    
    This matches graphs with 512 channels and 8x8 spatial size.
    Route: "512_8x8"
    """
    tmp_4 = in_4.reshape(1, 512, 8, 8)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace = True)
    return tmp_6

# Alternative pattern for 256 channels case
def pattern_256(in_0, in_1, in_2, in_3, in_4):
    """
    Pattern matches reshape -> batch_norm -> silu for 256 channels (16x16 spatial)
    Route: "256_16x16"
    """
    tmp_4 = in_4.reshape(1, 256, 16, 16)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace = True)
    return tmp_6

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    """
    Extract arguments: running_mean, running_var, bias, weight, input, and route string
    Route "512_8x8" is used to identify this pattern.
    """
    return (in_0, in_1, in_2, in_3, in_4, "512_8x8")

# Autotune configuration for optimal performance
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE": 512}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=3, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def fused_batchnorm_silu_kernel_512(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused BatchNorm + SiLU kernel for 512 channels (8x8 spatial).
    
    Computes: output = (x - mean) / sqrt(var + eps) * weight + bias
              output = output * sigmoid(output)  [SiLU]
    
    This fuses two operations into one kernel, reducing memory bandwidth
    and eliminating one intermediate tensor allocation.
    
    Uses fixed spatial dimensions: height=8, width=8 for optimized indexing.
    """
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input - convert to float32 for computation
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute channel and spatial indices for 8x8=64 spatial elements
    # Layout: [1, 512, 8, 8] -> flattened index maps to [batch*512*64 + ch*64 + spatial]
    ch_offsets = (offsets // 64) % 512
    
    # Load batch norm parameters (broadcast across spatial dims)
    mean = tl.load(running_mean_ptr + ch_offsets, mask=mask, other=0.0)
    var = tl.load(running_var_ptr + ch_offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + ch_offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + ch_offsets, mask=mask, other=0.0)
    
    # Compute normalized output: (x - mean) / sqrt(var + eps)
    std = tl.sqrt(var + eps)
    norm = (x - mean) / std
    
    # Apply affine transform: norm * weight + bias
    out = norm * weight + bias
    
    # Apply SiLU: out * sigmoid(out)
    sig = 1.0 / (1.0 + tl.exp(-out))
    out = out * sig
    
    # Store output - convert to bfloat16
    tl.store(output_ptr + offsets, out.to(tl.bfloat16), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_stages=3, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def fused_batchnorm_silu_kernel_256(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused BatchNorm + SiLU kernel for 256 channels (16x16 spatial).
    
    Computes: output = (x - mean) / sqrt(var + eps) * weight + bias
              output = output * sigmoid(output)  [SiLU]
    
    Uses fixed spatial dimensions: height=16, width=16 for optimized indexing.
    """
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input - convert to float32 for computation
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute channel and spatial indices for 16x16=256 spatial elements
    ch_offsets = (offsets // 256) % 256
    
    # Load batch norm parameters (broadcast across spatial dims)
    mean = tl.load(running_mean_ptr + ch_offsets, mask=mask, other=0.0)
    var = tl.load(running_var_ptr + ch_offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + ch_offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + ch_offsets, mask=mask, other=0.0)
    
    # Compute normalized output: (x - mean) / sqrt(var + eps)
    std = tl.sqrt(var + eps)
    norm = (x - mean) / std
    
    # Apply affine transform: norm * weight + bias
    out = norm * weight + bias
    
    # Apply SiLU: out * sigmoid(out)
    sig = 1.0 / (1.0 + tl.exp(-out))
    out = out * sig
    
    # Store output - convert to float16
    tl.store(output_ptr + offsets, out.to(tl.float16), mask=mask)


@torch.fx.wrap
def fused_batchnorm_silu_dispatch(running_mean, running_var, bias, weight, x, route):
    """
    Dispatch function for the fused batchnorm + silu kernel.
    Uses routing to select the appropriate kernel based on the matched pattern.
    
    Args:
        running_mean: Running mean [channels]
        running_var: Running variance [channels]
        bias: Bias [channels]
        weight: Weight [channels]
        x: Input tensor [batch, channels, height, width]
        route: String identifying which pattern was matched
    
    Returns:
        Output tensor after batchnorm + silu
    """
    # Get tensor info
    B, C, H, W = x.shape
    n_elements = B * C * H * W
    
    # Allocate output tensor (same dtype as input)
    output = torch.empty_like(x)
    
    # Launch kernel with grid sized for parallelism
    block_size = 1024
    grid = ((n_elements + block_size - 1) // block_size,)
    
    if route == "512_8x8":
        fused_batchnorm_silu_kernel_512[grid](
            x,
            running_mean,
            running_var,
            weight,
            bias,
            output,
            n_elements,
            1e-05,
        )
    elif route == "256_16x16":
        fused_batchnorm_silu_kernel_256[grid](
            x,
            running_mean,
            running_var,
            weight,
            bias,
            output,
            n_elements,
            1e-05,
        )
    
    return output


def replacement_func():
    """
    Returns the replacement function that implements fused batchnorm + silu.
    """
    return fused_batchnorm_silu_dispatch