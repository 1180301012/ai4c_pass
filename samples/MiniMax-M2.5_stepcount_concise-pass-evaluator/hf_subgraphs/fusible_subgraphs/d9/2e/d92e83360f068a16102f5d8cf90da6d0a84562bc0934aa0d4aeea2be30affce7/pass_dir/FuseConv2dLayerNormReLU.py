import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_C': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_C': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_C': 16}, num_warps=2, num_stages=3),
    ],
    key=['C'],
)
@triton.jit
def fused_conv_ln_relu_kernel(
    # Input tensor (N, Cin, 1, 1) stored as (N*Cin) contiguous
    input_ptr,
    # Conv weight (C, Cin, 1, 1)
    conv_weight_ptr,
    # Conv bias (C,)
    conv_bias_ptr,
    # LayerNorm weight (C,)
    ln_weight_ptr,
    # LayerNorm bias (C,)
    ln_bias_ptr,
    # Output tensor (N, C, 1, 1)
    output_ptr,
    # Tensor dimensions
    N: tl.constexpr,
    C: tl.constexpr,
    Cin: tl.constexpr,
    # Epsilon for LayerNorm
    eps: tl.constexpr,
    # Block size for channel dimension
    BLOCK_C: tl.constexpr,
):
    """
    Fused kernel for: Conv2d(1x1) -> LayerNorm -> ReLU
    
    Input shape: (N, Cin, 1, 1) stored as (N, Cin) 2D
    Conv weight shape: (C, Cin, 1, 1)
    Conv bias shape: (C,)
    Output shape: (N, C, 1, 1) stored as (N, C) 2D
    """
    # Get program ID for batch dimension
    pid_n = tl.program_id(0)
    
    # Calculate offsets
    input_offset = pid_n * Cin
    output_offset = pid_n * C
    
    # ---- Step 1: Conv2d (1x1, effectively matrix multiplication) ----
    # For each output channel c: out[c] = bias[c] + sum over cin of input[cin] * weight[c, cin]
    # Since spatial is 1x1, this is just a dot product
    
    # First compute convolution for all output channels
    # Use BLOCK_C to process in chunks
    for c_start in range(0, C, BLOCK_C):
        c_end = min(c_start + BLOCK_C, C)
        
        # Compute for each channel in this block
        for c in range(c_start, c_end):
            sum_val = 0.0
            # Load bias
            sum_val += tl.load(conv_bias_ptr + c)
            # Sum over input channels
            for cin in range(Cin):
                input_val = tl.load(input_ptr + input_offset + cin)
                weight_val = tl.load(conv_weight_ptr + c * Cin + cin)
                sum_val += input_val * weight_val
            
            # ---- Step 2: LayerNorm (per-channel, compute mean/var first) ----
            # We'll compute mean/var in a separate pass and store conv output temporarily
            # Actually, for simplicity, let's compute mean/var in a second pass
            
            # Store conv output to temp storage - but we don't have temp storage
            # So let's do a different approach: compute mean/var first
            
            # For now, just store conv output
            tl.store(output_ptr + output_offset + c, sum_val)
    
    # ---- Step 2: LayerNorm ----
    # Compute mean and variance over channel dimension
    # First pass: compute mean
    mean = 0.0
    for c in range(C):
        conv_val = tl.load(output_ptr + output_offset + c)
        mean += conv_val
    mean = mean / C
    
    # Second pass: compute variance
    var = 0.0
    for c in range(C):
        conv_val = tl.load(output_ptr + output_offset + c)
        diff = conv_val - mean
        var += diff * diff
    var = var / C + eps
    
    # Compute standard deviation
    rstd = 1.0 / tl.sqrt(var)
    
    # Third pass: apply LayerNorm and ReLU
    # LayerNorm: (x - mean) * rstd * gamma + beta
    for c in range(C):
        x = tl.load(output_ptr + output_offset + c)
        x_norm = (x - mean) * rstd
        gamma = tl.load(ln_weight_ptr + c)
        beta = tl.load(ln_bias_ptr + c)
        ln_output = x_norm * gamma + beta
        
        # ---- Step 3: ReLU (inplace) ----
        relu_output = tl.maximum(ln_output, 0.0)
        
        # Store final output
        tl.store(output_ptr + output_offset + c, relu_output)


@torch.fx.wrap
def fused_conv_ln_relu_wrapper(
    input: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Fused Conv2d + LayerNorm + ReLU kernel.
    
    Input: (N, Cin, 1, 1)
    Conv weight: (C, Cin, 1, 1) 
    Conv bias: (C,)
    LayerNorm weight: (C,)  # stored as (C, 1, 1) but we use (C,)
    LayerNorm bias: (C,)    # stored as (C, 1, 1) but we use (C,)
    
    Returns: (N, C, 1, 1)
    """
    N = input.shape[0]
    C = conv_weight.shape[0]
    Cin = conv_weight.shape[1]
    
    # Reshape input from (N, Cin, 1, 1) to (N, Cin) for easier access
    input_2d = input.squeeze(-1).squeeze(-1)  # (N, Cin)
    
    # Reshape output
    output = torch.empty((N, C, 1, 1), dtype=torch.float32, device=input.device)
    output_2d = output.squeeze(-1).squeeze(-1)  # (N, C)
    
    # Launch kernel - one program per batch element
    grid = (N,)
    
    fused_conv_ln_relu_kernel[grid](
        input_ptr=input_2d,
        conv_weight_ptr=conv_weight,
        conv_bias_ptr=conv_bias,
        ln_weight_ptr=ln_weight,
        ln_bias_ptr=ln_bias,
        output_ptr=output_2d,
        N=N,
        C=C,
        Cin=Cin,
        eps=eps,
    )
    
    return output


def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Pattern: Conv2d -> LayerNorm -> ReLU
    
    Args:
        in_0: conv bias (C,)
        in_1: conv weight (C, Cin, 1, 1)
        in_2: layer_norm bias (C, 1, 1)
        in_3: layer_norm weight (C, 1, 1)
        in_4: input tensor (N, Cin, 1, 1)
    """
    # Conv2d: stride=1, padding=0, dilation=1, groups=1
    tmp_4 = torch.conv2d(in_4, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    
    # LayerNorm: normalized_shape = (C, 1, 1), eps = 1e-05
    tmp_5 = torch.nn.functional.layer_norm(tmp_4, (tmp_4.shape[1], 1, 1), in_3, in_2, 1e-05)
    
    # ReLU with inplace=True
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=True)
    
    # Only return tmp_6 as that's the only value in the model's return
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    """
    Extract arguments for replacement function.
    """
    # Layer norm weight and bias need to be squeezed from (C, 1, 1) to (C,)
    ln_weight = in_3.squeeze(-1).squeeze(-1)
    ln_bias = in_2.squeeze(-1).squeeze(-1)
    
    return (in_4, in_1, in_0, ln_weight, ln_bias)


def replacement_func():
    """
    Returns the fused kernel function.
    """
    return fused_conv_ln_relu_wrapper