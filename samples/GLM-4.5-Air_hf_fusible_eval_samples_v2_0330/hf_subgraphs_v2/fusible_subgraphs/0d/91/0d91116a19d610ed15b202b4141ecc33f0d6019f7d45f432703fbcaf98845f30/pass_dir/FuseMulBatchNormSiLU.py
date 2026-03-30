import torch
import triton
import triton.language as tl

@triton.jit
def fused_mul_bn_silu_kernel(
    # Input tensors
    x_ptr,              # Main input [B, C, H, W]
    gate_ptr,           # Gate [B, C, 1, 1] 
    # Batch norm parameters (from CPU memory)
    running_mean_ptr,   # [C]
    running_var_ptr,    # [C]  
    weight_ptr,         # [C]
    bias_ptr,           # [C]
    # Output tensor
    out_ptr,            # [B, C, H, W]
    # Parameters
    N, C, H, W,         # Tensor dimensions
    eps: tl.constexpr,  # epsilon for batch norm
    momentum: tl.constexpr,  # momentum for batch norm
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    # Each program computes one output element
    pid_m = tl.program_id(0)  # Batch dimension (B)
    pid_n = tl.program_id(1)  # Channel dimension (C) 
    pid_h = tl.program_id(2)  # Height dimension (H)
    pid_w = tl.program_id(3)  # Width dimension (W)
    
    # Calculate pointers
    x_offset = pid_m * C * H * W + pid_n * H * W + pid_h * W + pid_w
    gate_offset = pid_m * C + pid_n  # Gate is [B, C, 1, 1]
    out_offset = x_offset
    
    # Load input and gate values
    x = tl.load(x_ptr + x_offset, other=0.0)
    gate = tl.load(gate_ptr + gate_offset, other=0.0)
    
    # Load batch norm parameters for this channel
    mean = tl.load(running_mean_ptr + pid_n, other=0.0)
    var = tl.load(running_var_ptr + pid_n, other=1.0)
    weight = tl.load(weight_ptr + pid_n, other=1.0)
    bias = tl.load(bias_ptr + pid_n, other=0.0)
    
    # Compute fused operations:
    # 1. Element-wise multiplication: x * gate
    mul_out = x * gate
    
    # 2. Batch normalization: (x - mean) / sqrt(var + eps) * weight + bias
    var_adjusted = var + eps
    std = tl.sqrt(var_adjusted)
    bn_out = (mul_out - mean) / std * weight + bias
    
    # 3. SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
    silu_out = bn_out / (1.0 + tl.exp(-bn_out))
    
    # Store output
    tl.store(out_ptr + out_offset, silu_out)

@torch.fx.wrap
def simple_mul(x, y):
    # Simple multiplication function to test pattern matching
    return x * y

def pattern(x, y):
    # Simple pattern to test if matching works
    return x * y

def replacement_args(x, y):
    # Extract all arguments needed for the fusion
    return (x, y)

def replacement_func():
    # Return the simple function reference
    return simple_mul