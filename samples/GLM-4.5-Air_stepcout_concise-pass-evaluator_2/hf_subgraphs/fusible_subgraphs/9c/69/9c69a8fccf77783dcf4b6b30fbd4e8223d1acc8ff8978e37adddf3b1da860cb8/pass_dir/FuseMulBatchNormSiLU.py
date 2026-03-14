import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """Match the computation pattern: Element-wise Multiply + BatchNorm + SiLU"""
    tmp_4 = in_5 * in_4
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return tmp_6

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)

# Triton kernel for fused Mul + BatchNorm + SiLU
@triton.jit
def fused_mul_batchnorm_silu_kernel(
    x_ptr,                    # Input tensor x_5 [N, C, H, W]
    sigmoid_ptr,              # Sigmoid tensor [N, C, 1, 1] or broadcastable
    running_mean_ptr,         # Running mean [C] (on CPU, needs transfer)
    running_var_ptr,          # Running var [C] (on CPU, needs transfer) 
    weight_ptr,               # Weight [C] (on CPU, needs transfer)
    bias_ptr,                 # Bias [C] (on CPU, needs transfer)
    out_ptr,                  # Output tensor [N, C, H, W]
    N, C, H, W,               # Tensor dimensions
    momentum: tl.constexpr,  # Batch norm momentum (0.1)
    eps: tl.constexpr,        # Batch norm eps (1e-05)
    BLOCK_SIZE: tl.constexpr,
):
    # Program identifiers
    pid = tl.program_id(0)
    num_programs = tl.cdiv(N * C * H * W, BLOCK_SIZE)
    
    # Calculate element offsets
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * C * H * W
    
    # Convert to 4D indices
    offset_4d = offsets
    w_idx = offset_4d % W
    h_idx = (offset_4d // W) % H
    c_idx = (offset_4d // (W * H)) % C
    n_idx = offset_4d // (W * H * C)
    
    # Load input tensor x_5
    x_val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load sigmoid tensor (with broadcasting)
    sigmoid_offset = (n_idx * C + c_idx) * (1 * 1) + (h_idx * 1 + w_idx)
    sigmoid_val = tl.load(sigmoid_ptr + sigmoid_offset, mask=mask, other=0.5)
    
    # Element-wise multiplication: x_5 * sigmoid
    mul_val = x_val * sigmoid_val
    
    # Load batch norm parameters (will be cached by Triton)
    mean_val = tl.load(running_mean_ptr + c_idx)
    var_val = tl.load(running_var_ptr + c_idx)
    weight_val = tl.load(weight_ptr + c_idx) 
    bias_val = tl.load(bias_ptr + c_idx)
    
    # Batch norm computation: (x - mean) / sqrt(var + eps) * weight + bias
    sqrt_var = tl.sqrt(var_val + eps)
    normalized = (mul_val - mean_val) / sqrt_var
    batchnorm_val = normalized * weight_val + bias_val
    
    # SiLU activation: x * sigmoid(x)
    silu_val = batchnorm_val * tl.sigmoid(batchnorm_val)
    
    # Store result
    tl.store(out_ptr + offsets, silu_val, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_mul_batchnorm_silu(in_0, in_1, in_2, in_3, in_4, in_5):
    # Get input shapes
    x_shape = in_5.shape  # [N, C, H, W]
    N, C, H, W = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
    
    # Create output tensor
    out = torch.empty_like(in_5)
    
    # Kernel parameters
    total_elements = N * C * H * W
    BLOCK_SIZE = 1024  # Can be autotuned
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Move batch norm parameters to CUDA for efficient access
    running_mean_cuda = in_0.to(in_5.device)
    running_var_cuda = in_1.to(in_5.device) 
    weight_cuda = in_3.to(in_5.device)
    bias_cuda = in_2.to(in_5.device)
    
    # Launch Triton kernel
    fused_mul_batchnorm_silu_kernel[(num_programs,)](
        x_ptr=in_5,
        sigmoid_ptr=in_4,
        running_mean_ptr=running_mean_cuda,
        running_var_ptr=running_var_cuda,
        weight_ptr=weight_cuda,
        bias_ptr=bias_cuda,
        out_ptr=out,
        N=N, C=C, H=H, W=W,
        momentum=0.1,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Kernel wrapper with autotuning
@torch.fx.wrap
def fused_mul_batchnorm_silu_autotune(in_0, in_1, in_2, in_3, in_4, in_5):
    # Get input shapes
    x_shape = in_5.shape  # [N, C, H, W]
    N, C, H, W = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
    
    # Create output tensor
    out = torch.empty_like(in_5)
    
    # Autotune kernel configurations
    @triton.heuristics({
        "BLOCK_SIZE": lambda args: 512 if args['N'] * args['C'] * args['H'] * args['W'] < 1000000 else 1024,
    })
    @triton.jit
    def autotuned_kernel(
        x_ptr,
        sigmoid_ptr,
        running_mean_ptr,
        running_var_ptr, 
        weight_ptr,
        bias_ptr,
        out_ptr,
        N, C, H, W,
        momentum: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_programs = tl.cdiv(N * C * H * W, BLOCK_SIZE)
        
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N * C * H * W
        
        offset_4d = offsets
        w_idx = offset_4d % W
        h_idx = (offset_4d // W) % H
        c_idx = (offset_4d // (W * H)) % C
        n_idx = offset_4d // (W * H * C)
        
        x_val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        sigmoid_offset = (n_idx * C + c_idx) * (1 * 1) + (h_idx * 1 + w_idx)
        sigmoid_val = tl.load(sigmoid_ptr + sigmoid_offset, mask=mask, other=0.5)
        
        mul_val = x_val * sigmoid_val
        
        mean_val = tl.load(running_mean_ptr + c_idx)
        var_val = tl.load(running_var_ptr + c_idx)
        weight_val = tl.load(weight_ptr + c_idx)
        bias_val = tl.load(bias_ptr + c_idx)
        
        sqrt_var = tl.sqrt(var_val + eps)
        normalized = (mul_val - mean_val) / sqrt_var
        batchnorm_val = normalized * weight_val + bias_val
        
        silu_val = batchnorm_val * tl.sigmoid(batchnorm_val)
        
        tl.store(out_ptr + offsets, silu_val, mask=mask)
    
    # Move batch norm parameters to CUDA
    running_mean_cuda = in_0.to(in_5.device)
    running_var_cuda = in_1.to(in_5.device)
    weight_cuda = in_3.to(in_5.device)
    bias_cuda = in_2.to(in_5.device)
    
    # Launch autotuned kernel
    total_elements = N * C * H * W
    num_programs = (total_elements + 1024 - 1) // 1024  # Initial estimate
    
    autotuned_kernel[(num_programs,)](
        x_ptr=in_5,
        sigmoid_ptr=in_4,
        running_mean_ptr=running_mean_cuda,
        running_var_ptr=running_var_cuda,
        weight_ptr=weight_cuda,
        bias_ptr=bias_cuda,
        out_ptr=out,
        N=N, C=C, H=H, W=W,
        momentum=0.1,
        eps=1e-05,
        BLOCK_SIZE=1024,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_mul_batchnorm_silu_autotune