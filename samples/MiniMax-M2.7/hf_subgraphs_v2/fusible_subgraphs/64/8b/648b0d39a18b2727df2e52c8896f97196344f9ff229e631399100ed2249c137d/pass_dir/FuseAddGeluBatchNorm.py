import torch
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(x):
    """Compute GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""
    cdf = 0.5 * (1.0 + tl.libdevice.tanh(0.7978845608028654 * (x + 0.044715 * tl.libdevice.pow(x, 3))))
    return x * cdf

@triton.jit
def fused_add_gelu_bn_kernel(
    in_4_ptr,
    in_5_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_gelu_ptr,
    out_bn_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of C*H*W elements across batch dimension
    pid = tl.program_id(0)
    num_blocks = N * C * H * W
    offset = pid * BLOCK_SIZE
    
    # Calculate actual element indices
    n_elements = N * C * H * W
    
    # Create index offsets
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate channel index for batch norm params
    # Layout: [N, C, H, W]
    c_indices = (offsets // (H * W)) % C
    
    # Load inputs
    in_4 = tl.load(in_4_ptr + offsets, mask=mask, other=0.0)
    in_5 = tl.load(in_5_ptr + offsets, mask=mask, other=0.0)
    
    # Compute add
    added = in_4 + in_5
    
    # Compute GELU
    gelu_out = gelu_kernel(added)
    
    # Store GELU output
    tl.store(out_gelu_ptr + offsets, gelu_out, mask=mask)
    
    # Load batch norm parameters (broadcast across spatial dims)
    mean = tl.load(running_mean_ptr + c_indices)
    var = tl.load(running_var_ptr + c_indices)
    weight = tl.load(weight_ptr + c_indices)
    bias = tl.load(bias_ptr + c_indices)
    
    # Compute batch norm: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = tl.libdevice.rsqrt(var + eps)
    bn_out = (gelu_out - mean) * inv_std * weight + bias
    
    # Store batch norm output
    tl.store(out_bn_ptr + offsets, bn_out, mask=mask)

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match the pattern:
    in_4 += in_5; in_6 = in_4
    tmp_5 = gelu(in_6)
    tmp_6 = batch_norm(tmp_5, running_mean, running_var, weight, bias, training=False, ...)
    tmp_7 = 0 + tmp_6
    return (tmp_5, tmp_7)
    """
    # in-place addition
    in_4_copy = in_4 + in_5
    # gelu activation
    tmp_5 = torch.nn.functional.gelu(in_4_copy, approximate='none')
    # batch norm (training=False, momentum=0.1, eps=1e-05)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    # identity add
    tmp_7 = 0 + tmp_6
    return (tmp_5, tmp_7)

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    # Extract and return arguments needed for the replacement
    # in_0: running_mean, in_1: running_var, in_2: bias, in_3: weight, in_4, in_5: tensors to add
    return (in_0, in_1, in_2, in_3, in_4, in_5)

@torch.fx.wrap
def fused_add_gelu_bn_wrapper(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Fused kernel that performs:
    1. in_4 += in_5 (element-wise addition)
    2. gelu activation
    3. batch normalization
    
    Returns (gelu_out, bn_out) - both intermediate outputs
    """
    N, C, H, W = in_4.shape
    
    # Create output tensors
    out_gelu = torch.empty_like(in_4)
    out_bn = torch.empty_like(in_4)
    
    # Block size for Triton kernel
    BLOCK_SIZE = 1024
    n_elements = N * C * H * W
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_add_gelu_bn_kernel[(num_programs,)](
        in_4_ptr=in_4,
        in_5_ptr=in_5,
        running_mean_ptr=in_0,
        running_var_ptr=in_1,
        weight_ptr=in_3,
        bias_ptr=in_2,
        out_gelu_ptr=out_gelu,
        out_bn_ptr=out_bn,
        N=N,
        C=C,
        H=H,
        W=W,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (out_gelu, out_bn)

def replacement_func():
    return fused_add_gelu_bn_wrapper