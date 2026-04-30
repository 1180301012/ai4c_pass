import torch
import triton
import triton.language as tl


@triton.jit
def fused_bn_add_relu_mean_kernel(
    x_ptr,           # in_4: input tensor
    bn_mean_ptr,     # in_0: running mean
    bn_var_ptr,      # in_1: running var
    bn_bias_ptr,     # in_2: bias
    bn_weight_ptr,   # in_3: weight
    add_ptr,         # in_5: tensor to add
    out_ptr,         # output tensor
    mean_out_ptr,    # mean output tensor
    N, C, H, W,      # tensor dimensions
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: BatchNorm -> Add -> ReLU -> Mean
    """
    # Calculate output indices
    pid = tl.program_id(0)
    num_elements = N * C * H * W
    
    if pid * BLOCK_SIZE >= num_elements:
        return
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Compute n, c, h, w indices
    W_vals = tl.load(x_ptr + offsets * 0, mask=mask, other=0.0) if False else None
    n = offsets // (C * H * W)
    c = (offsets // (H * W)) % C
    h = (offsets // W) % H
    w = offsets % W
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load BN parameters
    mean = tl.load(bn_mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(bn_var_ptr + c, mask=mask, other=0.0)
    weight = tl.load(bn_weight_ptr + c, mask=mask, other=0.0)
    bias = tl.load(bn_bias_ptr + c, mask=mask, other=0.0)
    
    # Load add tensor
    add_val = tl.load(add_ptr + offsets, mask=mask, other=0.0)
    
    # BatchNorm calculation: ((x - mean) / sqrt(var + eps)) * weight + bias
    inv_std = tl.rsqrt(var + eps)
    bn_out = (x - mean) * inv_std * weight + bias
    
    # Add + ReLU
    relu_out = tl.maximum(bn_out + add_val, 0.0)
    
    # Store output
    tl.store(out_ptr + offsets, relu_out, mask=mask)


@triton.jit
def mean_reduce_kernel(
    out_ptr,         # input tensor
    mean_out_ptr,    # mean output tensor
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel to compute spatial mean of the output.
    """
    pid = tl.program_id(0)
    num_outputs = N * C
    
    if pid >= num_outputs:
        return
    
    n = pid // C
    c = pid % C
    
    # Compute sum over H*W
    sum_val = 0.0
    for h_idx in range(H):
        for w_idx in range(W):
            offset = n * C * H * W + c * H * W + h_idx * W + w_idx
            val = tl.load(out_ptr + offset)
            sum_val = sum_val + val
    
    mean_val = sum_val / (H * W)
    
    # Store mean output with keepdim=True semantics
    # Output shape: [N, C, 1, 1]
    tl.store(mean_out_ptr + pid * 1, mean_val)


@torch.fx.wrap
def fused_bn_add_relu_mean_wrapper(
    in_0,  # bn_mean
    in_1,  # bn_var
    in_2,  # bn_bias
    in_3,  # bn_weight
    in_4,  # input tensor
    in_5,  # add tensor
):
    """
    Wrapper function that launches the fused kernel.
    """
    N, C, H, W = in_4.shape
    
    # Allocate output tensors
    out = torch.empty_like(in_4)
    mean_out = torch.empty((N, C, 1, 1), dtype=in_4.dtype, device=in_4.device)
    
    # Kernel configuration
    BLOCK_SIZE = 1024
    num_elements = N * C * H * W
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch first kernel for BN + Add + ReLU
    fused_bn_add_relu_mean_kernel[(num_programs,)](
        in_4, in_0, in_1, in_2, in_3, in_5,
        out, mean_out,
        N, C, H, W,
        1e-05,
        BLOCK_SIZE,
    )
    
    # Launch second kernel for mean reduction
    mean_reduce_kernel[(N * C,)](
        out, mean_out, N, C, H, W, 1
    )
    
    return out, mean_out


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Pattern: BatchNorm -> Add -> ReLU -> Mean
    """
    tmp_4 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_5 = in_5 + tmp_4
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=False)
    tmp_7 = tmp_6.mean((2, 3), keepdim=True)
    return tmp_6, tmp_7


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


def replacement_func():
    return fused_bn_add_relu_mean_wrapper