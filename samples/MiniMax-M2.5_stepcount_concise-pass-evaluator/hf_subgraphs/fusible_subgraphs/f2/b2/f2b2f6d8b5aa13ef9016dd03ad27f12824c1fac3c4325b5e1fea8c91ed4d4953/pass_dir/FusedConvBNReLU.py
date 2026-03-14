import torch
import triton
import triton.language as tl


@triton.jit
def fused_bn_relu_kernel(
    # Input tensor (row-major layout)
    input_ptr,
    # Batch norm parameters
    running_mean_ptr,
    running_var_ptr,
    bn_weight_ptr,
    bn_bias_ptr,
    # Output tensor
    output_ptr,
    # Sizes
    B,  # batch size
    C,  # channels
    H,  # height
    W,  # width
    # Meta parameters
    eps: tl.constexpr,
    negative_slope: tl.constexpr,
):
    """
    Fused BatchNorm + LeakyReLU kernel.
    
    For each element in the tensor:
    BatchNorm: z = (x - mean) / sqrt(var + eps) * weight + bias
    LeakyReLU: o = z if z > 0 else z * negative_slope
    """
    # Get program ID
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    
    # Each program processes BLOCK_SIZE elements
    # Calculate element positions
    num_elements = B * C * H * W
    
    for idx in range(pid, num_elements, num_pid):
        # Decode position
        elem_idx = idx
        
        # Calculate 4D position
        b = elem_idx // (C * H * W)
        rem = elem_idx % (C * H * W)
        c = rem // (H * W)
        rem = rem % (H * W)
        h = rem // W
        w = rem % W
        
        # Load input value
        input_offset = b * C * H * W + c * H * W + h * W + w
        x = tl.load(input_ptr + input_offset).to(tl.float32)
        
        # BatchNorm
        mean = tl.load(running_mean_ptr + c).to(tl.float32)
        var = tl.load(running_var_ptr + c).to(tl.float32)
        bn_w = tl.load(bn_weight_ptr + c).to(tl.float32)
        bn_b = tl.load(bn_bias_ptr + c).to(tl.float32)
        
        inv_std = 1.0 / tl.sqrt(var + eps)
        bn_out = (x - mean) * inv_std * bn_w + bn_b
        
        # LeakyReLU
        output_val = tl.where(bn_out > 0, bn_out, bn_out * negative_slope)
        
        # Store output
        tl.store(output_ptr + input_offset, output_val)


def fused_bn_relu(
    bn_out: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    eps: float = 1e-05,
    negative_slope: float = 0.01,
):
    """
    Fused BatchNorm + LeakyReLU kernel wrapper.
    
    Input shapes:
        bn_out: [B, C, H, W] - output of batch norm
        running_mean: [C]
        running_var: [C]
        bn_weight: [C]
        bn_bias: [C]
    
    Output shape: [B, C, H, W]
    """
    B, C, H, W = bn_out.shape
    
    output = torch.empty_like(bn_out)
    
    # Grid: use one program per element, but cap at 65536 to avoid too many blocks
    # Use tl.dot for grid calculation to avoid symbolic tracing issues
    num_elements = B * C * H * W
    
    # Use a fixed large grid - Triton will handle excess blocks efficiently
    grid_size = 65536
    
    # Launch fused BN+ReLU kernel
    fused_bn_relu_kernel[grid_size](
        input_ptr=bn_out,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        bn_weight_ptr=bn_weight,
        bn_bias_ptr=bn_bias,
        output_ptr=output,
        B=B,
        C=C,
        H=H,
        W=W,
        eps=eps,
        negative_slope=negative_slope,
    )
    
    return output


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Pattern: Conv2D + BatchNorm + LeakyReLU
    
    Matches:
        tmp_6 = torch.conv2d(tmp_5, tmp_4, None, (1, 1), (1, 1), (1, 1), 1)
        tmp_7 = torch.nn.functional.batch_norm(tmp_6, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_8 = torch.nn.functional.leaky_relu(tmp_7, 0.01, True)
        return (tmp_8,)
    
    Input mapping:
        in_0 -> running_mean (tmp_0)
        in_1 -> running_var (tmp_1)
        in_2 -> bias (tmp_2)
        in_3 -> weight (tmp_3)
        in_4 -> conv_weight (tmp_4)
        in_5 -> input (tmp_5)
    """
    tmp_6 = torch.conv2d(in_5, in_4, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_8 = torch.nn.functional.leaky_relu(tmp_7, 0.01, True)
    return tmp_8


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_5, in_4, in_0, in_1, in_3, in_2)


def replacement_func():
    return fused_bn_relu