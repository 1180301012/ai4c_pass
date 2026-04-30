import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Match the pattern: reshape -> avg_pool2d -> batch_norm -> silu
    This fuses reshape, avg_pool2d, batch_norm, and silu into a single kernel.
    
    Args:
        in_0: running_mean [512]
        in_1: running_var [512]
        in_2: bias [512]
        in_3: weight [512]
        in_4: input tensor [4, 128, 256]
    """
    tmp_4 = in_4.reshape(1, 512, 16, 16)
    tmp_5 = torch.nn.functional.avg_pool2d(tmp_4, 2, 2, 0, False, True, None)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.silu(tmp_6, inplace=True)
    return tmp_7

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    """Extract arguments needed for the replacement kernel."""
    return (in_0, in_1, in_2, in_3, in_4)


@triton.jit
def fused_kernel(
    in_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    C, out_H, out_W,
    eps: tl.constexpr,
):
    """
    Fused kernel: reshape + avg_pool2d + batch_norm + silu
    
    Input shape: [4, 128, 256] -> reshape to [1, 512, 16, 16]
    Output shape: [1, 512, 8, 8]
    
    Parallelization: Each program handles one (channel, out_y, out_x) position
    Total programs: C * out_H * out_W = 512 * 8 * 8 = 32768
    
    Input tensor layout [4, 128, 256]:
    - dim 0: 4 (first part of channels, ch_idx // 128)
    - dim 1: 128 (second part of channels, ch_idx % 128)  
    - dim 2: 256 (spatial H*W = 16*16)
    
    For avg_pool2d kernel=2, stride=2:
    - Output position (ch, out_y, out_x) averages input positions:
      (ch, 2*out_y, 2*out_x), (ch, 2*out_y+1, 2*out_x), (ch, 2*out_y, 2*out_x+1), (ch, 2*out_y+1, 2*out_x+1)
    """
    # Program ID: maps to (channel, out_y, out_x)
    pid = tl.program_id(0)
    
    # Compute channel, out_y, out_x from flat pid
    # Layout: pid = ch * out_H * out_W + out_y * out_W + out_x
    out_xy = pid % (out_H * out_W)
    ch = pid // (out_H * out_W)
    out_y = out_xy // out_W
    out_x = out_xy % out_W
    
    # Compute input spatial positions for 2x2 pooling
    in_y_base = out_y * 2
    in_x_base = out_x * 2
    
    # Channel to input index mapping: [4, 128, 256] 
    # channel ch maps to: (ch // 128, ch % 128)
    in_b = ch // 128
    in_c = ch % 128
    
    # Input tensor is [4, 128, 256], so strides are [128*256, 256, 1]
    # Linear offset in input: in_b * 128 * 256 + in_c * 256 + hw
    offset_base = in_b * 128 * 256 + in_c * 256
    
    # Compute hw indices for the 4 pooling positions
    # Spatial is 16x16, so hw = y * 16 + x
    hw00 = in_y_base * 16 + in_x_base
    hw01 = in_y_base * 16 + in_x_base + 1
    hw10 = (in_y_base + 1) * 16 + in_x_base
    hw11 = (in_y_base + 1) * 16 + in_x_base + 1
    
    # Load the 4 values for average pooling
    val00 = tl.load(in_ptr + offset_base + hw00)
    val01 = tl.load(in_ptr + offset_base + hw01)
    val10 = tl.load(in_ptr + offset_base + hw10)
    val11 = tl.load(in_ptr + offset_base + hw11)
    
    # Compute average (2x2 pooling with stride 2)
    pooled = (val00 + val01 + val10 + val11) * 0.25
    
    # Load batch norm parameters
    mean = tl.load(mean_ptr + ch)
    var = tl.load(var_ptr + ch)
    weight = tl.load(weight_ptr + ch)
    bias = tl.load(bias_ptr + ch)
    
    # Batch norm: (x - mean) / sqrt(var + eps) * weight + bias
    normed = (pooled - mean) * tl.rsqrt(var + eps) * weight + bias
    
    # SiLU activation: x * sigmoid(x)
    sig = 1.0 / (1.0 + tl.exp(-normed))
    out = normed * sig
    
    # Store result in output tensor [1, 512, 8, 8]
    # Layout: out_idx = ch * 64 + out_y * 8 + out_x (64 = 8*8)
    out_idx = ch * 64 + out_y * 8 + out_x
    tl.store(out_ptr + out_idx, out)


@torch.fx.wrap
def fused_reshape_avgpool_batchnorm_silu(mean, var, weight, bias, input_tensor):
    """
    Fused kernel: reshape + avg_pool2d + batch_norm + silu
    
    Input shape: [4, 128, 256]
    Output shape: [1, 512, 8, 8]
    
    Steps:
    1. Reshape to [1, 512, 16, 16]
    2. Avg pool with kernel=2, stride=2 -> [1, 512, 8, 8]
    3. Batch norm
    4. SiLU activation
    """
    C = 512
    out_H = 8
    out_W = 8
    
    # Grid: one program per output position (C * out_H * out_W)
    grid = (C * out_H * out_W,)
    
    # Output tensor [1, 512, 8, 8] stored as contiguous [C * out_H * out_W]
    output = torch.empty((C * out_H * out_W,), dtype=input_tensor.dtype, device=input_tensor.device)
    
    fused_kernel[grid](
        in_ptr=input_tensor,
        mean_ptr=mean,
        var_ptr=var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=output,
        C=C,
        out_H=out_H,
        out_W=out_W,
        eps=1e-05,
    )
    
    # Reshape output to [1, 512, 8, 8]
    return output.reshape(1, C, out_H, out_W)


def replacement_func():
    """Return the fused kernel wrapper function."""
    return fused_reshape_avgpool_batchnorm_silu