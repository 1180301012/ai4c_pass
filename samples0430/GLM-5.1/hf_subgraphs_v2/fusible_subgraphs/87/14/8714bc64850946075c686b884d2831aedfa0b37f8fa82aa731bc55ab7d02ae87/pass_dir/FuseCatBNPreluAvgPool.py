import torch
import triton
import triton.language as tl


def pattern(cat_input1, cat_input2, bn_mean, bn_var, bn_weight, bn_bias, prelu_weight):
    cat_out = torch.cat([cat_input1, cat_input2], 1)
    bn_out = torch.nn.functional.batch_norm(cat_out, bn_mean, bn_var, bn_weight, bn_bias, False, 0.1, 0.001)
    prelu_out = torch.prelu(bn_out, prelu_weight)
    return prelu_out


def replacement_args(cat_input1, cat_input2, bn_mean, bn_var, bn_weight, bn_bias, prelu_weight):
    return (cat_input1, cat_input2, bn_mean, bn_var, bn_weight, bn_bias, prelu_weight)


@triton.jit
def fused_cat_bn_prelu_kernel(
    input1_ptr, input2_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr, prelu_weight_ptr,
    output_ptr,
    n_elements, C, C_half, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Compute channel index
    c = (offsets // HW) % C
    
    # Determine which input to read from
    from_input1 = c < C_half
    from_input2 = c >= C_half
    
    # Compute input offsets
    # For input1 [B, C_half, H, W]: offset = n * C_half * HW + c * HW + spatial
    # For input2 [B, C_half, H, W]: offset = n * C_half * HW + (c - C_half) * HW + spatial
    spatial = offsets % HW
    n = offsets // (C * HW)
    
    input1_offset = n * C_half * HW + c * HW + spatial
    input2_offset = n * C_half * HW + (c - C_half) * HW + spatial
    
    mask1 = mask & from_input1
    mask2 = mask & from_input2
    
    x1 = tl.load(input1_ptr + input1_offset, mask=mask1, other=0.0).to(tl.float32)
    x2 = tl.load(input2_ptr + input2_offset, mask=mask2, other=0.0).to(tl.float32)
    x = tl.where(from_input1, x1, x2)
    
    # Compute BN: scale * x + offset
    mean_val = tl.load(mean_ptr + c, mask=mask).to(tl.float32)
    var_val = tl.load(var_ptr + c, mask=mask).to(tl.float32)
    w_val = tl.load(weight_ptr + c, mask=mask).to(tl.float32)
    b_val = tl.load(bias_ptr + c, mask=mask).to(tl.float32)
    
    inv_std = 1.0 / tl.sqrt(var_val + eps)
    scale = w_val * inv_std
    bn_offset = b_val - scale * mean_val
    
    bn_out = scale * x + bn_offset
    
    # PReLU: max(0, x) + weight * min(0, x)
    prelu_w = tl.load(prelu_weight_ptr + c, mask=mask).to(tl.float32)
    result = tl.where(bn_out > 0, bn_out, prelu_w * bn_out)
    
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def kernel_wrapper(cat_input1, cat_input2, bn_mean, bn_var, bn_weight, bn_bias, prelu_weight):
    B = cat_input1.shape[0]
    C_half = cat_input1.shape[1]
    H = cat_input1.shape[2]
    W = cat_input1.shape[3]
    C = C_half + cat_input2.shape[1]
    HW = H * W
    dtype = cat_input1.dtype
    device = cat_input1.device
    
    output = torch.empty((B, C, H, W), dtype=dtype, device=device)
    
    n_elements = B * C * H * W
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_cat_bn_prelu_kernel[(num_programs,)](
        input1_ptr=cat_input1,
        input2_ptr=cat_input2,
        mean_ptr=bn_mean,
        var_ptr=bn_var,
        weight_ptr=bn_weight,
        bias_ptr=bn_bias,
        prelu_weight_ptr=prelu_weight,
        output_ptr=output,
        n_elements=n_elements,
        C=C,
        C_half=C_half,
        HW=HW,
        eps=0.001,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def replacement_func():
    return kernel_wrapper