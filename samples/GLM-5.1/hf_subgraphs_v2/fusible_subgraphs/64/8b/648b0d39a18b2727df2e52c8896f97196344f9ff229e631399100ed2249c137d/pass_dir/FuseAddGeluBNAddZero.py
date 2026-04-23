import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    in_4 += in_5
    in_6 = in_4
    tmp_5 = torch.nn.functional.gelu(in_6, approximate='none')
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = 0 + tmp_6
    return (tmp_5, tmp_7)

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, "route_fused_add_gelu_bn_addzero")

@triton.jit
def fused_add_gelu_bn_addzero_kernel_flat(
    # Input pointers
    in4_ptr, in5_ptr,
    # BN parameter pointers (all shape [C])
    bn_mean_ptr, bn_var_ptr, bn_bias_ptr, bn_weight_ptr,
    # Output pointers
    gelu_out_ptr, bn_out_ptr,
    # Dimensions
    total_elements,
    num_channels,
    hw_size,
    # BN eps
    eps: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Compute channel index from element offset
    # Layout is (N, C, H, W) - contiguous, so offset = n*C*HW + c*HW + h*W + w
    # c_idx = (offset / HW) % C
    c_idx = (offsets // hw_size) % num_channels
    
    # Load BN parameters for this channel
    bn_mean = tl.load(bn_mean_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    bn_var = tl.load(bn_var_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    bn_weight = tl.load(bn_weight_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    bn_bias = tl.load(bn_bias_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    
    # Pre-compute BN scale and offset in float32
    inv_std = 1.0 / tl.sqrt(bn_var + eps)
    bn_scale = bn_weight * inv_std
    bn_offset = bn_bias - bn_mean * bn_scale
    
    # Load inputs and promote to float32 for computation
    val_in4 = tl.load(in4_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    val_in5 = tl.load(in5_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Step 1: Add
    added = val_in4 + val_in5
    
    # Step 2: GELU (exact: x * 0.5 * (1 + erf(x / sqrt(2))))
    sqrt2 = 1.4142135623730951
    gelu_val = added * 0.5 * (1.0 + tl.erf(added / sqrt2))
    
    # Step 3: BN (scale * input + offset per channel)
    bn_val = bn_scale * gelu_val + bn_offset
    
    # Store results - cast back to original dtype
    # Note: we need to determine the output dtype from the input
    # For now, we store as the same dtype as inputs
    tl.store(gelu_out_ptr + offsets, gelu_val, mask=mask)
    tl.store(bn_out_ptr + offsets, bn_val, mask=mask)


@torch.fx.wrap
def fused_add_gelu_bn_addzero_dispatch(in_0, in_1, in_2, in_3, in_4, in_5, route):
    if route == "route_fused_add_gelu_bn_addzero":
        return _fused_add_gelu_bn_addzero(in_0, in_1, in_2, in_3, in_4, in_5)
    else:
        raise ValueError(f"Unknown route: {route}")

def _fused_add_gelu_bn_addzero(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Computes fused: add + gelu + batch_norm + add_zero
    Returns (gelu_out, bn_out)
    """
    eps = 1e-5
    
    # Get tensor shapes
    N, C, H, W = in_4.shape
    total_elements = in_4.numel()
    hw_size = H * W
    
    # Allocate output tensors
    gelu_out = torch.empty_like(in_4)
    bn_out = torch.empty_like(in_4)
    
    # Use the flat kernel
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_add_gelu_bn_addzero_kernel_flat[(num_programs,)](
        in4_ptr=in_4,
        in5_ptr=in_5,
        bn_mean_ptr=in_0,
        bn_var_ptr=in_1,
        bn_bias_ptr=in_2,
        bn_weight_ptr=in_3,
        gelu_out_ptr=gelu_out,
        bn_out_ptr=bn_out,
        total_elements=total_elements,
        num_channels=C,
        hw_size=hw_size,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (gelu_out, bn_out)

def replacement_func():
    return fused_add_gelu_bn_addzero_dispatch