import torch
import triton
import triton.language as tl

# Pattern: conv2d -> sigmoid -> mul

def pattern(input, weight, bias, feature_map):
    # Conv2d with exact same signature as model.py
    conv_out = torch.conv2d(input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    # Sigmoid
    sigmoid_out = conv_out.sigmoid()
    # Multiply: feature_map * sigmoid_out (matching model's in_6 * tmp_7)
    mul_out = feature_map * sigmoid_out
    return mul_out


def replacement_args(input, weight, bias, feature_map):
    return (input, weight, bias, feature_map)


# Kernel 1: Compute conv2d + sigmoid on small tensor
@triton.jit
def conv_sigmoid_kernel(
    se_input_ptr,     # [B, C_in]
    weight_ptr,       # [C_out, C_in]
    bias_ptr,         # [C_out]
    output_ptr,       # [B, C_out]
    B, C_in, C_out,
    BLOCK_B: tl.constexpr,
    BLOCK_COUT: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_cout = tl.program_id(1)
    
    b_idx = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    cout_idx = pid_cout * BLOCK_COUT + tl.arange(0, BLOCK_COUT)
    
    b_mask = b_idx < B
    cout_mask = cout_idx < C_out
    
    acc = tl.zeros((BLOCK_B, BLOCK_COUT), dtype=tl.float32)
    
    for cin in range(C_in):
        inp = tl.load(se_input_ptr + b_idx * C_in + cin, mask=b_mask, other=0.0)
        w = tl.load(weight_ptr + cout_idx * C_in + cin, mask=cout_mask, other=0.0)
        acc += inp[:, None] * w[None, :]
    
    bias = tl.load(bias_ptr + cout_idx, mask=cout_mask, other=0.0)
    acc = acc + bias[None, :]
    result = tl.sigmoid(acc)
    
    out_ptrs = output_ptr + b_idx[:, None] * C_out + cout_idx[None, :]
    tl.store(out_ptrs, result, mask=b_mask[:, None] & cout_mask[None, :])


# Kernel 2: Broadcast multiply
@triton.jit
def broadcast_mul_kernel(
    sigmoid_ptr,      # [B, C_out]
    feature_ptr,      # [B, C_out, H, W]
    output_ptr,       # [B, C_out, H, W]
    C_out, HW,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    c = (offsets // HW) % C_out
    b = offsets // (C_out * HW)
    
    sigmoid_idx = b * C_out + c
    sigmoid_val = tl.load(sigmoid_ptr + sigmoid_idx, mask=mask, other=0.0)
    
    feature = tl.load(feature_ptr + offsets, mask=mask, other=0.0)
    result = feature * sigmoid_val
    
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def triton_conv_sigmoid_mul(input, weight, bias, feature_map):
    B, C_in, _, _ = input.shape
    B_fm, C_out, H, W = feature_map.shape
    
    input_flat = input.view(B, C_in).contiguous()
    weight_flat = weight.view(C_out, C_in).contiguous()
    
    sigmoid_vals = torch.empty((B, C_out), device=input.device, dtype=input.dtype)
    
    # Optimize block sizes for different scenarios
    BLOCK_B = min(32, B) if B > 1 else 1
    # Use larger BLOCK_COUT to reduce grid size  
    BLOCK_COUT = 128 if C_out >= 256 else 64
    
    grid1 = (triton.cdiv(B, BLOCK_B), triton.cdiv(C_out, BLOCK_COUT))
    
    conv_sigmoid_kernel[grid1](
        input_flat, weight_flat, bias, sigmoid_vals,
        B, C_in, C_out, BLOCK_B, BLOCK_COUT,
    )
    
    feature_contig = feature_map.contiguous()
    output = torch.empty_like(feature_contig)
    
    HW = H * W
    total_elements = B_fm * C_out * HW
    
    # Use larger block for larger tensors
    BLOCK_SIZE = 4096 if total_elements > 1000000 else 2048
    grid2 = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    broadcast_mul_kernel[grid2](
        sigmoid_vals, feature_contig, output,
        C_out, HW, total_elements, BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return triton_conv_sigmoid_mul