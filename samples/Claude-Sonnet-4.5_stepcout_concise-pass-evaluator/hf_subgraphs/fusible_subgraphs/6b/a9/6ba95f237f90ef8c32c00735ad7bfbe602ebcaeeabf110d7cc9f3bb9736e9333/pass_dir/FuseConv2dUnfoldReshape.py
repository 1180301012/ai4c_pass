import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Pattern to match: conv2d -> unfold -> reshape
    Must match exactly as in model.py
    """
    tmp_0 = in_0
    tmp_1 = torch.conv2d(in_1, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.unfold(tmp_1, kernel_size=(2, 2), stride=(2, 2))
    tmp_3 = tmp_2.reshape(1, 128, 4, -1)
    return (tmp_3,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_OC': 64, 'BLOCK_SIZE_HW': 64, 'BLOCK_SIZE_IC': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_OC': 128, 'BLOCK_SIZE_HW': 64, 'BLOCK_SIZE_IC': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_OC': 64, 'BLOCK_SIZE_HW': 128, 'BLOCK_SIZE_IC': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_OC': 128, 'BLOCK_SIZE_HW': 128, 'BLOCK_SIZE_IC': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_OC': 64, 'BLOCK_SIZE_HW': 64, 'BLOCK_SIZE_IC': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_OC': 32, 'BLOCK_SIZE_HW': 64, 'BLOCK_SIZE_IC': 64}, num_stages=5, num_warps=2),
    ],
    key=['OUT_C', 'IN_C', 'H', 'W'],
)
@triton.jit
def fused_conv_unfold_kernel(
    input_ptr, weight_ptr, output_ptr,
    B, IN_C, H, W, OUT_C,
    stride_ib, stride_ic, stride_ih, stride_iw,
    stride_woc, stride_wic,
    stride_ob, stride_oc, stride_op, stride_ow,
    BLOCK_SIZE_OC: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
    BLOCK_SIZE_IC: tl.constexpr,
):
    """
    Fused kernel for 1x1 conv2d + unfold (2x2, stride 2) + reshape
    Input: (B, IN_C, H, W)
    Weight: (OUT_C, IN_C, 1, 1)
    After conv: (B, OUT_C, H, W)
    After unfold: (B, OUT_C*4, H//2*W//2)
    After reshape: (B, OUT_C, 4, H//2*W//2)
    """
    # Grid: (num_oc_blocks, num_patch_blocks * 4, B)
    pid_oc = tl.program_id(0)
    pid_patch_p = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    H_out = H // 2
    W_out = W // 2
    num_patches = H_out * W_out
    
    # Split pid_patch_p into patch_id and p (0-3)
    num_patch_blocks = tl.cdiv(num_patches, BLOCK_SIZE_HW)
    pid_p = pid_patch_p // num_patch_blocks
    pid_patch = pid_patch_p % num_patch_blocks
    
    # Output channel block
    oc_start = pid_oc * BLOCK_SIZE_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_SIZE_OC)
    oc_mask = oc_offsets < OUT_C
    
    # Patch block
    patch_start = pid_patch * BLOCK_SIZE_HW
    patch_offsets = patch_start + tl.arange(0, BLOCK_SIZE_HW)
    patch_mask = patch_offsets < num_patches
    
    # Compute spatial positions from patch indices
    patch_h = patch_offsets // W_out
    patch_w = patch_offsets % W_out
    
    # Unfold offset within 2x2 patch
    p_h = pid_p // 2
    p_w = pid_p % 2
    
    # Actual spatial position in conv output
    h_pos = patch_h * 2 + p_h
    w_pos = patch_w * 2 + p_w
    
    # Accumulator
    acc = tl.zeros((BLOCK_SIZE_OC, BLOCK_SIZE_HW), dtype=tl.float32)
    
    # Iterate over input channels
    for ic_start in range(0, IN_C, BLOCK_SIZE_IC):
        ic_offsets = ic_start + tl.arange(0, BLOCK_SIZE_IC)
        ic_mask = ic_offsets < IN_C
        
        # Load weight: (OUT_C, IN_C, 1, 1)
        w_mask = oc_mask[:, None] & ic_mask[None, :]
        weight_ptrs = weight_ptr + (oc_offsets[:, None] * stride_wic * IN_C + ic_offsets[None, :] * stride_wic)
        w = tl.load(weight_ptrs, mask=w_mask, other=0.0)
        
        # Load input: (B, IN_C, H, W)
        # We need to load at positions (h_pos, w_pos) for all patches
        input_mask = ic_mask[None, :] & patch_mask[:, None]
        input_ptrs = (input_ptr + pid_b * stride_ib + 
                     ic_offsets[None, :] * stride_ic +
                     h_pos[:, None] * stride_ih +
                     w_pos[:, None] * stride_iw)
        x = tl.load(input_ptrs, mask=input_mask, other=0.0)
        
        # Compute: out[oc, hw] = sum over ic: w[oc, ic] * x[ic, hw]
        acc += tl.dot(w, x.T, allow_tf32=True)
    
    # Store result: (B, OUT_C, 4, H//2*W//2)
    output_mask = oc_mask[:, None] & patch_mask[None, :]
    output_ptrs = (output_ptr + pid_b * stride_ob +
                   oc_offsets[:, None] * stride_oc +
                   pid_p * stride_op +
                   patch_offsets[None, :] * stride_ow)
    tl.store(output_ptrs, acc, mask=output_mask)


@torch.fx.wrap
def fused_conv_unfold_reshape(weight, input_tensor):
    """
    Fused implementation of conv2d + unfold + reshape
    """
    B, IN_C, H, W = input_tensor.shape
    OUT_C, _, _, _ = weight.shape
    
    H_out = H // 2
    W_out = W // 2
    num_patches = H_out * W_out
    
    # Output shape: (B, OUT_C, 4, num_patches)
    output = torch.empty((B, OUT_C, 4, num_patches), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Grid dimensions
    BLOCK_SIZE_OC = 64
    BLOCK_SIZE_HW = 64
    BLOCK_SIZE_IC = 32
    
    num_oc_blocks = triton.cdiv(OUT_C, BLOCK_SIZE_OC)
    num_patch_blocks = triton.cdiv(num_patches, BLOCK_SIZE_HW)
    
    grid = (num_oc_blocks, num_patch_blocks * 4, B)
    
    fused_conv_unfold_kernel[grid](
        input_tensor, weight, output,
        B, IN_C, H, W, OUT_C,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE_OC=BLOCK_SIZE_OC,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
        BLOCK_SIZE_IC=BLOCK_SIZE_IC,
    )
    
    return output

def replacement_func():
    return fused_conv_unfold_reshape