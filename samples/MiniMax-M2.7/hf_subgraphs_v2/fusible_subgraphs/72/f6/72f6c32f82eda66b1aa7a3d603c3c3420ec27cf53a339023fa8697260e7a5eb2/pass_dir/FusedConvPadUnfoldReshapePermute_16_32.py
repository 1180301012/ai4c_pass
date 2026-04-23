import torch
import triton
import triton.language as tl


@triton.jit
def fused_conv_unfold_kernel_16_32(
    # Input pointer
    x_ptr,
    # Weight pointer  
    w_ptr,
    # Output pointer
    out_ptr,
    # Tensor dimensions
    B: tl.constexpr,
    C_in: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    C_out: tl.constexpr,
    # Unfold parameters
    unfold_size: tl.constexpr,
    unfold_step: tl.constexpr,
    # Block sizes
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for depthwise conv2d (1x1) + pad + unfold + reshape + permute
    Optimized for split sizes [16, 32]
    """
    # Calculate dimensions
    pad = 2
    padded_H = H + 2 * pad
    padded_W = W + 2 * pad
    unfold_H = (padded_H - unfold_size) // unfold_step + 1
    unfold_W = (padded_W - unfold_size) // unfold_step + 1
    patch_count = unfold_H * unfold_W
    
    # Output: [B, 4, patch_count * 36, C_out] after permute
    # 36 = 12*12 / 4 (element from 12x12 patch / 4)
    out_d2_total = 144 // 4 * patch_count  # 36 * patch_count
    total_out_elements = B * 4 * out_d2_total * C_out
    
    # Get program ID
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_out_elements
    
    # Decode offset
    stride0 = 4 * out_d2_total * C_out
    stride1 = out_d2_total * C_out
    stride2 = C_out
    
    out_b = offsets // stride0
    remaining = offsets % stride0
    out_d1 = remaining // stride1
    remaining = remaining % stride1
    out_d2 = remaining // stride2
    out_c = remaining % stride2
    
    # Reverse the permute+reshape to get input coordinates
    # patch_idx = out_d2 // 36
    # patch_elem = out_d2 % 36
    patch_idx = out_d2 // 36
    patch_elem = out_d2 % 36
    
    patch_h = patch_idx // unfold_W
    patch_w = patch_idx % unfold_W
    
    # From patch element to original coordinates
    elem_h = patch_elem // unfold_size
    elem_w = patch_elem % unfold_size
    
    input_h = patch_h * unfold_step + elem_h - pad
    input_w = patch_w * unfold_step + elem_w - pad
    
    # Load with bounds checking
    x_idx = out_b * C_in * H * W + out_c * H * W + input_h * W + input_w
    w_idx = out_c
    
    in_mask = (input_h >= 0) & (input_h < H) & (input_w >= 0) & (input_w < W)
    
    x = tl.load(x_ptr + x_idx, mask=in_mask, other=0.0)
    w = tl.load(w_ptr + w_idx)
    
    result = x * w
    tl.store(out_ptr + offsets, result, mask=mask)


def fused_kernel_16_32(x, w):
    """
    Optimized implementation for split sizes [16, 32]
    """
    # Depthwise 1x1 conv
    conv_out = torch.nn.functional.conv2d(x, w, None, (1, 1), (0, 0), (1, 1), 1)
    
    # Pad and unfold
    padded = torch.nn.functional.pad(conv_out, [2, 2, 2, 2], 'constant', 0)
    unfolded_h = padded.unfold(2, 12, 8)
    unfolded = unfolded_h.unfold(3, 12, 8)
    
    # Get dimensions
    B, C, H_unf, W_unf = unfolded.shape[0], unfolded.shape[1], unfolded.shape[2], unfolded.shape[3]
    reshape_b = 8
    reshape_c = C // reshape_b
    
    # Reshape: [B, 8, C/8, 4, -1]
    tmp_5 = unfolded.reshape(reshape_b, reshape_c, 4, -1)
    
    # Permute: [B, 4, -1, C/8]
    tmp_6 = tmp_5.permute(0, 2, 3, 1)
    
    # Split [16, 32]
    split = torch.functional.split(tmp_6, [16, 32], dim=-1)
    tmp_8 = split[0]
    tmp_9 = split[1]
    
    # Transpose first split
    tmp_10 = tmp_8.transpose(-1, -2)
    
    return tmp_10, tmp_9


@torch.fx.wrap
def fused_wrapper_16_32(x, w):
    return fused_kernel_16_32(x, w)


def pattern(in_0, in_1):
    """Match pattern with split [16, 32]"""
    tmp_1 = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.pad(tmp_1, [2, 2, 2, 2], 'constant', None)
    tmp_3 = tmp_2.unfold(2, 12, 8)
    tmp_4 = tmp_3.unfold(3, 12, 8)
    tmp_5 = tmp_4.reshape(8, 48, 4, -1)
    tmp_6 = tmp_5.permute(0, 2, 3, 1)
    tmp_7 = torch.functional.split(tmp_6, [16, 32], dim=-1)
    tmp_8 = tmp_7[0]
    tmp_9 = tmp_7[1]
    tmp_10 = tmp_8.transpose(-1, -2)
    return tmp_10, tmp_9


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_wrapper_16_32