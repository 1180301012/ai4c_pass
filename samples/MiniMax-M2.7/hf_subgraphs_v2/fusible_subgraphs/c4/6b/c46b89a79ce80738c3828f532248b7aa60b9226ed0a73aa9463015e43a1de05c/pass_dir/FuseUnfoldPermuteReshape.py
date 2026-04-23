import torch
import triton
import triton.language as tl


@triton.jit
def unfold_permute_reshape_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    height: tl.constexpr,
    width: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    patch_size: tl.constexpr,
    batch: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: unfold -> permute -> reshape
    - unfold: extracts patches from input
    - permute: moves patch dimension to front
    - reshape: flattens to (num_patches, batch, channels, kernel_h, kernel_w)
    
    This kernel directly computes the output in the target layout,
    avoiding intermediate tensor creation.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate patch grid dimensions
    out_h = (height - kernel_h) // stride_h + 1
    out_w = (width - kernel_w) // stride_w + 1
    
    # Decode flat index to (patch_idx, c, kh, kw)
    # Total elements per patch: batch * patch_size (where patch_size = channels * kernel_h * kernel_w)
    patch_elements = batch * patch_size
    patch_idx = offsets // patch_elements
    local_offset = offsets % patch_elements
    
    c = local_offset // (kernel_h * kernel_w)
    kh_kw_offset = local_offset % (kernel_h * kernel_w)
    kh = kh_kw_offset // kernel_w
    kw = kh_kw_offset % kernel_w
    
    # Convert patch index to (ph, pw) in the patch grid
    ph = patch_idx // out_w
    pw = patch_idx % out_w
    
    # Calculate input position for this patch element
    # Input format: (batch, channels, height, width)
    in_h = ph * stride_h + kh
    in_w = pw * stride_w + kw
    
    # Batch and channel from input
    in_b = c  # channels dimension maps to batch in output
    
    # Input linear index: (in_b * channels * height * width) + (c * height * width) + (in_h * width) + in_w
    # But since in_b and c are the same here, we have:
    # in_idx = c * height * width + in_h * width + in_w
    in_idx = c * height * width + in_h * width + in_w
    
    # Load from input
    val = tl.load(input_ptr + in_idx, mask=mask, other=0.0)
    
    # Store to output: (patch_idx, c, kh, kw) -> (patch_idx * batch * patch_size + c * kernel_h * kernel_w + kh * kernel_w + kw)
    out_idx = pid * BLOCK_SIZE + (offsets - block_start)
    tl.store(output_ptr + out_idx, val, mask=mask)


@torch.fx.wrap
def unfold_permute_reshape_fused(in_tensor, kernel_size, stride):
    """
    Fused operation: unfold + permute + reshape
    in_tensor: (batch, channels, height, width) - bfloat16 or float16
    kernel_size: tuple (kH, kW)
    stride: tuple (sH, sW)
    Returns: (num_patches, batch, channels, kH, kW) in target layout
    """
    B, C, H, W = in_tensor.shape
    kH, kW = kernel_size
    sH, sW = stride
    
    # Calculate output dimensions
    out_h = (H - kH) // sH + 1
    out_w = (W - kW) // sW + 1
    num_patches = out_h * out_w
    
    # Output shape: (num_patches, batch, channels, kH, kW)
    out_shape = (num_patches, B, C, kH, kW)
    out_elements = num_patches * B * C * kH * kW
    
    # Allocate output
    out = torch.empty(out_shape, dtype=in_tensor.dtype, device=in_tensor.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_programs = (out_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    unfold_permute_reshape_kernel[(num_programs,)](
        input_ptr=in_tensor,
        output_ptr=out,
        n_elements=out_elements,
        height=H,
        width=W,
        kernel_h=kH,
        kernel_w=kW,
        stride_h=sH,
        stride_w=sW,
        patch_size=C * kH * kW,
        batch=B,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_0, in_1, in_2):
    """
    Match the pattern:
    1. unfold(in_1, ...) -> permute -> reshape
    2. unfold(in_2, ...) -> permute -> reshape
    3. cat([unfolded_2, unfolded_1, in_0], dim=0)
    4. to(dtype=float16)
    """
    # First unfold branch
    tmp_0 = torch.nn.functional.unfold(in_1, kernel_size=(384, 384), stride=(192, 192))
    tmp_1 = tmp_0.permute(2, 0, 1)
    tmp_2 = tmp_1.reshape(-1, 3, 384, 384)
    
    # Second unfold branch
    tmp_3 = torch.nn.functional.unfold(in_2, kernel_size=(384, 384), stride=(288, 288))
    tmp_4 = tmp_3.permute(2, 0, 1)
    tmp_5 = tmp_4.reshape(-1, 3, 384, 384)
    
    # Concatenate
    tmp_6 = torch.cat([tmp_5, tmp_2, in_0], dim=0)
    
    # Type conversion
    tmp_7 = tmp_6.to(dtype=torch.float16)
    
    return tmp_7


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    def optimized_forward(in_0, in_1, in_2):
        # Fused unfold for in_1 (768x768, stride 192)
        tmp_2 = unfold_permute_reshape_fused(in_1, kernel_size=(384, 384), stride=(192, 192))
        
        # Fused unfold for in_2 (1536x1536, stride 288)
        tmp_5 = unfold_permute_reshape_fused(in_2, kernel_size=(384, 384), stride=(288, 288))
        
        # Concatenate along batch dimension
        tmp_6 = torch.cat([tmp_5, tmp_2, in_0], dim=0)
        
        # Type conversion
        tmp_7 = tmp_6.to(dtype=torch.float16)
        
        return tmp_7
    
    return optimized_forward