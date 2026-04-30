import torch
import triton
import triton.language as tl

@triton.jit
def fuse_permute_reshape_interpolate_kernel(
    input_ptr, output_ptr,
    batch_size, seq_len, channels,
    output_h, output_w,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_C: tl.constexpr,
    num_stages: tl.constexpr
):
    """
    Fused kernel for: permute + reshape + bilinear_interpolate
    Input shape: [batch, seq_len, channels] = [B, 256, 768]
    After permute: [batch, channels, seq_len] = [B, 768, 256]
    After reshape: [batch, channels, 16, 16] (since 256 = 16*16)
    After interpolate: [batch, channels, 128, 128]
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Output coordinates for this program
    out_y = tl.arange(0, BLOCK_SIZE_B)[:, None]
    out_x = tl.arange(0, BLOCK_SIZE_B)[None, :]
    
    # Source coordinates in the reshaped tensor [B, C, 16, 16]
    # Calculate source position based on output position
    src_h = 16.0 * (out_y + 0.5) / output_h
    src_w = 16.0 * (out_x + 0.5) / output_w
    
    # Compute integer and fractional parts for bilinear interpolation
    src_h0 = tl.cast(tl.floor(src_h), tl.int32)
    src_w0 = tl.cast(tl.floor(src_w), tl.int32)
    src_h1 = tl.min(src_h0 + 1, 15)
    src_w1 = tl.min(src_w0 + 1, 15)
    
    fh = src_h - tl.cast(src_h0, tl.float32)
    fw = src_w - tl.cast(src_w0, tl.float32)
    
    # Linear idx in original tensor: idx = c * 256 + h * 16 + w
    # Load 4 corners for bilinear interpolation
    def load_value(c, h, w):
        idx = c * seq_len + h * 16 + w
        return tl.load(input_ptr + pid_b * channels * seq_len + idx)
    
    # Precompute strides
    stride_h0 = 16
    stride_w0 = 1
    
    # Load corners
    v00 = load_value(pid_c, src_h0, src_w0)
    v01 = load_value(pid_c, src_h0, src_w1)
    v10 = load_value(pid_c, src_h1, src_w0)
    v11 = load_value(pid_c, src_h1, src_w1)
    
    # Bilinear interpolation
    v0 = v00 * (1.0 - fw) + v01 * fw
    v1 = v10 * (1.0 - fw) + v11 * fw
    result = v0 * (1.0 - fh) + v1 * fh
    
    # Store output
    out_idx = (pid_c * output_h + out_y) * output_w + out_x
    out_ptrs = output_ptr + pid_b * channels * output_h * output_w + out_idx
    tl.store(out_ptrs, result)


@torch.fx.wrap
def fused_permute_reshape_interpolate(x):
    """
    Fused operation: permute(0,2,1) -> reshape(*,16,16) -> interpolate(size=(128,128), mode='bilinear')
    Input x: [batch, seq_len=256, channels=768]
    Output: [batch, channels=768, 128, 128]
    """
    B, S, C = x.shape
    H_out, W_out = 128, 128
    
    # Allocate output
    output = torch.empty((B, C, H_out, W_out), device=x.device, dtype=x.dtype)
    
    # Grid configuration
    BLOCK_SIZE = 32
    
    grid = (B, C)
    
    # Launch fused kernel
    fuse_permute_reshape_interpolate_kernel[grid](
        x, output,
        B, S, C,
        H_out, W_out,
        BLOCK_SIZE_B=BLOCK_SIZE, BLOCK_SIZE_C=1,
        num_stages=1
    )
    
    return output


def pattern(linear_out):
    """
    Match: permute -> reshape -> interpolate
    linear_out: [batch, seq_len, channels]
    """
    # Step 1: permute [B, S, C] -> [B, C, S]
    permuted = linear_out.permute(0, 2, 1)
    
    # Step 2: reshape to spatial format [B, C, S] -> [B, C, 16, 16] (since S=256=16*16)
    reshaped = permuted.reshape(-1, 16, 16)
    
    # Step 3: interpolate bilinear from (16,16) to (128,128)
    interpolated = torch.nn.functional.interpolate(
        reshaped, size=(128, 128), mode='bilinear', align_corners=False
    )
    
    return interpolated


def replacement_args(linear_out):
    return (linear_out,)


def replacement_func():
    return fused_permute_reshape_interpolate