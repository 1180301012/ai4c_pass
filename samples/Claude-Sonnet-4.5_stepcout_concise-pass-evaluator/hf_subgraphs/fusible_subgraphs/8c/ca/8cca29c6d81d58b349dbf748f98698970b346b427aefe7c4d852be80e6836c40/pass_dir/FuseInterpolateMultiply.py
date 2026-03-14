import torch
import triton
import triton.language as tl


def pattern(input_tensor, scale_tensor, target_h, target_w):
    """
    Pattern: interpolate (nearest mode) followed by multiply
    """
    interpolated = torch.nn.functional.interpolate(input_tensor, size=(target_h, target_w), mode='nearest')
    result = scale_tensor * interpolated
    return result


def replacement_args(input_tensor, scale_tensor, target_h, target_w):
    return (input_tensor, scale_tensor, target_h, target_w)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_W': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_W': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_H': 8, 'BLOCK_SIZE_W': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE_H': 64, 'BLOCK_SIZE_W': 8}, num_warps=2),
    ],
    key=['N', 'C', 'H_out', 'W_out'],
)
@triton.jit
def fused_interpolate_multiply_kernel(
    input_ptr, scale_ptr, output_ptr,
    N, C, H_in, W_in, H_out, W_out,
    scale_h: tl.constexpr, scale_w: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
):
    # Get program IDs
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate output positions
    h_start = pid_h * BLOCK_SIZE_H
    w_start = pid_w * BLOCK_SIZE_W
    
    h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)
    
    # Mask for valid output positions
    h_mask = h_offsets < H_out
    w_mask = w_offsets < W_out
    
    # Calculate corresponding input positions (nearest neighbor)
    # For nearest mode: input_pos = floor(output_pos / scale)
    h_in = (h_offsets * H_in) // H_out
    w_in = (w_offsets * W_in) // W_out
    
    # Clamp to valid input range
    h_in = tl.minimum(tl.maximum(h_in, 0), H_in - 1)
    w_in = tl.minimum(tl.maximum(w_in, 0), W_in - 1)
    
    # Calculate base pointers
    input_base = pid_n * C * H_in * W_in + pid_c * H_in * W_in
    scale_base = pid_n * C * H_out * W_out + pid_c * H_out * W_out
    output_base = pid_n * C * H_out * W_out + pid_c * H_out * W_out
    
    # Load, compute, and store for each position
    for i in range(BLOCK_SIZE_H):
        if h_start + i < H_out:
            h_in_idx = (((h_start + i) * H_in) // H_out)
            h_in_idx = tl.minimum(tl.maximum(h_in_idx, 0), H_in - 1)
            
            for j in range(BLOCK_SIZE_W):
                if w_start + j < W_out:
                    w_in_idx = (((w_start + j) * W_in) // W_out)
                    w_in_idx = tl.minimum(tl.maximum(w_in_idx, 0), W_in - 1)
                    
                    # Load input value
                    input_idx = input_base + h_in_idx * W_in + w_in_idx
                    input_val = tl.load(input_ptr + input_idx)
                    
                    # Load scale value
                    scale_idx = scale_base + (h_start + i) * W_out + (w_start + j)
                    scale_val = tl.load(scale_ptr + scale_idx)
                    
                    # Multiply
                    output_val = input_val * scale_val
                    
                    # Store output value
                    output_idx = output_base + (h_start + i) * W_out + (w_start + j)
                    tl.store(output_ptr + output_idx, output_val)


@torch.fx.wrap
def fused_interpolate_multiply(input_tensor, scale_tensor, target_h, target_w):
    N, C, H_in, W_in = input_tensor.shape
    H_out, W_out = target_h, target_w
    
    # Ensure scale_tensor matches output size
    assert scale_tensor.shape == (N, C, H_out, W_out), \
        f"Scale tensor shape {scale_tensor.shape} doesn't match expected ({N}, {C}, {H_out}, {W_out})"
    
    output = torch.empty((N, C, H_out, W_out), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Calculate scale factors
    scale_h = H_out / H_in
    scale_w = W_out / W_in
    
    # Launch kernel
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    
    grid = (
        N,
        C,
        triton.cdiv(H_out, BLOCK_SIZE_H),
        triton.cdiv(W_out, BLOCK_SIZE_W)
    )
    
    fused_interpolate_multiply_kernel[grid](
        input_tensor, scale_tensor, output,
        N, C, H_in, W_in, H_out, W_out,
        scale_h, scale_w,
    )
    
    return output


def replacement_func():
    return fused_interpolate_multiply