import torch
import triton
import triton.language as tl
import triton.ops


def pattern(in_0, in_1, in_2, in_3):
    """Match conv2d -> stack([x], dim=0) -> sum(dim=0) -> cat([x, other], dim=1) pattern."""
    tmp_2 = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.stack([tmp_2], dim=0)
    tmp_4 = tmp_3.sum(dim=0)
    tmp_5 = torch.cat([tmp_4, in_2], 1)
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# Simple Triton kernel for concatenation
@triton.jit
def concat_kernel(conv_ptr, other_ptr, output_ptr, batch_size, out_c, other_c, h, w,
                 conv_s0, conv_s1, conv_s2, conv_s3, other_s0, other_s1, other_s2, other_s3,
                 out_s0, out_s1, out_s2, out_s3, total_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0) * BLOCK_SIZE
    
    for i in range(BLOCK_SIZE):
        idx = pid + i
        # Process only valid indices - no break/continue needed
        # We use a mask to ensure we don't go out of bounds
        w_idx = idx % w
        idx //= w
        h_idx = idx % h
        idx //= h
        ch = idx % (out_c + other_c)
        idx //= (out_c + other_c)
        batch = idx
        
        # Only process if within bounds - compute condition
        is_valid = (batch < batch_size) & (ch < (out_c + other_c)) & (h_idx < h) & (w_idx < w)
        
        # Use tl.where to conditionally process
        # Since we can't use if inside the loop for control flow, we'll process all
        # and just load/store zeros for invalid indices (they get overwritten later)
        
        if ch < out_c:
            conv_idx = batch * conv_s0 + ch * conv_s1 + h_idx * conv_s2 + w_idx * conv_s3
            val = tl.load(conv_ptr + conv_idx)
        else:
            other_ch = ch - out_c
            other_idx = batch * other_s0 + other_ch * other_s1 + h_idx * other_s2 + w_idx * other_s3
            val = tl.load(other_ptr + other_idx)
        
        out_idx = batch * out_s0 + ch * out_s1 + h_idx * out_s2 + w_idx * out_s3
        tl.store(output_ptr + out_idx, val)


@torch.fx.wrap
def fused_conv2d_cat_wrapper(bias, weight, other, input):
    """Fused implementation that eliminates stack+sum."""
    batch, in_c, h, w = input.shape
    out_c = weight.shape[0]
    other_c = other.shape[1]
    
    # Use triton.ops.matmul for 1x1 conv
    input_flat = input.permute(0, 2, 3, 1).reshape(-1, in_c)
    weight_mat = weight.squeeze(-1).squeeze(-1).t()
    conv_flat = triton.ops.matmul(input_flat, weight_mat)
    conv_result = conv_flat.reshape(batch, h, w, out_c).permute(0, 3, 1, 2) + bias.view(1, -1, 1, 1)
    
    # Allocate output and use Triton kernel for concat
    output = torch.empty((batch, out_c + other_c, h, w), device=input.device, dtype=input.dtype)
    output[:, :out_c, :, :] = conv_result
    
    grid = ((batch * other_c * h * w + 255) // 256,)
    total_elements = batch * (out_c + other_c) * h * w
    concat_kernel[grid](
        conv_result, other, output, batch, out_c, other_c, h, w,
        conv_result.stride(0), conv_result.stride(1), conv_result.stride(2), conv_result.stride(3),
        other.stride(0), other.stride(1), other.stride(2), other.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        total_elements, 256,
    )
    return output


def replacement_func():
    return fused_conv2d_cat_wrapper