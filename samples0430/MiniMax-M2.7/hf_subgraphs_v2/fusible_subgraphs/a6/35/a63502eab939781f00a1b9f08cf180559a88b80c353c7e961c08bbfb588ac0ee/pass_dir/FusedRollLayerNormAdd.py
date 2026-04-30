import torch
import triton
import triton.language as tl

@triton.jit
def fused_roll_ln_add_kernel(
    in_3_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    # Shape info
    B: tl.constexpr,
    H1: tl.constexpr,
    W1: tl.constexpr,
    H2: tl.constexpr,
    W2: tl.constexpr,
    C: tl.constexpr,
    # Roll shift
    roll_h: tl.constexpr,
    roll_w: tl.constexpr,
    # LayerNorm epsilon
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one output position
    row_id = tl.program_id(0)
    
    H_out = H1 * H2  # 32
    W_out = W1 * W2  # 32
    N_out = H_out * W_out  # 1024
    
    # Output position in the view grid [1, H_out, W_out, C] -> flattened
    out_h = row_id // W_out
    out_w = row_id % W_out
    
    # After roll, we read from (out_h + roll_h, out_w + roll_w) % (H_out, W_out)
    # This is equivalent to reading from (out_h - roll_h, out_w - roll_w) in the rolled tensor
    src_h = (out_h + roll_h) % H_out
    src_w = (out_w + roll_w) % W_out
    
    # Compute source flat index in the permuted/flattened layout
    # The view pattern: [B, H1, W1, H2, W2, C] -> permute(0,1,3,2,4,5) -> [B, H1, H2, W1, W2, C]
    # -> reshape to [B, H1*H2, W1*W2, C] where H1*H2=32, W1*W2=32
    # Source position maps: out_h = h1*H2 + h2, out_w = w1*W2 + w2
    # So: h1 = out_h // H2, h2 = out_h % H2, w1 = out_w // W2, w2 = out_w % W2
    # Original 6D position: [b, h1, w1, h2, w2, c]
    
    h1 = src_h // H2
    h2 = src_h % H2
    w1 = src_w // W2
    w2 = src_w % W2
    
    # Compute flat index in the 6D tensor [B, H1, W1, H2, W2, C]
    # With row-major order, strides are:
    # [H1*W1*H2*W2*C, W1*H2*W2*C, H2*W2*C, W2*C, C, 1]
    # For B=0 (first batch), the flat index is:
    # h1 * (W1*H2*W2*C) + w1 * (H2*W2*C) + h2 * (W2*C) + w2 * C + c
    # But since we load C elements at once, we need: orig_idx = h1*W1*H2*W2 + w1*H2*W2 + h2*W2 + w2
    # Then src_offset = orig_idx * C
    orig_idx = h1 * W1 * H2 * W2 + w1 * H2 * W2 + h2 * W2 + w2
    
    # Row offsets
    row_offset = row_id * C
    src_offset = orig_idx * C
    
    # Load C elements for this position
    # Use 1024 as BLOCK_SIZE (power of 2) even if C is smaller
    offsets = row_offset + tl.arange(0, 1024)
    mask = offsets < row_offset + C
    
    src_offsets = src_offset + tl.arange(0, 1024)
    src_mask = src_offsets < src_offset + C
    
    # Load the rolled value
    x = tl.load(in_3_ptr + src_offsets, mask=src_mask, other=0.0)
    
    # Compute layer norm
    mean = tl.sum(x, axis=0) / C
    var = tl.sum((x - mean) * (x - mean), axis=0) / C
    rstd = tl.rsqrt(var + eps)
    x_norm = (x - mean) * rstd
    
    # Apply weight and bias
    w = tl.load(weight_ptr + tl.arange(0, 1024), mask=src_mask, other=0.0)
    b = tl.load(bias_ptr + tl.arange(0, 1024), mask=src_mask, other=0.0)
    x_norm = x_norm * w + b
    
    # Load residual and add
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    out = x_norm + residual
    
    # Store result
    tl.store(output_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_roll_ln_add_wrapper(in_3, residual, weight, bias, eps=1e-05):
    """
    Fused kernel: roll + layer_norm + add
    All computation done in Triton.
    """
    # Ensure input is contiguous
    in_3 = in_3.contiguous()
    
    # Parse shape [B, H1, W1, H2, W2, C]
    B = in_3.shape[0]
    H1 = in_3.shape[1]
    W1 = in_3.shape[2]
    H2 = in_3.shape[3]
    W2 = in_3.shape[4]
    C = in_3.shape[5]
    
    H_out = H1 * H2
    W_out = W1 * W2
    N_out = H_out * W_out
    
    # Allocate output
    output = torch.empty_like(residual)
    
    # Launch kernel
    BLOCK_SIZE = min(1024, C)
    roll_shift = 4
    
    fused_roll_ln_add_kernel[(B * N_out,)](
        in_3_ptr=in_3,
        residual_ptr=residual,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        B=B,
        H1=H1,
        W1=W1,
        H2=H2,
        W2=W2,
        C=C,
        roll_h=roll_shift,
        roll_w=roll_shift,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the complete pattern:
    1. contiguous()
    2. view(-1, 32, 32, 768)
    3. roll(shifts=(4, 4), dims=(1, 2))
    4. view(1, 1024, 768)
    5. layer_norm((768,), ...)
    6. add
    """
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 32, 32, 768)
    tmp_4 = torch.roll(tmp_3, shifts=(4, 4), dims=(1, 2))
    tmp_5 = tmp_4.view(1, 1024, 768)
    tmp_6 = torch.nn.functional.layer_norm(tmp_5, (768,), in_1, in_0, 1e-05)
    tmp_7 = in_2 + tmp_6
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_3, in_2, in_1, in_0, 1e-05)


def replacement_func():
    return fused_roll_ln_add_wrapper