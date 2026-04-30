import torch
import triton
import triton.language as tl

# Pattern to match the full computation
def pattern(in_0, in_1, in_2, in_3, in_4):
    # Conv2D with bias
    conv2d = torch.conv2d(in_0, in_4, in_3, (2, 2), (0, 0), (1, 1), 1)
    tmp_6 = conv2d.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (16,), in_2, in_1, 1e-05)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
    tmp_10 = tmp_9.view(1, 16, 16, 16)
    tmp_11 = torch.nn.functional.pad(tmp_10, (0, 0, 0, 0, 0, 0), 'constant', None)
    tmp_12 = tmp_11.view(1, 8, 2, 8, 2, 16)
    tmp_13 = tmp_12.permute(0, 1, 3, 2, 4, 5)
    return (tmp_9, tmp_13)

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.jit
def triton_reshape_permute_kernel(
    input_ptr,
    output_ptr,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    h1: tl.constexpr,
    s1: tl.constexpr,
    h2: tl.constexpr,
    s2: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Decode output position [B, h1, h2, s1, s2, C]
    tmp = offsets
    c_idx = tmp % C
    tmp = tmp // C
    s2_idx = tmp % s2
    tmp = tmp // s2
    s1_idx = tmp % s1
    tmp = tmp // s1
    h2_idx = tmp % h2
    tmp = tmp // h2
    h1_idx = tmp % h1
    b_idx = tmp // h1
    
    # Map to input position [B, H, W, C] = [B, h1*s1, h2*s2, C]
    h_in = h1_idx * s1 + s1_idx
    w_in = h2_idx * s2 + s2_idx
    
    # Linear offset in input [B, H*W*C] 
    input_offset = b_idx * H * W * C + h_in * W * C + w_in * C + c_idx
    
    val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, val, mask=mask)


@torch.fx.wrap
def triton_reshape_permute_optimized(input_tensor):
    """Fused reshape + permute using Triton kernel"""
    B, N, C = input_tensor.shape
    h1, s1, h2, s2 = 8, 2, 8, 2
    H, W = h1 * s1, h2 * s2
    
    # Output shape: [B, h1, h2, s1, s2, C]
    out_shape = (B, h1, h2, s1, s2, C)
    output = torch.empty(out_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Compute total elements
    n_elements = B * h1 * h2 * s1 * s2 * C
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    triton_reshape_permute_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        H=H,
        W=W,
        C=C,
        h1=h1,
        s1=s1,
        h2=h2,
        s2=s2,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


@torch.fx.wrap
def optimized_wrapper(in_0, in_1, in_2, in_3, in_4):
    """Optimized implementation using Triton for reshape/permute"""
    # Original computation
    conv_out = torch.conv2d(in_0, in_4, in_3, (2, 2), (0, 0), (1, 1), 1)
    B, C_out, H_out, W_out = conv_out.shape
    conv_reshaped = conv_out.reshape(B, C_out, H_out * W_out)
    conv_transposed = conv_reshaped.permute(0, 2, 1)
    ln_out = torch.nn.functional.layer_norm(conv_transposed, (16,), in_2, in_1, 1e-05)
    dropout_out = ln_out  # p=0.0 is no-op
    
    # Optimized reshape + permute using Triton
    perm_out = triton_reshape_permute_optimized(dropout_out)
    
    return (dropout_out, perm_out)


def replacement_func():
    return optimized_wrapper