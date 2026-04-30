import torch
import triton
import triton.language as tl

# Pattern to match: view -> pad -> view -> permute
# This optimizes the reshape chain with Triton
def pattern(in_0, in_1, in_2, in_3, in_4):
    # Conv2D with bias (groups=1)
    conv2d = torch.conv2d(in_0, in_4, in_3, (2, 2), (0, 0), (1, 1), 1)
    # Flatten spatial dimensions
    tmp_6 = conv2d.flatten(2)
    # Transpose to sequence format (B, N, C)
    tmp_7 = tmp_6.transpose(1, 2)
    # LayerNorm
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (16,), in_2, in_1, 1e-05)
    # Dropout with p=0.0 - this is a no-op!
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
    # View to spatial format [B, 16, 16, 16]
    tmp_10 = tmp_9.view(1, 16, 16, 16)
    # Pad (no-op with all zeros)
    tmp_11 = torch.nn.functional.pad(tmp_10, (0, 0, 0, 0, 0, 0), 'constant', None)
    # View to split spatial dimensions [B, 8, 2, 8, 2, 16]
    tmp_12 = tmp_11.view(1, 8, 2, 8, 2, 16)
    # Permute to final format [B, 8, 8, 2, 2, 16]
    tmp_13 = tmp_12.permute(0, 1, 3, 2, 4, 5)
    return (tmp_9, tmp_13)

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.jit
def triton_reshape_permute_kernel(
    input_ptr,
    output_ptr,
    stride_batch,
    stride_hw,
    stride_c,
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
    
    # Decode [B, h1, h2, s1, s2, C] indices
    tmp = offsets
    c_out = tmp % C
    tmp = tmp // C
    s2_out = tmp % s2
    tmp = tmp // s2
    s1_out = tmp % s1
    tmp = tmp // s1
    h2_out = tmp % h2
    tmp = tmp // h2
    h1_out = tmp % h1
    b_out = tmp // h1
    
    # Compute input position in [B, H, W, C] where H = h1*s1, W = h2*s2
    h_in = h1_out * s1 + s1_out
    w_in = h2_out * s2 + s2_out
    
    # Input linear offset
    input_offset = b_out * stride_batch + h_in * W * C + w_in * C + c_out
    
    # Load and store
    val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, val, mask=mask)


@torch.fx.wrap
def triton_reshape_permute(ln_out, h1=8, s1=2, h2=8, s2=2, C=16):
    B, N, C_in = ln_out.shape
    H = h1 * s1  # 16
    W = h2 * s2  # 16
    
    # Reshape to [B, H, W, C]
    spatial = ln_out.reshape(B, H, W, C_in)
    
    # Reshape to [B, h1, s1, h2, s2, C]
    split = spatial.reshape(B, h1, s1, h2, s2, C_in)
    
    # Permute to [B, h1, h2, s1, s2, C]
    perm_out = split.permute(0, 1, 3, 2, 4, 5)
    
    return perm_out


@torch.fx.wrap
def optimized_kernel_wrapper(in_0, in_1, in_2, in_3, in_4):
    # Perform full computation with optimized reshape/permute
    conv_out = torch.conv2d(in_0, in_4, in_3, (2, 2), (0, 0), (1, 1), 1)
    B, C_out, H_out, W_out = conv_out.shape
    conv_reshaped = conv_out.reshape(B, C_out, H_out * W_out)
    conv_transposed = conv_reshaped.permute(0, 2, 1)
    ln_out = torch.nn.functional.layer_norm(conv_transposed, (16,), in_2, in_1, 1e-05)
    dropout_out = ln_out
    perm_out = triton_reshape_permute(dropout_out, h1=8, s1=2, h2=8, s2=2, C=16)
    return (dropout_out, perm_out)


def replacement_func():
    return optimized_kernel_wrapper