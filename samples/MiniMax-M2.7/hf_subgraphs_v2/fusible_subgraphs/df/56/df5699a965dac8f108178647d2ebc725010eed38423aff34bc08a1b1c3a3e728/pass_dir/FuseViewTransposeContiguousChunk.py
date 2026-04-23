import torch
import triton
import triton.language as tl

@triton.jit
def fused_view_transpose_chunk_kernel(
    input_ptr,
    output0_ptr,
    output1_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    stride_in: tl.constexpr,
    stride_out: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs view-transpose-contiguous-view-chunk.
    
    Input shape: (B, 2*D, H, W) - stored as (B, 2, D, H, W) conceptually
    Transpose(1, 2) on the 5D view
    Output shape: (B, 2*D, H, W)
    Chunk on dim=1 -> outputs of shape (B, D, H, W)
    
    D = C // 2
    """
    pid = tl.program_id(0)
    
    D = C // 2
    
    # Each program processes one element
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (B * C * H * W)
    
    # Compute output indices
    b = (offsets // (C * H * W)) % B
    c = (offsets // (H * W)) % C
    h = (offsets // W) % H
    w = offsets % W
    
    # Output channel c -> c2 = c // 2 (outer), c1 = c % 2 (inner)
    # After transpose(1,2): output channel c comes from input channel c1 at position c2
    # So: output[b, c, h, w] = input[b, c1, c2, h, w]
    # In 4D view: input[..., c1*D+c2, h, w]
    # But the input is (B, 2*D, H, W), so we access input[b, c1*D+c2, h, w]
    
    c2 = c // 2  # D dimension index (0 to D-1)
    c1 = c % 2   # 2 dimension index (0 or 1)
    
    # Input offset for the 4D tensor (B, 2*D, H, W)
    # But we need to compute it as if we have (B, 2, D, H, W)
    # c1 is the "2" dimension, c2 is the "D" dimension
    input_c = c1 * D + c2
    input_offset = b * stride_in + input_c * (H * W) + h * W + w
    
    # Load from input
    val = tl.load(input_ptr + b * stride_in + input_c * (H * W) + h * W + w, mask=mask, other=0.0)
    
    # Write to output0 (first chunk: channels 0 to D-1)
    # These are output channels c where c < D
    # For output0, output channel c corresponds to c < D
    out0_c = c
    out0_offset = b * stride_out + out0_c * (H * W) + h * W + w
    out0_mask = c < D
    tl.store(output0_ptr + out0_offset, val, mask=mask and out0_mask)
    
    # Write to output1 (second chunk: channels D to 2*D-1)
    # For output1, we need output channel c - D
    out1_c = c - D
    out1_offset = b * stride_out + out1_c * (H * W) + h * W + w
    out1_mask = c >= D
    tl.store(output1_ptr + out1_offset, val, mask=mask and out1_mask)


def triton_fused_view_transpose_chunk(in_tensor):
    """
    Fused kernel for the view-transpose-contiguous-view-chunk pattern.
    
    Args:
        in_tensor: Input tensor of shape (B, 2*D, H, W)
    
    Returns:
        Two tensors from chunk operation
    """
    B, C, H, W = in_tensor.shape
    D = C // 2
    n_elements = B * C * H * W
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output0 = torch.empty([B, D, H, W], dtype=in_tensor.dtype, device=in_tensor.device)
    output1 = torch.empty([B, D, H, W], dtype=in_tensor.dtype, device=in_tensor.device)
    
    stride_in = C * H * W
    stride_out = D * H * W
    
    fused_view_transpose_chunk_kernel[(num_programs,)](
        input_ptr=in_tensor,
        output0_ptr=output0,
        output1_ptr=output1,
        B=B,
        C=C,
        H=H,
        W=W,
        stride_in=stride_in,
        stride_out=stride_out,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output0, output1


@torch.fx.wrap
def fused_view_transpose_chunk_wrapper(cat1_out, cat2_out, B, D1, H1, W1, D2, H2, W2):
    """
    Wrapper function that fuses two parallel view-transpose-contiguous-view-chunk operations.
    
    Original:
        tmp_5 = cat([in_2, in_4], dim=1)  # (B, 40, 64, 48)
        tmp_7 = tmp_5.view(B, 2, 20, 64, 48)
        tmp_8 = torch.transpose(tmp_7, 1, 2)
        tmp_9 = tmp_8.contiguous()
        tmp_10 = tmp_9.view(B, 40, 64, 48)
        tmp_15 = tmp_10.chunk(2, dim=1)
        
        tmp_6 = cat([in_3, tmp_4], dim=1)  # (B, 80, 32, 24)
        tmp_11 = tmp_6.view(B, 2, 40, 32, 24)
        tmp_12 = torch.transpose(tmp_11, 1, 2)
        tmp_13 = tmp_12.contiguous()
        tmp_14 = tmp_13.view(B, 80, 32, 24)
        tmp_18 = tmp_14.chunk(2, dim=1)
        
        return (tmp_15[0], tmp_18[0], tmp_15[1], tmp_18[1])
    
    Fused:
        out1_0, out1_1 = triton_fused_view_transpose_chunk(cat1_out)
        out2_0, out2_1 = triton_fused_view_transpose_chunk(cat2_out)
        return (out1_0, out2_0, out1_1, out2_1)
    """
    # Process first path
    out1_0, out1_1 = triton_fused_view_transpose_chunk(cat1_out)
    
    # Process second path
    out2_0, out2_1 = triton_fused_view_transpose_chunk(cat2_out)
    
    return (out1_0, out2_0, out1_1, out2_1)


def pattern(in_2, in_3, in_4, in_5, conv_weight, conv_bias, in_6):
    """
    Match the full computation pattern including:
    1. conv2d + sigmoid + mul (attention)
    2. Two cat + view + transpose + contiguous + view + chunk operations
    """
    # Path 1: attention
    conv_out = torch.conv2d(in_6, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)
    sig = torch.sigmoid(conv_out)
    tmp_4 = in_5 * sig
    
    # Path 2: cat + reshape-transpose
    cat1 = torch.cat([in_2, in_4], dim=1)  # (B, 40, 64, 48)
    cat2 = torch.cat([in_3, tmp_4], dim=1)  # (B, 80, 32, 24)
    
    # First reshape-transpose path
    v1 = cat1.view(-1, 2, 20, 64, 48)
    t1 = torch.transpose(v1, 1, 2)
    c1 = t1.contiguous()
    r1 = c1.view(-1, 40, 64, 48)
    chunk1 = r1.chunk(2, dim=1)
    out1_0 = chunk1[0]
    out1_1 = chunk1[1]
    
    # Second reshape-transpose path
    v2 = cat2.view(-1, 2, 40, 32, 24)
    t2 = torch.transpose(v2, 1, 2)
    c2 = t2.contiguous()
    r2 = c2.view(-1, 80, 32, 24)
    chunk2 = r2.chunk(2, dim=1)
    out2_0 = chunk2[0]
    out2_1 = chunk2[1]
    
    return (out1_0, out2_0, out1_1, out2_1)


def replacement_args(in_2, in_3, in_4, in_5, conv_weight, conv_bias, in_6):
    """
    Extract the arguments needed for the replacement function.
    We need the shapes to configure the kernel.
    """
    B1 = in_2.shape[0]
    B2 = in_3.shape[0]
    B3 = in_4.shape[0]
    B4 = in_5.shape[0]
    B5 = in_6.shape[0]
    
    # Determine batch size from any input
    B = B1  # All should be the same
    
    # First path dimensions
    H1, W1 = 64, 48
    D1 = 20
    
    # Second path dimensions
    H2, W2 = 32, 24
    D2 = 40
    
    return (in_2, in_3, in_4, in_5, conv_weight, conv_bias, in_6, B, D1, H1, W1, D2, H2, W2)


def replacement_func():
    """
    Returns the replacement function.
    """
    def replacement(in_2, in_3, in_4, in_5, conv_weight, conv_bias, in_6, B, D1, H1, W1, D2, H2, W2):
        # Compute attention path
        conv_out = torch.conv2d(in_6, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)
        sig = torch.sigmoid(conv_out)
        tmp_4 = in_5 * sig
        
        # Cat operations
        cat1 = torch.cat([in_2, in_4], dim=1)  # (B, 40, 64, 48)
        cat2 = torch.cat([in_3, tmp_4], dim=1)  # (B, 80, 32, 24)
        
        # Fused view-transpose-chunk
        out1_0, out2_0, out1_1, out2_1 = fused_view_transpose_chunk_wrapper(
            cat1, cat2, B, D1, H1, W1, D2, H2, W2
        )
        
        return (out1_0, out2_0, out1_1, out2_1)
    
    return replacement