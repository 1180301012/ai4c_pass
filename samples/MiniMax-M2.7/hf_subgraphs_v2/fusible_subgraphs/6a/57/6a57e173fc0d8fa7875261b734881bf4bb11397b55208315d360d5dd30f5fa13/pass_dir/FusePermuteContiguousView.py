import torch
import triton
import triton.language as tl


def pattern(x):
    """
    Match the pattern: permute(0, 2, 1, 3) -> contiguous()
    This pattern appears in all the target graphs.
    """
    tmp_3 = x.permute(0, 2, 1, 3)
    tmp_4 = tmp_3.contiguous()
    return tmp_4


def replacement_args(x):
    return (x,)


@triton.jit
def permute_contiguous_kernel(
    input_ptr,
    output_ptr,
    stride_b: tl.constexpr,
    stride_g: tl.constexpr,
    stride_s: tl.constexpr,
    stride_h: tl.constexpr,
    B: tl.constexpr,
    G: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: permute(0, 2, 1, 3) + contiguous
    
    Input shape: [B, G, S, H]
    Output shape after permute: [B, S, G, H]
    
    Precomputes strides for faster index calculation.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < B * S * G * H
    
    # Compute strides for output [B, S, G, H] (row-major)
    out_stride_s = G * H
    out_stride_b = S * G * H
    
    # Output position [b, s, g, h] in output tensor
    b_out = offsets // (S * G * H)
    rem = offsets % (S * G * H)
    s_out = rem // (G * H)
    rem = rem % (G * H)
    g_out = rem // H
    h_out = rem % H
    
    # After permute(0, 2, 1, 3), output[b, s, g, h] = input[b, g, s, h]
    # Compute input index using precomputed strides
    input_idx = b_out * stride_b + g_out * stride_g + s_out * stride_s + h_out * stride_h
    
    # Load and store
    val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, val, mask=mask)


@torch.fx.wrap
def permute_contiguous_wrapper(x):
    """
    Wrapper function that launches the fused permute+contiguous kernel.
    """
    B, G, S, H = x.shape
    total_elements = B * S * G * H
    
    output = torch.empty((B, S, G, H), dtype=x.dtype, device=x.device)
    
    # Use larger block size for better GPU utilization
    BLOCK_SIZE = 2048
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Precompute strides for input [B, G, S, H] (row-major)
    stride_h = 1
    stride_s = H
    stride_g = S * H
    stride_b = G * S * H
    
    permute_contiguous_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=output,
        stride_b=stride_b,
        stride_g=stride_g,
        stride_s=stride_s,
        stride_h=stride_h,
        B=B,
        G=G,
        S=S,
        H=H,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return permute_contiguous_wrapper