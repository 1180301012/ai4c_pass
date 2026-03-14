import torch
import triton
import triton.language as tl

def pattern(conv_out, pos_embed):
    """
    Pattern: flatten(2) -> transpose(1,2) -> add -> dropout(p=0.0)
    """
    tmp_6 = conv_out.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    tmp_8 = tmp_7 + pos_embed
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
    return tmp_9

def replacement_args(conv_out, pos_embed):
    return (conv_out, pos_embed)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def fused_flatten_transpose_add_kernel(
    conv_ptr,
    pos_ptr,
    out_ptr,
    B, C, H, W,
    M, N,
    stride_cb, stride_cc, stride_ch, stride_cw,
    stride_pb, stride_pm, stride_pn,
    stride_ob, stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Read from conv_out[b, c, h, w]
    2. Transpose to [b, h*w, c] 
    3. Add pos_embed[b, h*w, c]
    4. Write output
    
    conv_out shape: [B, C, H, W]
    pos_embed shape: [B, M, N] where M = H*W, N = C
    output shape: [B, M, N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Compute offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # For each position (m, n) in output:
    # m corresponds to spatial position (h*W + w)
    # n corresponds to channel c
    # We need to read conv_out[b, n, h, w]
    
    # Compute h and w from m
    # m = h * W + w, so h = m // W, w = m % W
    h = offs_m // W
    w = offs_m % W
    c = offs_n
    
    # Load from conv_out: [B, C, H, W]
    conv_offsets = (pid_b * stride_cb + 
                    c[None, :] * stride_cc + 
                    h[:, None] * stride_ch + 
                    w[:, None] * stride_cw)
    conv_data = tl.load(conv_ptr + conv_offsets, mask=mask, other=0.0)
    
    # Load from pos_embed: [B, M, N]
    pos_offsets = pid_b * stride_pb + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn
    pos_data = tl.load(pos_ptr + pos_offsets, mask=mask, other=0.0)
    
    # Add
    result = conv_data + pos_data
    
    # Store to output: [B, M, N]
    out_offsets = pid_b * stride_ob + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptr + out_offsets, result, mask=mask)

@torch.fx.wrap
def fused_flatten_transpose_add(conv_out, pos_embed):
    """
    Fused operation combining flatten(2) -> transpose(1,2) -> add -> dropout(p=0.0)
    
    conv_out: [B, C, H, W]
    pos_embed: [B, H*W, C]
    returns: [B, H*W, C]
    """
    B, C, H, W = conv_out.shape
    M = H * W
    N = C
    
    # Output shape: [B, M, N]
    output = torch.empty((B, M, N), device=conv_out.device, dtype=conv_out.dtype)
    
    # Grid
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 256
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
        B,
    )
    
    fused_flatten_transpose_add_kernel[grid](
        conv_out, pos_embed, output,
        B, C, H, W,
        M, N,
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2), conv_out.stride(3),
        pos_embed.stride(0), pos_embed.stride(1), pos_embed.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
    )
    
    return output

def replacement_func():
    return fused_flatten_transpose_add