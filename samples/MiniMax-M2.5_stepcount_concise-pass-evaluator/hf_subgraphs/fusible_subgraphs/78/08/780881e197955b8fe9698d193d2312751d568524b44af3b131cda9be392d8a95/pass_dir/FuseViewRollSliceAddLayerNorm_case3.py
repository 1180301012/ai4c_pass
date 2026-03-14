import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern: view -> roll -> slice -> contiguous -> view -> add -> layer_norm
    Case 3: view(-1, 35, 35, 384), slice to 32, N=1024, C=384
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 35, 35, 384)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4[slice(None, None, None), slice(None, 32, None), slice(None, 32, None), slice(None, None, None)]
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.view(1, 1024, 384)
    tmp_8 = in_2 + tmp_7
    tmp_9 = torch.nn.functional.layer_norm(tmp_8, (384,), tmp_1, tmp_0, 1e-05)
    return (tmp_8, tmp_9)


def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments needed for the replacement kernel."""
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def fused_kernel_case3(
    in_3_ptr, in_2_ptr, in_1_ptr, in_0_ptr,
    out_8_ptr, out_9_ptr,
    N: tl.constexpr, C: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
    R: tl.constexpr, S: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for case 3: view(-1, 35, 35, 384), slice to 32, N=1024, C=384
    """
    pid = tl.program_id(0)
    
    row_offset = pid * BLOCK_SIZE
    rows = row_offset + tl.arange(0, BLOCK_SIZE)
    mask = rows < N
    
    h = rows // W
    w = rows % W
    
    # Apply roll: shift by (3, 3)
    rolled_h = (h + R) % H
    rolled_w = (w + S) % W
    
    c_offset = tl.arange(0, C)
    channel_idx = c_offset
    
    # Load in_3 data with roll applied
    in_3_offset = rolled_h * W * C + rolled_w * C + channel_idx
    x = tl.load(in_3_ptr + in_3_offset, mask=mask, other=0.0)
    
    # Load in_2 data
    in_2_offset = rows * C + channel_idx
    y = tl.load(in_2_ptr + in_2_offset, mask=mask, other=0.0)
    
    # Add
    z = x + y
    
    # Layer norm
    mean = tl.sum(z, axis=0) / C
    var = tl.sum((z - mean) * (z - mean), axis=0) / C
    eps = 1e-05
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    weight = tl.load(in_1_ptr + channel_idx)
    bias = tl.load(in_0_ptr + channel_idx)
    
    normalized = (z - mean) * inv_std
    normalized = normalized * weight + bias
    
    # Store outputs
    tl.store(out_8_ptr + in_2_offset, z, mask=mask)
    tl.store(out_9_ptr + in_2_offset, normalized, mask=mask)


@torch.fx.wrap
def fused_kernel_wrapper_case3(in_0, in_1, in_2, in_3):
    """
    Wrapper function for case 3.
    in_3 shape: [1, 5, 7, 5, 7, 384] -> H=35, W=35, C=384
    in_2 shape: [1, 1024, 384]
    """
    in_3_shape = in_3.shape
    C = in_3_shape[-1]
    H = in_3_shape[1] * in_3_shape[2]  # 5 * 7 = 35
    W = in_3_shape[3] * in_3_shape[4]  # 5 * 7 = 35
    N = in_2.numel() // C  # 1024
    
    R = 3  # roll shift height
    S = 3  # roll shift width
    
    out_8 = torch.empty_like(in_2)
    out_9 = torch.empty_like(in_2)
    
    BLOCK_SIZE = 512
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    in_3_contig = in_3.contiguous()
    
    fused_kernel_case3[grid](
        in_3_ptr=in_3_contig,
        in_2_ptr=in_2,
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        out_8_ptr=out_8,
        out_9_ptr=out_9,
        N=N,
        C=C,
        H=H,
        W=W,
        R=R,
        S=S,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_8, out_9


def replacement_func():
    return fused_kernel_wrapper_case3