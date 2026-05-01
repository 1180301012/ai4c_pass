import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
    ],
    key=['B', 'C', 'HW'],
)
@triton.jit
def permute_reshape_sigmoid_kernel(
    input_ptr,
    output_ptr,
    B, C, HW,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: permute(0,2,3,1) + reshape + sigmoid
    Input shape:  [B, C, H, W]  (contiguous, from conv2d)
    Output shape: [B*HW*C] flat (will be viewed as [B, HW, C])
    
    Output flat index i encodes:
        c  = i % C
        sb = i // C      (= b*HW + s)
        b  = sb // HW
        s  = sb % HW
    maps to input flat index:
        b*(C*HW) + c*HW + s
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    c  = offsets % C
    sb = offsets // C
    b  = sb // HW
    s  = sb % HW

    in_idx = b * (C * HW) + c * HW + s

    x     = tl.load(input_ptr + in_idx, mask=mask)
    x_f32 = x.to(tl.float32)
    y     = 1.0 / (1.0 + tl.exp(-x_f32))
    y_out = y.to(x.dtype)

    tl.store(output_ptr + offsets, y_out, mask=mask)


@torch.fx.wrap
def fused_permute_reshape_sigmoid(x):
    """
    x : [B, C, H, W]  – output of conv2d
    Returns [B, H*W, C] with sigmoid applied.
    (equiv. to x.permute(0,2,3,1).reshape(B,-1,C).sigmoid())
    """
    B  = x.shape[0]
    C  = x.shape[1]
    H  = x.shape[2]
    W  = x.shape[3]
    HW = H * W
    total = B * C * HW

    output = torch.empty(total, dtype=x.dtype, device=x.device)

    grid = lambda meta: ((total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    permute_reshape_sigmoid_kernel[grid](
        x, output,
        B, C, HW,
        total,
    )

    return output.reshape(B, HW, C)