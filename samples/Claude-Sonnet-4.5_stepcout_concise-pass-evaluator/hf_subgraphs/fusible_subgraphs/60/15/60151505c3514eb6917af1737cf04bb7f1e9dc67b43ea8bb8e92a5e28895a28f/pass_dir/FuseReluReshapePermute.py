import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_1 = in_0.reshape(32, 256, -1)
    tmp_2 = tmp_0.reshape(32, 256, -1)
    tmp_3 = tmp_2.permute(0, 2, 1)
    return (tmp_3, tmp_1)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['N'],
)
@triton.jit
def fused_relu_reshape_permute_kernel(
    in_ptr,
    out_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < N
    
    # Offsets are in output layout [B, H, C]
    temp = offsets
    c = temp % C
    temp = temp // C
    h = temp % H
    b = temp // H
    
    # Input layout [B, C, H] (ignoring last dimension of 1)
    in_offset = b * (C * H) + c * H + h
    
    val = tl.load(in_ptr + in_offset, mask=mask, other=0.0)
    val = tl.maximum(val, 0.0)
    
    tl.store(out_ptr + offsets, val, mask=mask)

@torch.fx.wrap
def fused_relu_reshape_permute(in_0, in_1):
    B, C, H, W = in_1.shape
    
    # Output shape: [B, H, C]
    out = torch.empty((B, H, C), device=in_1.device, dtype=in_1.dtype)
    
    N = B * H * C
    grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    fused_relu_reshape_permute_kernel[grid](
        in_1, out, B, C, H, N
    )
    
    # Handle the reshape of in_0
    tmp_1 = in_0.reshape(B, C, -1)
    
    return (out, tmp_1)

def replacement_func():
    return fused_relu_reshape_permute