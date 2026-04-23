import torch
import triton
import triton.language as tl


@triton.jit
def fused_reshape_swinv2_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for Swinv2: view -> permute fusion
    Input: [1, 16, 16, 16] contiguous
    Output: [1, 8, 8, 2, 2, 16]
    
    The transformation is:
    tmp_10 = tmp_9.view(1, 16, 16, 16)
    tmp_11 = pad(tmp_10) # no-op
    tmp_12 = tmp_11.view(1, 8, 2, 8, 2, 16)
    tmp_13 = tmp_12.permute(0, 1, 3, 2, 4, 5) -> [1, 8, 8, 2, 2, 16]
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # For Swinv2: input [1, 16, 16, 16], output [1, 8, 8, 2, 2, 16]
    # Both have 4096 elements (1*16*16*16 = 1*8*8*2*2*16 = 4096)
    
    # Load from input [1, 16, 16, 16]
    # Index mapping: flat_idx -> (b, h, w, c) where b=0
    b = 0
    h = (offsets // 16) % 16
    w = (offsets // (16 * 16)) % 16
    c = offsets % 16
    
    # For output [1, 8, 8, 2, 2, 16], the permute is (0, 1, 3, 2, 4, 5)
    # So output indices (b, h1, w1, h2, w2, c) map to input (b, h1*2+h2, w1*2+w2, c)
    # h = h1*2 + h2, w = w1*2 + w2
    # h1 = h // 2, h2 = h % 2, w1 = w // 2, w2 = w % 2
    
    h1 = h // 2
    h2 = h % 2
    w1 = w // 2
    w2 = w % 2
    
    # Output flat index: b*8*8*2*2*16 + h1*8*2*2*16 + w1*2*2*16 + h2*2*16 + w2*16 + c
    out_idx = h1 * (8 * 2 * 2 * 16) + w1 * (2 * 2 * 16) + h2 * (2 * 16) + w2 * 16 + c
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + out_idx, x, mask=mask)


@triton.jit
def fused_reshape_swin_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for Swin: view -> permute fusion
    Input: [1, 256, 256, 96] contiguous
    Output: [1, 32, 32, 8, 8, 96]
    
    The transformation is:
    tmp_10 = tmp_9.view(1, 256, 256, 96)
    tmp_11 = pad(tmp_10) # no-op
    tmp_12 = tmp_11.view(1, 32, 8, 32, 8, 96)
    tmp_13 = tmp_12.permute(0, 1, 3, 2, 4, 5) -> [1, 32, 32, 8, 8, 96]
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # For Swin: input [1, 256, 256, 96], output [1, 32, 32, 8, 8, 96]
    # Both have 196608 elements (1*256*256*96 = 1*32*32*8*8*96)
    
    # Load from input [1, 256, 256, 96]
    b = 0
    h = (offsets // (256 * 96)) % 256
    w = (offsets // 96) % 256
    c = offsets % 96
    
    # For output [1, 32, 32, 8, 8, 96], the permute is (0, 1, 3, 2, 4, 5)
    # So output indices (b, h1, w1, h2, w2, c) map to input (b, h1*8+h2, w1*8+w2, c)
    # h = h1*8 + h2, w = w1*8 + w2
    # h1 = h // 8, h2 = h % 8, w1 = w // 8, w2 = w % 8
    
    h1 = h // 8
    h2 = h % 8
    w1 = w // 8
    w2 = w % 8
    
    # Output flat index: b*32*32*8*8*96 + h1*32*8*8*96 + w1*8*8*96 + h2*8*96 + w2*96 + c
    out_idx = h1 * (32 * 8 * 8 * 96) + w1 * (8 * 8 * 96) + h2 * (8 * 96) + w2 * 96 + c
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + out_idx, x, mask=mask)


def reshape_8x8x2x2x16_wrapper(x):
    """Wrapper for 8x8x2x2x16 pattern (Swinv2 models)"""
    # Input shape: [1, 16, 16, 16] = 4096 elements
    # Output shape: [1, 8, 8, 2, 2, 16] = 4096 elements
    output_shape = (1, 8, 8, 2, 2, 16)
    
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_reshape_swinv2_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def reshape_32x32x8x8x96_wrapper(x):
    """Wrapper for 32x32x8x8x96 pattern (Swin models)"""
    # Input shape: [1, 256, 256, 96] = 196608 elements
    # Output shape: [1, 32, 32, 8, 8, 96] = 196608 elements
    output_shape = (1, 32, 32, 8, 8, 96)
    
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_reshape_swin_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out