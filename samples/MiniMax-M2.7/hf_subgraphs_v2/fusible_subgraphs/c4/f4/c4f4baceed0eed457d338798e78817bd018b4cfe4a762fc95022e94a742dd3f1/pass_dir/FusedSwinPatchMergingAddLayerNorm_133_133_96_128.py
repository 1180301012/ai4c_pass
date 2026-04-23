import torch
import triton
import triton.language as tl


@triton.jit
def fused_roll_slice_view_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that does:
    1. Reshape input to (H, W, C)
    2. Roll by (3, 3) along dims (1, 2)
    3. Slice to get (S, S, C) starting from (0, 0)
    4. Reshape to (1, S*S, C)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate output coordinates from linear index
    # idx -> (h, w, c) where c is innermost
    c = offsets % C
    w = (offsets // C) % S
    h = (offsets // (S * C)) % S
    
    # After roll: input position is (h+3) % H, (w+3) % W
    input_h = (h + 3) % H
    input_w = (w + 3) % W
    
    # Calculate linear index in input tensor: (n, H, W, C) -> (n * H * W * C + h * W * C + w * C + c)
    # n=0 since first dimension is 1
    input_idx = input_h * W * C + input_w * C + c
    
    # Load from input (input is already flattened as (n_elements,))
    x = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
    
    # Store to output
    tl.store(output_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def fused_roll_slice_view(in_3, H, W, C, S):
    """Fused roll + slice + view operation"""
    # Output elements: S * S * C
    output_elements = S * S * C
    
    # Allocate output
    output = torch.empty((1, S, S, C), dtype=in_3.dtype, device=in_3.device)
    
    # Flatten in_3 for kernel
    in_3_flat = in_3.reshape(-1)
    
    BLOCK_SIZE = 1024
    num_programs = (output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_roll_slice_view_kernel[(num_programs,)](
        input_ptr=in_3_flat,
        output_ptr=output.view(-1),
        n_elements=output_elements,
        H=H,
        W=W,
        C=C,
        S=S,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(in_0, in_1, in_2, in_3):
    """Pattern matching the Swin patch merging + add + layer_norm sequence"""
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 133, 133, 96)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4[(slice(None, None, None), slice(None, 128, None), slice(None, 128, None), slice(None, None, None))]
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.view(1, 16384, 96)
    tmp_8 = in_2 + tmp_7
    tmp_9 = torch.nn.functional.layer_norm(tmp_8, (96,), in_1, in_0, 1e-05)
    return tmp_8, tmp_9


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, 133, 133, 96, 128)


def replacement_func():
    return _fused_swin_patch_merging_add_layernorm


def _fused_swin_patch_merging_add_layernorm(in_0, in_1, in_2, in_3, H, W, C, S):
    """
    Fused implementation of:
    1. contiguous + view + roll + slice + contiguous + view
    2. add
    3. layer_norm
    
    Returns (tmp_8, tmp_9) matching the original pattern output.
    """
    # Step 1: Fused roll + slice + view
    merged = fused_roll_slice_view(in_3, H, W, C, S)
    
    # Step 2: Add
    tmp_8 = in_2 + merged
    
    # Step 3: LayerNorm
    tmp_9 = torch.nn.functional.layer_norm(tmp_8, (C,), in_1, in_0, 1e-05)
    
    return tmp_8, tmp_9