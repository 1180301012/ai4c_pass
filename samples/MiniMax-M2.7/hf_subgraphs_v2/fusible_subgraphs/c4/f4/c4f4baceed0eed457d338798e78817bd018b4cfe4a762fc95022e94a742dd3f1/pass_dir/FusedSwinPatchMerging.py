import torch
import triton
import triton.language as tl


@triton.jit
def fused_roll_slice_view_kernel_35_35_384_32(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Kernel for H=35, W=35, C=384, S=32"""
    H, W, C, S = 35, 35, 384, 32
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    c = offsets % C
    w = (offsets // C) % S
    h = (offsets // (S * C)) % S
    
    input_h = (h + 3) % H
    input_w = (w + 3) % W
    input_idx = input_h * W * C + input_w * C + c
    
    x = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, x, mask=mask)


@triton.jit
def fused_roll_slice_view_kernel_70_70_192_64(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Kernel for H=70, W=70, C=192, S=64"""
    H, W, C, S = 70, 70, 192, 64
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    c = offsets % C
    w = (offsets // C) % S
    h = (offsets // (S * C)) % S
    
    input_h = (h + 3) % H
    input_w = (w + 3) % W
    input_idx = input_h * W * C + input_w * C + c
    
    x = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, x, mask=mask)


@triton.jit
def fused_roll_slice_view_kernel_133_133_96_128(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Kernel for H=133, W=133, C=96, S=128"""
    H, W, C, S = 133, 133, 96, 128
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    c = offsets % C
    w = (offsets // C) % S
    h = (offsets // (S * C)) % S
    
    input_h = (h + 3) % H
    input_w = (w + 3) % W
    input_idx = input_h * W * C + input_w * C + c
    
    x = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, x, mask=mask)


def _placeholder_a(in_0, in_1, in_2, in_3):
    """Placeholder for route_a - never called directly"""
    pass


def _placeholder_b(in_0, in_1, in_2, in_3):
    """Placeholder for route_b - never called directly"""
    pass


def _placeholder_c(in_0, in_1, in_2, in_3):
    """Placeholder for route_c - never called directly"""
    pass


@torch.fx.wrap
def _fused_swin_patch_merging_dispatch(in_0, in_1, in_2, in_3, route):
    """
    Shared dispatch wrapper for all route variants.
    Uses if/elif to route to the correct implementation.
    """
    if route == "route_a":
        H, W, C, S = 35, 35, 384, 32
        output_elements = S * S * C
        output = torch.empty((1, S, S, C), dtype=in_3.dtype, device=in_3.device)
        in_3_flat = in_3.reshape(-1)
        BLOCK_SIZE = 1024
        num_programs = (output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        fused_roll_slice_view_kernel_35_35_384_32[(num_programs,)](
            input_ptr=in_3_flat,
            output_ptr=output.view(-1),
            n_elements=output_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        merged = output
        
    elif route == "route_b":
        H, W, C, S = 70, 70, 192, 64
        output_elements = S * S * C
        output = torch.empty((1, S, S, C), dtype=in_3.dtype, device=in_3.device)
        in_3_flat = in_3.reshape(-1)
        BLOCK_SIZE = 1024
        num_programs = (output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        fused_roll_slice_view_kernel_70_70_192_64[(num_programs,)](
            input_ptr=in_3_flat,
            output_ptr=output.view(-1),
            n_elements=output_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        merged = output
        
    elif route == "route_c":
        H, W, C, S = 133, 133, 96, 128
        output_elements = S * S * C
        output = torch.empty((1, S, S, C), dtype=in_3.dtype, device=in_3.device)
        in_3_flat = in_3.reshape(-1)
        BLOCK_SIZE = 1024
        num_programs = (output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        fused_roll_slice_view_kernel_133_133_96_128[(num_programs,)](
            input_ptr=in_3_flat,
            output_ptr=output.view(-1),
            n_elements=output_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        merged = output
    else:
        raise ValueError(f"Unknown route: {route}")
    
    # Step 2: Add
    tmp_8 = in_2 + merged
    
    # Step 3: LayerNorm
    tmp_9 = torch.nn.functional.layer_norm(tmp_8, (C,), in_1, in_0, 1e-05)
    
    return tmp_8, tmp_9


# ============= Pattern 1: H=35, W=35, C=384, S=32 =============
def pattern(in_0, in_1, in_2, in_3):
    """Pattern matching the Swin patch merging + add + layer_norm sequence"""
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 35, 35, 384)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4[(slice(None, None, None), slice(None, 32, None), slice(None, 32, None), slice(None, None, None))]
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.view(1, 1024, 384)
    tmp_8 = in_2 + tmp_7
    tmp_9 = torch.nn.functional.layer_norm(tmp_8, (384,), in_1, in_0, 1e-05)
    return tmp_8, tmp_9


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "route_a")


def replacement_func():
    return _fused_swin_patch_merging_dispatch