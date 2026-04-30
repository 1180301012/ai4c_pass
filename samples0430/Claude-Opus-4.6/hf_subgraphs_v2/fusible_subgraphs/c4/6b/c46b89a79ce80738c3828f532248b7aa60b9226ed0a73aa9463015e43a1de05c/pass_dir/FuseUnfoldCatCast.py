import torch
import triton
import triton.language as tl


def pattern(in_0, unfold_out_1, unfold_out_2):
    tmp_1 = unfold_out_1.permute(2, 0, 1)
    tmp_2 = tmp_1.reshape(-1, 3, 384, 384)
    tmp_4 = unfold_out_2.permute(2, 0, 1)
    tmp_5 = tmp_4.reshape(-1, 3, 384, 384)
    tmp_6 = torch.cat([tmp_5, tmp_2, in_0], dim = 0)
    tmp_7 = tmp_6.to(dtype = torch.float16)
    return tmp_7


def replacement_args(in_0, unfold_out_1, unfold_out_2):
    return (in_0, unfold_out_1, unfold_out_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
        triton.Config({'BLOCK_SIZE': 8192}),
    ],
    key=['L1', 'L2', 'PATCH_ELEMS'],
)
@triton.jit
def fused_permute_reshape_cat_cast_kernel(
    in_0_ptr, uf1_ptr, uf2_ptr, out_ptr,
    L1: tl.constexpr,  # 9
    L2: tl.constexpr,  # 25
    PATCH_ELEMS: tl.constexpr,  # 442368
    BLOCK_SIZE: tl.constexpr,
):
    patch_idx = tl.program_id(0)  # 0..34
    block_id = tl.program_id(1)

    local_offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = local_offsets < PATCH_ELEMS

    # Determine source based on patch_idx
    is_from_2 = patch_idx < L2
    is_from_1 = (patch_idx >= L2) & (patch_idx < L2 + L1)

    # unfold output layout: [1, PATCH_ELEMS, L], strides: [PATCH_ELEMS*L, L, 1]
    # After permute(2,0,1)+reshape: element (patch_within_src, elem) maps to offset elem*L + patch_within_src
    
    # For unfold_out_2: patch index within source = patch_idx (0..24)
    addr_2 = local_offsets * L2 + patch_idx

    # For unfold_out_1: patch index within source = patch_idx - L2 (0..8)
    addr_1 = local_offsets * L1 + (patch_idx - L2)

    # For in_0: direct (shape [1, 3, 384, 384], contiguous)
    addr_0 = local_offsets

    # Load from appropriate source
    val_2 = tl.load(uf2_ptr + addr_2, mask=mask & is_from_2, other=0.0)
    val_1 = tl.load(uf1_ptr + addr_1, mask=mask & is_from_1, other=0.0)
    val_0 = tl.load(in_0_ptr + addr_0, mask=mask & (~is_from_2) & (~is_from_1), other=0.0)

    # Select value
    val = tl.where(is_from_2, val_2, tl.where(is_from_1, val_1, val_0))

    # Cast to float16
    val_fp16 = val.to(tl.float16)

    # Store to output (contiguous)
    out_offset = patch_idx * PATCH_ELEMS + local_offsets
    tl.store(out_ptr + out_offset, val_fp16, mask=mask)


@torch.fx.wrap
def fused_permute_reshape_cat_cast(in_0, unfold_out_1, unfold_out_2):
    PATCH_ELEMS = 3 * 384 * 384  # 442368
    L1 = 9
    L2 = 25
    N_PATCHES = L2 + L1 + 1  # 35

    out = torch.empty(N_PATCHES, 3, 384, 384, dtype=torch.float16, device=in_0.device)

    grid = lambda meta: (N_PATCHES, (PATCH_ELEMS + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'])
    fused_permute_reshape_cat_cast_kernel[grid](
        in_0, unfold_out_1, unfold_out_2, out,
        L1=L1,
        L2=L2,
        PATCH_ELEMS=PATCH_ELEMS,
    )

    return out


def replacement_func():
    return fused_permute_reshape_cat_cast