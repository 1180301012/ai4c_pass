import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: matches the full position-ID chain in model.py
# ne(1) -> int() -> cumsum(dim=1) -> type_as -> +0 -> *mask -> long() -> +1
# ---------------------------------------------------------------------------
def pattern(in_0):
    tmp_1 = in_0.ne(1)
    tmp_2 = tmp_1.int()
    tmp_3 = torch.cumsum(tmp_2, dim=1)
    tmp_4 = tmp_3.type_as(tmp_2)
    tmp_5 = tmp_4 + 0
    tmp_6 = tmp_5 * tmp_2
    tmp_7 = tmp_6.long()
    tmp_8 = tmp_7 + 1
    return tmp_8


def replacement_args(in_0):
    return (in_0,)


# ---------------------------------------------------------------------------
# Triton kernel: one program per row, handles entire row in a single block
# ---------------------------------------------------------------------------
@triton.jit
def position_ids_kernel(
    in_ptr,
    out_ptr,
    L,
    BLOCK_L: tl.constexpr,
):
    row_id = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_L)
    mask_valid = offsets < L

    # Load int64 input; pad out-of-range positions with 1 so ne_mask=0 there
    in_vals = tl.load(in_ptr + row_id * L + offsets, mask=mask_valid, other=1)

    # Step 1-2: ne(1) -> int32
    ne_mask = (in_vals != 1).to(tl.int32)

    # Step 3-4: cumsum (inclusive prefix sum) — equivalent to torch.cumsum
    cumsum = tl.cumsum(ne_mask, axis=0)

    # Step 5-8: (cumsum + 0) * mask -> long() -> +1
    result = (cumsum * ne_mask).to(tl.int64) + 1

    # Store only valid positions
    tl.store(out_ptr + row_id * L + offsets, result, mask=mask_valid)


# ---------------------------------------------------------------------------
# Python wrapper (must be @torch.fx.wrap so FX doesn't trace into it)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def position_ids_wrapper(in_0):
    B, L = in_0.shape
    out = torch.empty((B, L), dtype=torch.int64, device=in_0.device)
    BLOCK_L = triton.next_power_of_2(L)
    # Each program handles exactly one row
    grid = (B,)
    position_ids_kernel[grid](
        in_ptr=in_0,
        out_ptr=out,
        L=L,
        BLOCK_L=BLOCK_L,
    )
    return out


def replacement_func():
    return position_ids_wrapper