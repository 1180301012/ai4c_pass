import torch
import triton
import triton.language as tl


@triton.jit
def _relu_norm_scale_kernel(
    in0_ptr,
    in1_ptr,
    in2_ptr,
    out_ptr,
    rows,
    D,
    scale_const,
    clamp_min,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DTYPE_CODE: tl.constexpr,
):
    block_id = tl.program_id(0)
    row_base = block_id * BLOCK_ROWS

    in0 = tl.load(in0_ptr).to(tl.float32)
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    for i in tl.static_range(BLOCK_ROWS):
        row = row_base + i
        row_ok = row < rows

        norm_val = tl.load(in2_ptr + row, mask=row_ok, other=1.0).to(tl.float32)
        s = tl.maximum(norm_val * scale_const, clamp_min)
        factor = in0 / s

        full_mask = d_mask & row_ok
        x = tl.load(in1_ptr + row * D + d_offs, mask=full_mask, other=0.0).to(tl.float32)
        out = x * factor

        if DTYPE_CODE == 0:
            tl.store(out_ptr + row * D + d_offs, out.to(tl.float16), mask=full_mask)
        elif DTYPE_CODE == 1:
            tl.store(out_ptr + row * D + d_offs, out.to(tl.bfloat16), mask=full_mask)
        else:
            tl.store(out_ptr + row * D + d_offs, out, mask=full_mask)


@torch.fx.wrap
def relu_norm_scale_dispatch(in0, in1, in2, route):
    """Shared dispatch: in1=[B,N,D] flattened tensor, in2=[B,N,1] norm result."""
    B = in1.shape[0]
    N = in1.shape[1]
    D = in1.numel() // (B * N)
    rows = B * N

    BLOCK_D = 1
    while BLOCK_D < D:
        BLOCK_D *= 2

    BLOCK_ROWS = 8
    n_blocks = (rows + BLOCK_ROWS - 1) // BLOCK_ROWS

    dtype_map = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}
    DTYPE_CODE = dtype_map.get(in1.dtype, 2)

    if route == "s144":
        scale_val = 0.14433756729740643
    else:  # route == "s072"
        scale_val = 0.07216878364870322

    out = torch.empty((B, N, D), dtype=in1.dtype, device=in1.device)

    _relu_norm_scale_kernel[(n_blocks,)](
        in0, in1, in2, out,
        rows, D,
        scale_val, 1e-5,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_D=BLOCK_D,
        DTYPE_CODE=DTYPE_CODE,
        num_warps=4,
    )

    return out