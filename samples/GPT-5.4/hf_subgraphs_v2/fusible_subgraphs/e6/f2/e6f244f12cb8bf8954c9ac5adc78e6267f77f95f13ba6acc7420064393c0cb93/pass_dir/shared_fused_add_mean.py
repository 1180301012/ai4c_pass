import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_S": 32}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_S": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_S": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 256}, num_warps=4, num_stages=2),
    ],
    key=["S", "op_kind"],
)
@triton.jit

def _fused_add_mean_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    mean_ptr,
    rows,
    S,
    op_kind: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= rows:
        return

    base = row * S
    acc = tl.sum(tl.zeros((BLOCK_S,), dtype=tl.float32), axis=0)

    for start in range(0, S, BLOCK_S):
        offs = start + tl.arange(0, BLOCK_S)
        mask = offs < S

        a = tl.load(a_ptr + base + offs, mask=mask, other=0.0)
        if op_kind == 1:
            out = a
        elif op_kind == 2:
            b = tl.load(b_ptr + base + offs, mask=mask, other=0.0)
            out = a + b
        else:
            b = tl.load(b_ptr + base + offs, mask=mask, other=0.0)
            c = tl.load(c_ptr + base + offs, mask=mask, other=0.0)
            tmp = a + b
            out = tmp + c

        tl.store(out_ptr + base + offs, out, mask=mask)
        acc += tl.sum(out.to(tl.float32), axis=0)

    mean_val = acc / S
    tl.store(mean_ptr + row, mean_val)


@torch.fx.wrap
def shared_replacement(a, b, c, route):
    base = a
    n = base.size(0)
    ch = base.size(1)
    h = base.size(2)
    w = base.size(3)
    rows = n * ch
    s = h * w

    out = torch.empty_like(base)
    mean_out = torch.empty((n, ch, 1, 1), device=base.device, dtype=base.dtype)

    if route == "identity":
        op_kind = 1
    elif route == "add2":
        op_kind = 2
    elif route == "add3":
        op_kind = 3
    else:
        raise RuntimeError("unknown route")

    grid = (rows,)
    _fused_add_mean_kernel[grid](
        a,
        b,
        c,
        out,
        mean_out,
        rows,
        s,
        op_kind=op_kind,
    )
    return out, mean_out