import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_4 = torch.nn.functional.layer_norm(in_2, (1024,), in_1, in_0, 1e-05)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_2, in_1, in_0)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
    ],
    key=["n_rows"],
)
@triton.jit
def _layer_norm_1024_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_rows,
    eps,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    row_start = row * N_COLS

    tl.static_assert(N_COLS % BLOCK_SIZE == 0)

    sum_acc = tl.zeros((), dtype=tl.float32)
    for off in tl.static_range(0, N_COLS, BLOCK_SIZE):
        x = tl.load(x_ptr + row_start + off + cols).to(tl.float32)
        sum_acc += tl.sum(x, axis=0)
    mean = sum_acc / N_COLS

    var_acc = tl.zeros((), dtype=tl.float32)
    for off in tl.static_range(0, N_COLS, BLOCK_SIZE):
        x = tl.load(x_ptr + row_start + off + cols).to(tl.float32)
        diff = x - mean
        var_acc += tl.sum(diff * diff, axis=0)
    rstd = tl.rsqrt(var_acc / N_COLS + eps)

    for off in tl.static_range(0, N_COLS, BLOCK_SIZE):
        x = tl.load(x_ptr + row_start + off + cols).to(tl.float32)
        w = tl.load(weight_ptr + off + cols).to(tl.float32)
        b = tl.load(bias_ptr + off + cols).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(out_ptr + row_start + off + cols, y)


@torch.fx.wrap
def triton_layer_norm_1024(x, weight, bias):
    out = torch.empty_like(x)
    n_rows = x.numel() // 1024
    _layer_norm_1024_kernel[(n_rows,)](
        x,
        weight,
        bias,
        out,
        n_rows,
        1e-05,
        N_COLS=1024,
    )
    return out


def replacement_func():
    return triton_layer_norm_1024