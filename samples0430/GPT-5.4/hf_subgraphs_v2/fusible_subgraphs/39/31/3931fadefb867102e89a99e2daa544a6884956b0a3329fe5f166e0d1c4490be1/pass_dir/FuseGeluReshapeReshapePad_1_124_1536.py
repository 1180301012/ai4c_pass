import torch
import triton
import triton.language as tl


# Match the full observable subgraph exactly.
def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0)
    tmp_1 = tmp_0.reshape(1, 124, 2, 768)
    tmp_2 = tmp_1.reshape(1, 248, 768)
    tmp_3 = torch.nn.functional.pad(tmp_2, (0, 0, 0, 1), 'constant', None)
    return (tmp_3,)


# Only the original input is required for the fused implementation.
def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
    ],
    key=["n_out"],
)
@triton.jit
def fused_gelu_pad_kernel(
    x_ptr,
    out_ptr,
    n_in,
    n_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    in_mask = offsets < n_in
    out_mask = offsets < n_out

    x = tl.load(x_ptr + offsets, mask=in_mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))).
    y = 0.5 * x_f32 * (1.0 + tl.erf(x_f32 * 0.7071067811865475))

    tl.store(out_ptr + offsets, y, mask=out_mask)


@torch.fx.wrap
def fused_gelu_reshape_reshape_pad(in_0):
    # Fixed-shape fusion for the matched graph:
    # [1, 124, 1536] -> GELU -> view/view -> pad on dim=1 -> [1, 249, 768]
    out = torch.empty((1, 249, 768), device=in_0.device, dtype=in_0.dtype)

    n_in = 190464
    n_out = 191232
    grid = lambda META: (triton.cdiv(n_out, META["BLOCK_SIZE"]),)

    fused_gelu_pad_kernel[grid](
        x_ptr=in_0,
        out_ptr=out,
        n_in=n_in,
        n_out=n_out,
    )
    return (out,)


def replacement_func():
    return fused_gelu_reshape_reshape_pad