import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_3 = torch.nn.functional.embedding(in_1, in_2, None, None, 2.0, False, False)
    tmp_4 = in_0.long()
    return (tmp_3, tmp_4)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 256}, num_warps=8, num_stages=2),
    ],
    key=["n_tokens", "hidden_dim"],
)
@triton.jit
def embedding_gather_kernel(
    ids_ptr,
    weight_ptr,
    out_ptr,
    n_tokens,
    hidden_dim,
    stride_w0,
    stride_w1,
    stride_o0,
    stride_o1,
    BLOCK_D: tl.constexpr,
):
    pid_token = tl.program_id(0)
    pid_block = tl.program_id(1)

    if pid_token >= n_tokens:
        return

    offs_d = pid_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < hidden_dim

    token_idx = tl.load(ids_ptr + pid_token).to(tl.int64)
    weight_offsets = token_idx * stride_w0 + offs_d * stride_w1
    out_offsets = pid_token * stride_o0 + offs_d * stride_o1

    vals = tl.load(weight_ptr + weight_offsets, mask=mask_d)
    tl.store(out_ptr + out_offsets, vals, mask=mask_d)


@torch.fx.wrap
def fused_embedding_identity_long(in_0, in_1, in_2):
    n_tokens = in_1.numel()
    hidden_dim = in_2.shape[1]

    out = torch.empty((n_tokens, hidden_dim), device=in_2.device, dtype=in_2.dtype)

    grid = (n_tokens, triton.cdiv(hidden_dim, 256))
    embedding_gather_kernel[grid](
        in_1,
        in_2,
        out,
        n_tokens,
        hidden_dim,
        in_2.stride(0),
        in_2.stride(1),
        out.stride(0),
        out.stride(1),
    )

    return (out.reshape(*in_1.shape, hidden_dim), in_0)


def replacement_func():
    return fused_embedding_identity_long