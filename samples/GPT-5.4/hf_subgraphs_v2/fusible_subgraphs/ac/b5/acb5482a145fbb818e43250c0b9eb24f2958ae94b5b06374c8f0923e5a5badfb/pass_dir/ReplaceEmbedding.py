import torch
import triton
import triton.language as tl


EMBED_DIM = 1536


def pattern(in_1, in_2):
    tmp_3 = torch.nn.functional.embedding(in_1, in_2, None, None, 2.0, False, False)
    return tmp_3


def replacement_args(in_1, in_2):
    return (in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 256, "EMBED_DIM": 1536}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 512, "EMBED_DIM": 1536}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 512, "EMBED_DIM": 1536}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 1024, "EMBED_DIM": 1536}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 1024, "EMBED_DIM": 1536}, num_warps=8, num_stages=2),
    ],
    key=["num_tokens"],
)
@triton.jit
def _embedding_row_kernel(
    ids_ptr,
    weight_ptr,
    out_ptr,
    num_tokens,
    BLOCK_N: tl.constexpr,
    EMBED_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0)
    token_mask = token_idx < num_tokens

    row_idx = tl.load(ids_ptr + token_idx, mask=token_mask, other=0)
    row_idx = row_idx.to(tl.int32)

    row_base = row_idx * EMBED_DIM
    out_base = token_idx * EMBED_DIM

    for start in tl.static_range(0, EMBED_DIM, BLOCK_N):
        offsets = start + tl.arange(0, BLOCK_N)
        mask = token_mask & (offsets < EMBED_DIM)
        vals = tl.load(weight_ptr + row_base + offsets, mask=mask, other=0)
        tl.store(out_ptr + out_base + offsets, vals, mask=mask)


@torch.fx.wrap
def triton_embedding(in_1, in_2):
    num_tokens = in_1.numel()
    out = torch.empty((*in_1.shape, EMBED_DIM), device=in_2.device, dtype=in_2.dtype)

    _embedding_row_kernel[(num_tokens,)](
        in_1,
        in_2,
        out,
        num_tokens,
    )
    return out


def replacement_func():
    return triton_embedding