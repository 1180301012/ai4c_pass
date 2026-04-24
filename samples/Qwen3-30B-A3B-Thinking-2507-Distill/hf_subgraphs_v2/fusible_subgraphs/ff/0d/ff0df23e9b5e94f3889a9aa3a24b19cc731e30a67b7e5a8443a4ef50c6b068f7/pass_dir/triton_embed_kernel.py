import torch
import triton
import triton.language as tl


@triton.jit
def fused_embed_permute_kernel(
    indices_ptr,   # [n, n] int64, contiguous
    weight_ptr,    # [num_emb, emb_dim] float, contiguous
    output_ptr,    # [B, emb_dim, n, n] float, contiguous
    n, emb_dim, B,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    # No mask needed if total is divisible by BLOCK; otherwise mask
    mask = offsets < n * n * emb_dim * B

    # Decode [b, k, i, j] indices from flat linear index
    # Output layout: [B, emb_dim, n, n], strides [emb_dim*n*n, n*n, n, 1]
    j = offsets % n
    tmp = offsets // n
    i = tmp % n
    tmp2 = tmp // n
    k = tmp2 % emb_dim
    b = tmp2 // emb_dim

    # Load index value for position (i, j) — cast to int32 for safe pointer arithmetic
    idx = tl.load(indices_ptr + i * n + j, mask=mask, other=0).to(tl.int32)

    # Gather embedding weight[k, idx]
    emb_val = tl.load(weight_ptr + k * emb_dim + idx)

    # Write to output[b, k, i, j]
    out_offset = b * emb_dim * n * n + k * n * n + i * n + j
    tl.store(output_ptr + out_offset, emb_val, mask=mask)


@torch.fx.wrap
def fused_embed_permute(in_0, in_1, B, n, emb_dim):
    # dtype and device are Python properties, not dispatched — safe on PosionDispatchTensor
    dtype = in_0.dtype
    device = in_0.device

    # torch.as_tensor with device uses aten._to_copy (whitelisted in PosionDispatchTensor)
    # and returns a PosionDispatchTensor with the correct dtype
    indices_cuda = torch.as_tensor(in_1, device=device)
    weight_cuda = torch.as_tensor(in_0, device=device)

    # Allocate output: dtype is a concrete torch.dtype (not PosionDispatch), so this
    # goes through the normal dispatcher and produces a regular tensor with the right dtype
    output = torch.empty((B, emb_dim, n, n), dtype=dtype, device=device)

    total = B * emb_dim * n * n
    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)

    fused_embed_permute_kernel[grid](
        indices_cuda, weight_cuda, output,
        n, emb_dim, B,
        BLOCK=BLOCK,
    )

    return output