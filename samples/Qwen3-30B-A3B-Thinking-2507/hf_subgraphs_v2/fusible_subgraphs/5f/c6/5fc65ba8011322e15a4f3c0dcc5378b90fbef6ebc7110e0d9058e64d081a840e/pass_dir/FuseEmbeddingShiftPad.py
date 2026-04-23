import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_2 = torch.nn.functional.embedding(in_0, in_1, 0, None, 2.0, False, False)
    tmp_3 = tmp_2[(slice(None, None, None), slice(1, None, None))]
    tmp_4 = torch.nn.functional.pad(tmp_3, [0, 0, 0, 1, 0, 0], 'constant', 0.0)
    tmp_5 = tmp_2[(slice(None, None, None), slice(None, -1, None))]
    tmp_6 = torch.nn.functional.pad(tmp_5, [0, 0, 1, 0, 0, 0], 'constant', 0.0)
    tmp_7 = torch.cat([tmp_4, tmp_2, tmp_6], dim=2)
    return (tmp_7,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_embedding_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    batch,
    seq_len,
    hidden_size,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Convert 1D offset to 3D (b, s, h)
    idx_b = (offsets // (seq_len * 3 * hidden_size)) % batch
    idx_s = (offsets // (3 * hidden_size)) % seq_len
    idx_h = offsets % (3 * hidden_size)

    current_idx = tl.load(in_0_ptr + idx_b * seq_len + idx_s)
    left_idx = tl.load(in_0_ptr + idx_b * seq_len + idx_s + 1) if idx_s < seq_len - 1 else 0
    right_idx = tl.load(in_0_ptr + idx_b * seq_len + idx_s - 1) if idx_s > 0 else 0

    if idx_h < hidden_size:
        if idx_s < seq_len - 1:
            val = tl.load(in_1_ptr + left_idx * hidden_size + idx_h)
        else:
            val = 0.0
    elif idx_h < 2 * hidden_size:
        val = tl.load(in_1_ptr + current_idx * hidden_size + (idx_h - hidden_size))
    else:
        if idx_s > 0:
            val = tl.load(in_1_ptr + right_idx * hidden_size + (idx_h - 2 * hidden_size))
        else:
            val = 0.0

    tl.store(out_ptr + offsets, val, mask=mask)

@torch.fx.wrap
def fused_embedding(in_0, in_1):
    batch, seq_len = in_0.shape
    hidden_size = in_1.shape[1]
    n_out = 3 * hidden_size
    n_elements = batch * seq_len * n_out
    out = torch.empty((batch, seq_len, n_out), dtype=in_1.dtype, device=in_1.device)
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    fused_embedding_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        batch=batch,
        seq_len=seq_len,
        hidden_size=hidden_size,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def replacement_func():
    return fused_embedding