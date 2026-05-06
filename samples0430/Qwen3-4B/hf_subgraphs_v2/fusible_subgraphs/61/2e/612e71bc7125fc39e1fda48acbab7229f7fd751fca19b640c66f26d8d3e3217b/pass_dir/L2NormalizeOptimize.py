import torch
import triton
import triton.language as tl

def pattern(in_1, in_0):
    norm = in_1.norm(p=2, dim=-1, keepdim=True)
    normalized = in_1 / norm
    transposed = in_0.t()
    transposed = transposed.to(device='cuda')
    return (normalized, transposed)

def replacement_args(in_1, in_0):
    return (in_1, in_0)

@triton.jit
def normalize_kernel(
    in_ptr: tl.bfloat16,
    out_ptr: tl.bfloat16,
    n_seq: tl.int32,
    n_feat: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    seq_id = tl.program_id(0)
    if seq_id >= n_seq:
        return

    total_sq = tl.zeros(1, dtype=tl.bfloat16)
    for i in range(n_feat):
        offset = seq_id * n_feat + i
        val = tl.load(in_ptr + offset, dtype=tl.bfloat16)
        val_sq = val * val
        total_sq += val_sq

    norm = tl.sqrt(total_sq)
    for i in range(n_feat):
        offset = seq_id * n_feat + i
        val = tl.load(in_ptr + offset, dtype=tl.bfloat16)
        normalized = val / norm
        tl.store(out_ptr + offset, normalized)

def triton_normalize(in_1, in_0):
    n_seq = in_1.shape[0]
    n_feat = in_1.shape[1]
    out_1 = torch.empty_like(in_1)
    num_programs = n_seq
    
    normalize_kernel[(num_programs,)](
        in_ptr=in_1,
        out_ptr=out_1,
        n_seq=n_seq,
        n_feat=n_feat,
        BLOCK_SIZE=128,
    )
    
    transposed = in_0.t()
    transposed = transposed.to(device='cuda')
    return (out_1, transposed)

@torch.fx.wrap
def triton_normalize_wrapper(in_1, in_0):
    return triton_normalize(in_1, in_0)

def replacement_func():
    return triton_normalize_wrapper