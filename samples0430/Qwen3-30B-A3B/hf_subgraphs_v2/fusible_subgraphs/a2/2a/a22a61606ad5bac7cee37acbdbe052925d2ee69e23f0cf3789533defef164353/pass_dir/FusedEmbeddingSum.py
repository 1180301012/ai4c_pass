import torch
import triton
import triton.language as tl

# Pattern matching function

def pattern(in_0, in_1, in_2, in_3, in_6, in_7):
    emb1 = torch.nn.functional.embedding(in_0, in_3, 0, None, 2.0, False, False)
    emb2 = torch.nn.functional.embedding(in_6, in_2, None, None, 2.0, False, False)
    sum1 = emb1 + emb2
    emb3 = torch.nn.functional.embedding(in_7, in_1, None, None, 2.0, False, False)
    sum2 = sum1 + emb3
    return sum2

# Argument extraction function

def replacement_args(in_0, in_1, in_2, in_3, in_6, in_7):
    return (in_0, in_1, in_2, in_3, in_6, in_7)

# Triton kernel
@triton.jit
def fused_embedding_add_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr, in_6_ptr, in_7_ptr,
    out_ptr,
    B, L, D,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of [B, L] elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < B * L

    # Calculate (b_idx, l_idx)
    b_idx = offsets // L
    l_idx = offsets % L

    # Load the input indices
    idx_0 = tl.load(in_0_ptr + b_idx * L + l_idx, mask=mask, other=0)
    idx_6 = tl.load(in_6_ptr + b_idx * L + l_idx, mask=mask, other=0)
    # For in_7, it's [1, L], so we use the same l_idx for all b_idx
    idx_7 = tl.load(in_7_ptr + l_idx, mask=mask, other=0)

    # Load the weight vectors for the embeddings
    # For in_3 (word embeddings): [30592, D]
    emb1 = tl.load(in_3_ptr + idx_0 * D + tl.arange(0, D), mask=mask, other=0.0)
    # For in_2 (token type embeddings): [4, D]
    emb2 = tl.load(in_2_ptr + idx_6 * D + tl.arange(0, D), mask=mask, other=0.0)
    # For in_1 (position embeddings): [512, D]
    emb3 = tl.load(in_1_ptr + idx_7 * D + tl.arange(0, D), mask=mask, other=0.0)

    # Sum the embeddings
    emb_sum = emb1 + emb2 + emb3

    # Store the result
    tl.store(out_ptr + offsets * D + tl.arange(0, D), emb_sum, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_embedding_add(in_0, in_1, in_2, in_3, in_6, in_7):
    B, L = in_0.shape
    D = in_3.shape[1]  # D = 1024 (from weight_meta)
    
    # Create output tensor
    out = torch.empty((B, L, D), dtype=in_3.dtype, device=in_3.device)

    # Determine block size and grid size for Triton
    BLOCK_SIZE = 1024
    num_programs = (B * L + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    fused_embedding_add_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        in_3_ptr=in_3,
        in_6_ptr=in_6,
        in_7_ptr=in_7,
        out_ptr=out,
        B=B,
        L=L,
        D=D,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_embedding_add