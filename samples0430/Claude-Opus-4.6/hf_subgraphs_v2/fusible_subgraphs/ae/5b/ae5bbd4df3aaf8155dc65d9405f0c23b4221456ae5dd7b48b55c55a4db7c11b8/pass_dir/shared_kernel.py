import torch
import triton
import triton.language as tl


@triton.jit
def _fused_gru_rel_pos_kernel(
    in_3_ptr, in_1_ptr, in_0_ptr, in_2_ptr, out_ptr,
    total_rows, S,
    BLOCK_M: tl.constexpr,
    K: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    row_offsets = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < total_rows
    k_range = tl.arange(0, K)

    # Load input block [BLOCK_M, K]
    x = tl.load(in_3_ptr + row_offsets[:, None] * K + k_range[None, :],
                mask=row_mask[:, None], other=0.0).to(tl.float32)

    # Load and sum weight rows for gate (rows 0-3)
    w_gate = tl.load(in_1_ptr + k_range).to(tl.float32)
    w_gate += tl.load(in_1_ptr + K + k_range).to(tl.float32)
    w_gate += tl.load(in_1_ptr + 2 * K + k_range).to(tl.float32)
    w_gate += tl.load(in_1_ptr + 3 * K + k_range).to(tl.float32)

    # Load and sum weight rows for factor (rows 4-7)
    w_factor = tl.load(in_1_ptr + 4 * K + k_range).to(tl.float32)
    w_factor += tl.load(in_1_ptr + 5 * K + k_range).to(tl.float32)
    w_factor += tl.load(in_1_ptr + 6 * K + k_range).to(tl.float32)
    w_factor += tl.load(in_1_ptr + 7 * K + k_range).to(tl.float32)

    # Load and sum biases
    b_gate = (tl.load(in_0_ptr).to(tl.float32) + tl.load(in_0_ptr + 1).to(tl.float32) +
              tl.load(in_0_ptr + 2).to(tl.float32) + tl.load(in_0_ptr + 3).to(tl.float32))
    b_factor = (tl.load(in_0_ptr + 4).to(tl.float32) + tl.load(in_0_ptr + 5).to(tl.float32) +
                tl.load(in_0_ptr + 6).to(tl.float32) + tl.load(in_0_ptr + 7).to(tl.float32))

    # Compute dot products: [BLOCK_M]
    gate_vals = tl.sum(x * w_gate[None, :], axis=1) + b_gate
    factor_vals = tl.sum(x * w_factor[None, :], axis=1) + b_factor

    # Sigmoid
    gate = tl.sigmoid(gate_vals)
    factor = tl.sigmoid(factor_vals)

    # Load in_2 for each row's head
    h_ids = row_offsets // S
    in_2_vals = tl.load(in_2_ptr + h_ids, mask=row_mask, other=0.0).to(tl.float32)

    # Combine: gate * (factor * in_2 - 1.0) + 2.0
    result = gate * (factor * in_2_vals - 1.0) + 2.0

    # Store
    tl.store(out_ptr + row_offsets, result, mask=row_mask)


@torch.fx.wrap
def _fused_gru_rel_pos_dispatch(in_0, in_1, in_2, in_3, H, S):
    K = 64
    total_rows = H * S

    out = torch.empty(1, H, S, 1, dtype=in_3.dtype, device=in_3.device)

    BLOCK_M = 32
    grid = ((total_rows + BLOCK_M - 1) // BLOCK_M,)

    _fused_gru_rel_pos_kernel[grid](
        in_3, in_1, in_0, in_2, out,
        total_rows, S,
        BLOCK_M=BLOCK_M, K=K,
        num_warps=2, num_stages=1,
    )

    return out