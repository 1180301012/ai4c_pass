import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    matmul = in_1 @ in_3
    tmp_1 = matmul.reshape(-1, 16, 31)
    tmp_2 = torch.nn.functional.pad(tmp_1, [0, 1], 'constant', None)
    tmp_3 = tmp_2.flatten(1)
    tmp_4 = torch.nn.functional.pad(tmp_3, [0, 15], 'constant', None)
    tmp_5 = tmp_4.reshape(-1, 17, 31)
    tmp_6 = tmp_5[(slice(None, None, None), slice(None, 16, None), slice(15, None, None))]
    tmp_7 = tmp_6.reshape(4, 16, 1, 16, 16)
    tmp_8 = tmp_7.expand(-1, -1, 16, -1, -1)
    tmp_9 = tmp_8.permute((0, 3, 1, 4, 2))
    tmp_10 = tmp_9 + in_2
    tmp_11 = tmp_10.reshape(4, 256, 256)
    tmp_12 = in_0 + tmp_11
    tmp_13 = tmp_12.softmax(dim=-1)
    matmul_1 = tmp_13 @ in_4
    tmp_15 = matmul_1.transpose(-1, -2)
    return (tmp_15,)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4, "route_256")


from pass_dir.FuseBotNetAttention_256_clean import fused_botnet_attention_dispatch


def replacement_func():
    return fused_botnet_attention_dispatch
    s1 = q % S_h
    s2_q = q // S_h

    # ------------------------------------------------
    # Step 1: Compute  in_1[b, s1, s2_q, :] @ in_3  -> [D_pos] vector
    # ------------------------------------------------
    # We accumulate matmul_result[D_pos] in registers
    # D_pos is small (31 or 15), so we can hold it entirely

    # Initialize matmul_result
    mlp_offset = tl.arange(0, D_pos)  # [D_pos] - but D_pos must be constexpr...
    # Since D_pos differs per route, we use a constexpr max and mask
    # For route_256: D_pos=31, for route_64: D_pos=15
    # We'll handle both via constexpr branching

    if route_hash == 0:  # route_256
        D_POS_MAX: tl.constexpr = 32
    else:  # route_64
        D_POS_MAX: tl.constexpr = 16

    d_pos_offsets = tl.arange(0, D_POS_MAX)
    d_pos_mask = d_pos_offsets < D_pos

    matmul_result = tl.zeros([D_POS_MAX], dtype=tl.float32)

    # Tile over D_in dimension
    for d_in_start in range(0, D_in, BLOCK_DIN):
        d_in_offsets = d_in_start + tl.arange(0, BLOCK_DIN)
        d_in_mask = d_in_offsets < D_in

        # Load query vector: in_1[b, s1, s2_q, d_in_offsets]
        q_offsets = b * s1_b + s1 * s1_si + s2_q * s1_sq + d_in_offsets * s1_d
        q_vec = tl.load(in_1_ptr + q_offsets, mask=d_in_mask, other=0.0)  # [BLOCK_DIN]

        # Load weight matrix tile: in_3[d_in_offsets, d_pos_offsets]
        # in_3 shape: [D_in, D_pos]
        w_offsets = d_in_offsets[:, None] * s3_d + d_pos_offsets[None, :] * s3_p
        w_mask = d_in_mask[:, None] & d_pos_mask[None, :]
        w_tile = tl.load(in_3_ptr + w_offsets, mask=w_mask, other=0.0)  # [BLOCK_DIN, D_POS_MAX]

        # Accumulate dot product
        matmul_result += tl.sum(q_vec[:, None] * w_tile, axis=0)  # [D_POS_MAX]

    # ------------------------------------------------
    # Step 2: Flash attention - online softmax + matmul with in_4
    # Process key dimension in tiles
    # ------------------------------------------------
    # Running max, running sum of exp, running output accumulation
    running_max = tl.full([BLOCK_D], float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_D], dtype=tl.float32)
    running_out = tl.zeros([BLOCK_D], dtype=tl.float32)

    for k_start in range(0, S, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < S

        # Decode key spatial coords: k = s2_k * S_h + s1_k
        s1_k = k_offsets % S_h
        s2_k = k_offsets // S_h

        # Compute relative position index for each key
        # offset = s2_k - s2_q,  d_pos_idx = center + offset
        d_pos_idx = center + s2_k - s2_q  # [BLOCK_K]
        d_pos_idx_mask = (d_pos_idx >= 0) & (d_pos_idx < D_pos) & k_mask

        # Get matmul contribution for each key
        # matmul_result[d_pos_idx]  - need to gather from the computed vector
        matmul_contrib = tl.load(matmul_result + d_pos_idx, mask=d_pos_idx_mask, other=0.0)
        # Wait - matmul_result is in registers, not in memory. Can't use tl.load on it.
        # Need to index it differently.

        # Actually, since matmul_result is a Triton tensor in registers,
        # I can index it using tl.load from a "local" pointer? No, that doesn't work.
        # Let me use a different approach: compute the attention scores directly.

        # Load in_0[b, q, k_offsets]  - attention mask/base scores
        in_0_offsets = b * s0_b + q * s0_q + k_offsets * s0_k
        in_0_vals = tl.load(in_0_ptr + in_0_offsets, mask=k_mask, other=0.0)  # [BLOCK_K]

        # Load in_2[b, s2_q, s1, s2_k, s1_k]  - relative logits
        # in_2 shape: [B, S_h, S_h, S_h, S_h]
        # Indexing: (b, s2_q, s1, s2_k, s1_k)
        in_2_offsets = b * s2_b + s2_q * s2_d0 + s1 * s2_d1 + s2_k * s2_d2 + s1_k * s2_d3
        in_2_vals = tl.load(in_2_ptr + in_2_offsets, mask=k_mask, other=0.0)  # [BLOCK_K]

        # Compute attention scores
        # score = in_0 + matmul_contribution + in_2
        # Need to extract matmul_contribution from matmul_result

        # Since d_pos_idx varies per key position, I need to gather from matmul_result
        # matmul_result is a [D_POS_MAX] tensor. I can use indexing:
        # matmul_contrib = matmul_result[d_pos_idx]

        # In Triton, I can do: gather_result = tl.load(matmul_result_ptr + d_pos_idx)
        # But matmul_result is a register tensor, not a pointer.
        # Alternative: use tl.where to select the right value for each d_pos_idx

        # Since D_pos is small (31 or 15), I can compute the contribution per key
        # by iterating over possible d_pos_idx values

        # Actually, the simplest approach: for each key, compute the matmul result
        # for the specific d_pos_idx value. But d_pos_idx varies per key.

        # Let me reconsider. I'll compute the matmul contribution differently.
        # Instead of computing the full matmul_result first, I'll compute
        # the dot product for each key position's d_pos_idx on-the-fly.

        # But that means recomputing the dot product for each key tile,
        # which is wasteful since D_pos << S.

        # Alternative: store matmul_result in a small shared memory buffer
        # and gather from it. Triton doesn't have shared memory directly,
        # but we can use tl.load from a global memory buffer.

        # Or: since D_POS_MAX <= 32, I can do explicit indexing:
        # matmul_contrib = 0.0
        # for dp in range(D_POS_MAX):
        #     if d_pos_idx == dp: matmul_contrib += matmul_result[dp]
        # This is O(BLOCK_K * D_POS_MAX) comparisons, which might be slow.

        # Best approach: write matmul_result to a small global memory buffer,
        # then gather from it. But allocating a per-program buffer is complex.

        # Simpler approach: compute the matmul contribution per key position
        # by doing a single dot product (in_1[b,s1,s2_q,:] dot in_3[:, d_pos_idx])
        # for each key. Since d_pos_idx varies per key but is determined by
        # s2_k - s2_q, many keys share the same d_pos_idx (those with same s2_k).

        # Actually, the most practical approach: compute the full matmul_result
        # once and store it. Then for each key, extract the right element.
        # In Triton, I can do this by making matmul_result a local array
        # and using indirect indexing.

        # Triton supports: result = tensor[indices] where indices is another tensor
        # This should work for register tensors.

        # Let me try: matmul_contrib = matmul_result[d_pos_idx]
        # where matmul_result is [D_POS_MAX] and d_pos_idx is [BLOCK_K]

        # Hmm, Triton might not support gathering from register tensors with
        # dynamic indices. Let me check.

        # Actually, in Triton, you can't do tensor[dynamic_index] for register tensors.
        # You can only do tensor[const_expr_index] or tensor[tl.arange(...)].

        # So I need a different approach. Let me write matmul_result to global memory
        # (a small scratch buffer) and then gather from it.

        # OR: I can restructure the computation. Instead of computing the full
        # matmul_result first, I'll compute the attention scores directly.
        pass  # placeholder

    # Placeholder - will be replaced with actual implementation
    pass


# I realize the above approach has issues with gathering from register tensors.
# Let me redesign the kernel to avoid this problem.

# New approach: Compute everything inline without needing to gather from
# a pre-computed matmul_result vector.

# For each key position k, the attention score contribution from the first matmul is:
# matmul[b, s1, s2_q, center + s2_k - s2_q]
# = sum_d in_1[b, s1, s2_q, d] * in_3[d, center + s2_k - s2_q]

# Instead of computing the full matmul_result and then gathering,
# I can compute the dot product for each (query, key) pair directly.
# But this means recomputing the dot product for each key position,
# which is wasteful.

# Better approach: Since s2_k changes slowly (every S_h keys have the same s2_k),
# I can compute the matmul contribution for each s2_k value and then
# broadcast it to all s1_k values within that group.

# For the 256 case: S_h=16, so there are 16 distinct s2_k values.
# For each s2_k, compute: dot(in_1[b,s1,s2_q,:], in_3[:, center+s2_k-s2_q])
# This gives 16 scalar values, one per s2_k.

# Then for each key k = s2_k*S_h + s1_k, the contribution is the s2_k-th scalar.

# This is much more efficient than recomputing for all 256 keys.

# Implementation: compute 16 (or 8) scalar dot products for each s2_k,
# then gather them when computing attention scores.

# For Triton, I can compute the s2_k dot products using a small matrix:
# in_1[b, s1, s2_q, :] (128 elements) @ in_3[:, relevant_d_pos_idx] (128 x 16 elements)
# where relevant_d_pos_idx = center + s2_k - s2_q for s2_k in [0, S_h-1]

# This is a [1, 128] @ [128, S_h] = [1, S_h] matmul.
# S_h = 16 (or 8), which is small enough to hold in registers.

@triton.jit
def fused_botnet_attn_kernel_v2(
    # pointers
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr, in_4_ptr, out_ptr,
    # dimensions
    B, S, S_h, D_in, D_pos, center, dim,
    # strides for in_0  [B, S, S]
    s0_b, s0_q, s0_k,
    # strides for in_1  [B, S_h, S_h, D_in]
    s1_b, s1_si, s1_sq, s1_d,
    # strides for in_2  [B, S_h, S_h, S_h, S_h]
    s2_b, s2_d0, s2_d1, s2_d2, s2_d3,
    # strides for in_3  [D_in, D_pos]
    s3_d, s3_p,
    # strides for in_4  [B, S, dim]
    s4_b, s4_k, s4_d,
    # strides for out   [B, dim, S]
    so_b, so_d, so_q,
    # tile sizes
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DIN: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // S
    q = pid % S

    # Decode query spatial coords: q = s2_q * S_h + s1
    s1 = q % S_h
    s2_q = q // S_h

    # ------------------------------------------------
    # Step 1: Compute relative position matmul for all s2_k values
    # For each s2_k in [0, S_h-1], compute:
    #   rpos[s2_k] = in_1[b, s1, s2_q, :] @ in_3[:, center + s2_k - s2_q]
    # ------------------------------------------------
    # s2_k offsets for the relevant d_pos indices
    s2_k_all = tl.arange(0, S_h)  # [S_h] constexpr
    d_pos_idx_all = center + s2_k_all - s2_q  # [S_h]
    d_pos_idx_mask_all = (d_pos_idx_all >= 0) & (d_pos_idx_all < D_pos)

    rpos = tl.zeros([S_h], dtype=tl.float32)

    # Tile over D_in dimension
    for d_in_start in range(0, D_in, BLOCK_DIN):
        d_in_off = d_in_start + tl.arange(0, BLOCK_DIN)
        d_in_mask = d_in_off < D_in

        # Load query vector: in_1[b, s1, s2_q, d_in_off]
        q_off = b * s1_b + s1 * s1_si + s2_q * s1_sq + d_in_off * s1_d
        q_vec = tl.load(in_1_ptr + q_off, mask=d_in_mask, other=0.0)  # [BLOCK_DIN]

        # Load weight columns: in_3[d_in_off, d_pos_idx_all]
        w_off = d_in_off[:, None] * s3_d + d_pos_idx_all[None, :] * s3_p
        w_mask = d_in_mask[:, None] & d_pos_idx_mask_all[None, :]
        w_tile = tl.load(in_3_ptr + w_off, mask=w_mask, other=0.0)  # [BLOCK_DIN, S_h]

        # Accumulate
        rpos += tl.sum(q_vec[:, None] * w_tile, axis=0)  # [S_h]

    # ------------------------------------------------
    # Step 2: Flash attention - online softmax + matmul with in_4
    # ------------------------------------------------
    running_max = tl.full([BLOCK_D], float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_D], dtype=tl.float32)
    running_out = tl.zeros([BLOCK_D], dtype=tl.float32)

    d_out_off = tl.arange(0, BLOCK_D)

    for k_start in range(0, S, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_off < S

        # Decode key spatial coords
        s1_k = k_off % S_h
        s2_k = k_off // S_h

        # Gather rpos value for each key's s2_k
        # rpos[s2_k] - since s2_k is dynamic, we need to handle this carefully
        # s2_k is in [0, S_h-1] and rpos is a [S_h] register tensor
        # We CAN do register tensor indexing with tl.arange-based offsets
        # but NOT with dynamic indices.
        
        # Since s2_k = k_off // S_h, and k_off varies, s2_k is dynamic.
        # We need a workaround.

        # Approach: compute rpos_k by iterating over possible s2_k values
        # and selecting the right one using tl.where.
        # This is O(S_h) comparisons per key position, but S_h is small (16 or 8).

        # Better approach: since within a BLOCK_K tile, many keys share the same s2_k,
        # we can group keys by s2_k. But this complicates the kernel.

        # Alternative: Use tl.load to gather from a small buffer in global memory.
        # We can write rpos to a per-program scratch space and gather from it.

        # Actually, the simplest approach that works in Triton:
        # For each key position, compute the matmul contribution by doing
        # a direct dot product for that specific d_pos_idx.
        # Since we process BLOCK_K keys at a time, this means BLOCK_K dot products.
        # But each dot product is only 128 elements, and we tile over D_in.

        # However, this approach recomputes the dot product for keys that share
        # the same s2_k. For S_h=16 and BLOCK_K=64, there are at most 4 distinct
        # s2_k values in a tile (64/16=4), but we'd compute 64 dot products.

        # This is wasteful. Let me think of another way.

        # Key insight: in Triton, we can use tl.load to read from a pointer
        # that we computed. If we store rpos in a tensor that we can index
        # with dynamic indices, we can gather efficiently.

        # The problem is that rpos is in registers, not in global memory.
        # We could write it to global memory (a scratch buffer), then gather.

        # But allocating per-program scratch buffers is complicated.

        # Alternative: use a GROUPED approach. Process keys in groups of S_h
        # (same s2_k), computing one dot product per group and broadcasting.

        # This changes the loop structure:
        # for s2_k in [0, S_h-1]:
        #   compute rpos[s2_k] (already done in Step 1)
        #   for s1_k in [0, S_h-1]:
        #     compute attention score for key = s2_k * S_h + s1_k
        #     accumulate for softmax

        # This approach processes keys in a specific order (grouped by s2_k),
        # which might affect the softmax computation (but softmax is order-independent
        # for the final result).

        # For flash attention, the online softmax algorithm works with any order
        # of processing keys. So this grouped approach is fine!

        # Let me redesign the kernel to use this grouped approach.
        pass

    pass


# Let me redesign the entire kernel with the grouped approach.
# Process keys grouped by s2_k, which allows efficient reuse of rpos values.

@triton.jit
def fused_botnet_attn_kernel_v3(
    # pointers
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr, in_4_ptr, out_ptr,
    # dimensions
    B, S, S_h, D_in, center, dim,
    # strides for in_0  [B, S, S]
    s0_b, s0_q, s0_k,
    # strides for in_1  [B, S_h, S_h, D_in]
    s1_b, s1_si, s1_sq, s1_d,
    # strides for in_2  [B, S_h, S_h, S_h, S_h]
    s2_b, s2_d0, s2_d1, s2_d2, s2_d3,
    # strides for in_3  [D_in, D_pos]
    s3_d, s3_p,
    # strides for in_4  [B, S, dim]
    s4_b, s4_k, s4_d,
    # strides for out   [B, dim, S]
    so_b, so_d, so_q,
    # tile sizes
    BLOCK_S1K: tl.constexpr,  # tile for s1_k dimension within a s2_k group
    BLOCK_D: tl.constexpr,    # tile for value dimension
    BLOCK_DIN: tl.constexpr,  # tile for D_in in the first matmul
):
    pid = tl.program_id(0)
    b = pid // S
    q = pid % S

    # Decode query spatial coords
    s1 = q % S_h
    s2_q = q // S_h

    # ------------------------------------------------
    # Step 1: Compute rpos for all s2_k values
    # rpos[s2_k] = in_1[b, s1, s2_q, :] @ in_3[:, center + s2_k - s2_q]
    # ------------------------------------------------
    s2_k_all = tl.arange(0, S_h)
    d_pos_idx_all = center + s2_k_all - s2_q
    d_pos_valid = (d_pos_idx_all >= 0)

    rpos = tl.zeros([S_h], dtype=tl.float32)

    for d_in_start in range(0, D_in, BLOCK_DIN):
        d_in_off = d_in_start + tl.arange(0, BLOCK_DIN)
        d_in_mask = d_in_off < D_in

        q_off = b * s1_b + s1 * s1_si + s2_q * s1_sq + d_in_off * s1_d
        q_vec = tl.load(in_1_ptr + q_off, mask=d_in_mask, other=0.0)

        w_off = d_in_off[:, None] * s3_d + d_pos_idx_all[None, :] * s3_p
        w_mask = d_in_mask[:, None] & d_pos_valid[None, :]
        w_tile = tl.load(in_3_ptr + w_off, mask=w_mask, other=0.0)

        rpos += tl.sum(q_vec[:, None] * w_tile, axis=0)

    # ------------------------------------------------
    # Step 2: Flash attention - grouped by s2_k
    # ------------------------------------------------
    # Online softmax: track running max, running sum, running output
    running_max = tl.full([BLOCK_D], float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_D], dtype=tl.float32)
    running_out = tl.zeros([BLOCK_D], dtype=tl.float32)

    d_out_off = tl.arange(0, BLOCK_D)
    d_out_mask = d_out_off < dim

    # Process each s2_k group
    for s2_k_idx in range(S_h):
        # rpos value for this s2_k
        rpos_val = rpos[s2_k_idx]  # scalar - constexpr index works!

        # Process all s1_k values within this s2_k group
        # Key index k = s2_k_idx * S_h + s1_k
        for s1_k_start in range(0, S_h, BLOCK_S1K):
            s1_k_off = s1_k_start + tl.arange(0, BLOCK_S1K)
            s1_k_mask = s1_k_off < S_h

            k_off = s2_k_idx * S_h + s1_k_off  # [BLOCK_S1K]
            k_valid = s1_k_mask

            # Load in_0[b, q, k_off] - attention mask
            in_0_off = b * s0_b + q * s0_q + k_off * s0_k
            attn_base = tl.load(in_0_ptr + in_0_off, mask=k_valid, other=0.0)

            # Load in_2[b, s2_q, s1, s2_k_idx, s1_k_off]
            in_2_off = b * s2_b + s2_q * s2_d0 + s1 * s2_d1 + s2_k_idx * s2_d2 + s1_k_off * s2_d3
            rel_logits = tl.load(in_2_ptr + in_2_off, mask=k_valid, other=0.0)

            # Compute attention score
            score = attn_base + rpos_val + rel_logits  # [BLOCK_S1K]

            # Online softmax: compute new max
            score_max = tl.max(score, axis=0)  # scalar
            new_max = tl.maximum(running_max, score_max)

            # Rescale running values
            rescale = tl.exp(running_max - new_max)
            running_sum *= rescale
            running_out *= rescale[:, None] if False else running_out * rescale  # [BLOCK_D] * scalar

            # Compute exp of current scores
            exp_score = tl.exp(score - new_max)  # [BLOCK_S1K]
            exp_sum = tl.sum(exp_score, axis=0)  # scalar

            # Load V[b, k_off, d_out_off] - value matrix
            # in_4 shape: [B, S, dim]
            v_off = b * s4_b + k_off[:, None] * s4_k + d_out_off[None, :] * s4_d
            v_mask = k_valid[:, None] & d_out_mask[None, :]
            v_tile = tl.load(in_4_ptr + v_off, mask=v_mask, other=0.0)  # [BLOCK_S1K, BLOCK_D]

            # Accumulate: softmax_weight * V
            running_out += tl.sum(exp_score[:, None] * v_tile, axis=0)  # [BLOCK_D]
            running_sum += exp_sum
            running_max = new_max

    # ------------------------------------------------
    # Step 3: Normalize and write output
    # ------------------------------------------------
    # out[b, d, q] = running_out / running_sum
    out_vals = running_out / running_sum  # [BLOCK_D]
    out_off = b * so_b + d_out_off * so_d + q * so_q
    tl.store(out_ptr + out_off, out_vals, mask=d_out_mask)


# Kernel wrapper with routing
@torch.fx.wrap
def fused_botnet_attention_dispatch(in_0, in_1, in_2, in_3, in_4, route):
    B = in_0.shape[0]
    dim = in_4.shape[2]

    if route == "route_256":
        S = 256
        S_h = 16
        D_in = 128
        D_pos = 31
        center = 15
    elif route == "route_64":
        S = 64
        S_h = 8
        D_in = 128
        D_pos = 15
        center = 7
    else:
        raise ValueError(f"Unknown route: {route}")

    # Allocate output: [B, dim, S]
    out = torch.empty((B, dim, S), dtype=in_0.dtype, device=in_0.device)

    # Strides
    s0 = in_0.stride()
    s1 = in_1.stride()
    s2 = in_2.stride()
    s3 = in_3.stride()
    s4 = in_4.stride()
    so = out.stride()

    # Choose block sizes based on route
    if route == "route_256":
        BLOCK_S1K = 16  # S_h = 16, so one tile covers all s1_k
        BLOCK_D = 32
        BLOCK_DIN = 32
    else:  # route_64
        BLOCK_S1K = 8   # S_h = 8
        BLOCK_D = 32
        BLOCK_DIN = 32

    num_programs = B * S

    fused_botnet_attn_kernel_v3[(num_programs,)](
        in_0_ptr=in_0, in_1_ptr=in_1, in_2_ptr=in_2, in_3_ptr=in_3, in_4_ptr=in_4, out_ptr=out,
        B=B, S=S, S_h=S_h, D_in=D_in, center=center, dim=dim,
        s0_b=s0[0], s0_q=s0[1], s0_k=s0[2],
        s1_b=s1[0], s1_si=s1[1], s1_sq=s1[2], s1_d=s1[3],
        s2_b=s2[0], s2_d0=s2[1], s2_d1=s2[2], s2_d2=s2[3], s2_d3=s2[4],
        s3_d=s3[0], s3_p=s3[1],
        s4_b=s4[0], s4_k=s4[1], s4_d=s4[2],
        so_b=so[0], so_d=so[1], so_q=so[2],
        BLOCK_S1K=BLOCK_S1K,
        BLOCK_D=BLOCK_D,
        BLOCK_DIN=BLOCK_DIN,
    )

    return out


def replacement_func():
    return fused_botnet_attention_dispatch