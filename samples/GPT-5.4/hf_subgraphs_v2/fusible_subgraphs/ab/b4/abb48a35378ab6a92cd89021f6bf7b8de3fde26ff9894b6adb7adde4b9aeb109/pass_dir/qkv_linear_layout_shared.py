import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _qkv_gemm_split_store_kernel(
    a_ptr,
    b_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    M,
    N,
    K,
    TOKENS,
    stride_ab,
    stride_at,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_qb,
    stride_qh,
    stride_qt,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vt,
    stride_vd,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    batch_ids = offs_m // TOKENS
    token_ids = offs_m % TOKENS
    a_ptrs = a_ptr + batch_ids[:, None] * stride_ab + token_ids[:, None] * stride_at + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k0 * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[:, None] < N) & (offs_k[None, :] < k_remaining), other=0.0)
        acc = tl.dot(a, tl.trans(b), acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_n_b = offs_n[None, :]
    head_span = NUM_HEADS * HEAD_DIM
    valid_m = offs_m[:, None] < M
    q_mask = valid_m & (offs_n_b < head_span)
    k_mask = valid_m & (offs_n_b >= head_span) & (offs_n_b < 2 * head_span)
    v_mask = valid_m & (offs_n_b >= 2 * head_span) & (offs_n_b < 3 * head_span)

    q_head = offs_n_b // HEAD_DIM
    q_d = offs_n_b % HEAD_DIM
    q_ptrs = q_ptr + batch_ids[:, None] * stride_qb + q_head * stride_qh + token_ids[:, None] * stride_qt + q_d * stride_qd
    tl.store(q_ptrs, acc, mask=q_mask)

    k_col = offs_n_b - head_span
    k_head = k_col // HEAD_DIM
    k_d = k_col % HEAD_DIM
    k_ptrs = k_ptr + batch_ids[:, None] * stride_kb + k_head * stride_kh + token_ids[:, None] * stride_kt + k_d * stride_kd
    tl.store(k_ptrs, acc, mask=k_mask)

    v_col = offs_n_b - 2 * head_span
    v_head = v_col // HEAD_DIM
    v_d = v_col % HEAD_DIM
    v_ptrs = v_ptr + batch_ids[:, None] * stride_vb + v_head * stride_vh + token_ids[:, None] * stride_vt + v_d * stride_vd
    tl.store(v_ptrs, acc, mask=v_mask)


@torch.fx.wrap
def qkv_linear_layout_dispatch(in_0, in_1, route):
    num_heads = 0
    if route == "h4":
        num_heads = 4
    elif route == "h9":
        num_heads = 9
    elif route == "h16":
        num_heads = 16
    else:
        raise RuntimeError("unknown route")

    head_dim = 48
    batch = in_1.shape[0]
    tokens = in_1.shape[1]
    embed = in_1.shape[2]
    out_features = in_0.shape[0]
    expected_out = 3 * num_heads * head_dim
    if out_features != expected_out:
        raise RuntimeError(f"unexpected weight shape: got {out_features}, expect {expected_out}")

    m = batch * tokens
    n = out_features
    k = embed

    q = torch.empty((batch, num_heads, tokens, head_dim), device=in_1.device, dtype=in_1.dtype)
    k_out = torch.empty((batch, num_heads, tokens, head_dim), device=in_1.device, dtype=in_1.dtype)
    v = torch.empty((batch, num_heads, tokens, head_dim), device=in_1.device, dtype=in_1.dtype)

    def grid(meta):
        return (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)

    _qkv_gemm_split_store_kernel[grid](
        in_1,
        in_0,
        q,
        k_out,
        v,
        m,
        n,
        k,
        tokens,
        in_1.stride(0),
        in_1.stride(1),
        in_1.stride(2),
        in_0.stride(0),
        in_0.stride(1),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k_out.stride(0),
        k_out.stride(1),
        k_out.stride(2),
        k_out.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
    )
    return q, k_out, v


def replacement_func():
    return qkv_linear_layout_dispatch