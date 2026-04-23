import torch
import triton
import triton.language as tl

NUM_HEADS = 9
HEAD_DIM = 48


def pattern(in_0, in_1):
    linear = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2 = linear.reshape(1, 197, 3, 9, 48)
    tmp_3 = tmp_2.permute(2, 0, 3, 1, 4)
    unbind = tmp_3.unbind(0)
    tmp_5 = unbind[0]
    tmp_6 = unbind[1]
    tmp_7 = unbind[2]
    tmp_8 = tmp_6.transpose(-2, -1)
    return (tmp_5, tmp_8, tmp_7)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_qkv_kernel(
    # Pointers
    input_ptr, weight_ptr, q_ptr, kt_ptr, v_ptr,
    # Dimensions
    seq_len, hidden_dim, num_heads: tl.constexpr, head_dim: tl.constexpr,
    total_heads: tl.constexpr,  # 3 * num_heads
    # Strides for input (batch, seq, hidden)
    in_s_stride, in_h_stride,
    # Strides for weight (out, in)
    w_o_stride, w_i_stride,
    # Strides for output Q/V (batch, heads, seq, head_dim)
    q_h_stride, q_s_stride, q_d_stride,
    # Strides for Kt (batch, heads, head_dim, seq)
    kt_h_stride, kt_d_stride, kt_s_stride,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # pid_m: which block of seq positions
    # pid_h: which head of which QKV component (0 to 3*num_heads-1)
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < seq_len

    # Determine which QKV component and which head
    qkv_idx = pid_h // num_heads  # 0=Q, 1=K, 2=V
    head_idx = pid_h % num_heads

    # Starting row in weight matrix for this head of this QKV component
    n_start = pid_h * head_dim
    d_offsets = tl.arange(0, head_dim)

    # Accumulators: [BLOCK_M, head_dim]
    acc = tl.zeros((BLOCK_M, head_dim), dtype=tl.float32)

    # Iterate over input dimension (K dimension of matmul)
    for k_start in range(0, hidden_dim, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load input tile: [BLOCK_M, BLOCK_K]
        input_vals = tl.load(input_ptr + m_offsets[:, None] * in_s_stride + k_offsets[None, :] * in_h_stride,
                             mask=m_mask[:, None] & (k_offsets[None, :] < hidden_dim), other=0.0)

        # Load weight tile: [head_dim, BLOCK_K]
        w_row_offsets = n_start + d_offsets[:, None]
        weight_vals = tl.load(weight_ptr + w_row_offsets * w_o_stride + k_offsets[None, :] * w_i_stride,
                              mask=(d_offsets[:, None] < head_dim) & (k_offsets[None, :] < hidden_dim), other=0.0)

        # acc += input @ weight.T: [BLOCK_M, BLOCK_K] @ [BLOCK_K, head_dim]
        acc += tl.dot(input_vals, tl.trans(weight_vals), allow_tf32=False)

    # Convert accumulator to output dtype
    out_vals = acc  # Keep in float32 for precision, will convert on store

    # Store results based on QKV component
    if qkv_idx == 0:  # Q: (batch, head_idx, seq, head_dim)
        out_mask = m_mask[:, None] & (d_offsets[None, :] < head_dim)
        base_offset = head_idx * q_h_stride
        tl.store(q_ptr + base_offset + m_offsets[:, None] * q_s_stride + d_offsets[None, :] * q_d_stride,
                 out_vals, mask=out_mask)

    elif qkv_idx == 1:  # Kt: (batch, head_idx, head_dim, seq) - transposed storage
        # Store transposed: acc[m, d] -> kt[head_idx, d, m]
        out_mask = m_mask[:, None] & (d_offsets[None, :] < head_dim)
        base_offset = head_idx * kt_h_stride
        tl.store(kt_ptr + base_offset + d_offsets[None, :] * kt_d_stride + m_offsets[:, None] * kt_s_stride,
                 out_vals, mask=out_mask)

    else:  # V: (batch, head_idx, seq, head_dim)
        out_mask = m_mask[:, None] & (d_offsets[None, :] < head_dim)
        base_offset = head_idx * q_h_stride  # V has same layout as Q
        tl.store(v_ptr + base_offset + m_offsets[:, None] * q_s_stride + d_offsets[None, :] * q_d_stride,
                 out_vals, mask=out_mask)


@torch.fx.wrap
def fused_qkv_9(weight, input):
    batch = input.shape[0]
    seq_len = input.shape[1]
    hidden_dim = input.shape[2]

    num_heads = NUM_HEADS
    head_dim = HEAD_DIM
    total_heads = 3 * num_heads  # 27

    dtype = input.dtype

    # Output shapes
    # Q: (batch, num_heads, seq_len, head_dim)
    q = torch.empty((batch, num_heads, seq_len, head_dim), dtype=dtype, device=input.device)
    # Kt: (batch, num_heads, head_dim, seq_len) - K transposed
    kt = torch.empty((batch, num_heads, head_dim, seq_len), dtype=dtype, device=input.device)
    # V: (batch, num_heads, seq_len, head_dim)
    v = torch.empty((batch, num_heads, seq_len, head_dim), dtype=dtype, device=input.device)

    BLOCK_M = 16
    BLOCK_K = 64

    grid = (
        (seq_len + BLOCK_M - 1) // BLOCK_M,  # pid_m: over seq positions
        total_heads,  # pid_h: one for each head of each QKV component
    )

    fused_qkv_kernel[grid](
        input_ptr=input, weight_ptr=weight,
        q_ptr=q, kt_ptr=kt, v_ptr=v,
        seq_len=seq_len, hidden_dim=hidden_dim,
        num_heads=num_heads, head_dim=head_dim,
        total_heads=total_heads,
        in_s_stride=input.stride(1), in_h_stride=input.stride(2),
        w_o_stride=weight.stride(0), w_i_stride=weight.stride(1),
        q_h_stride=q.stride(1), q_s_stride=q.stride(2), q_d_stride=q.stride(3),
        kt_h_stride=kt.stride(1), kt_d_stride=kt.stride(2), kt_s_stride=kt.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
    )

    return (q, kt, v)


def replacement_func():
    return fused_qkv_9