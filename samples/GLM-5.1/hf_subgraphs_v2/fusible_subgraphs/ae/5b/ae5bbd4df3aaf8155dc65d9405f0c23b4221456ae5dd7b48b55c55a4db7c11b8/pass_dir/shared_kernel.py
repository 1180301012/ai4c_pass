import torch
import triton
import triton.language as tl


@triton.jit
def fused_linear_gru_kernel(
    in_3_ptr, in_1_ptr, in_0_ptr, in_2_ptr, out_ptr,
    n_heads, n_seq, K,
    stride_in3_batch, stride_in3_head, stride_in3_seq, stride_in3_k,
    stride_in1_out, stride_in1_k,
    stride_in2_head,
    BLOCK_K: tl.constexpr,
):
    """Fused linear + GRU gating kernel.
    
    Each program handles one (head, seq) output position.
    Computes linear + view + sum + sigmoid + GRU gating entirely in registers.
    Uses scalar accumulators for the 8 output channels to avoid tensor indexing issues.
    """
    pid_h = tl.program_id(0)
    pid_s = tl.program_id(1)

    # Load bias in_0 [8] as separate scalars
    b0 = tl.load(in_0_ptr + 0).to(tl.float32)
    b1 = tl.load(in_0_ptr + 1).to(tl.float32)
    b2 = tl.load(in_0_ptr + 2).to(tl.float32)
    b3 = tl.load(in_0_ptr + 3).to(tl.float32)
    b4 = tl.load(in_0_ptr + 4).to(tl.float32)
    b5 = tl.load(in_0_ptr + 5).to(tl.float32)
    b6 = tl.load(in_0_ptr + 6).to(tl.float32)
    b7 = tl.load(in_0_ptr + 7).to(tl.float32)

    # Load in_2 for this head
    in_2_val = tl.load(in_2_ptr + pid_h * stride_in2_head).to(tl.float32)

    # Scalar accumulators for 8 output channels
    a0 = 0.0
    a1 = 0.0
    a2 = 0.0
    a3 = 0.0
    a4 = 0.0
    a5 = 0.0
    a6 = 0.0
    a7 = 0.0
    
    in3_base = pid_h * stride_in3_head + pid_s * stride_in3_seq
    
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        
        # Load input slice [BLOCK_K]
        in3_vals = tl.load(in_3_ptr + in3_base + k_offsets * stride_in3_k, mask=k_mask, other=0.0).to(tl.float32)
        
        # Load weight slices for each output channel and accumulate dot product
        w0 = tl.load(in_1_ptr + 0 * stride_in1_out + k_offsets * stride_in1_k, mask=k_mask, other=0.0).to(tl.float32)
        w1 = tl.load(in_1_ptr + 1 * stride_in1_out + k_offsets * stride_in1_k, mask=k_mask, other=0.0).to(tl.float32)
        w2 = tl.load(in_1_ptr + 2 * stride_in1_out + k_offsets * stride_in1_k, mask=k_mask, other=0.0).to(tl.float32)
        w3 = tl.load(in_1_ptr + 3 * stride_in1_out + k_offsets * stride_in1_k, mask=k_mask, other=0.0).to(tl.float32)
        w4 = tl.load(in_1_ptr + 4 * stride_in1_out + k_offsets * stride_in1_k, mask=k_mask, other=0.0).to(tl.float32)
        w5 = tl.load(in_1_ptr + 5 * stride_in1_out + k_offsets * stride_in1_k, mask=k_mask, other=0.0).to(tl.float32)
        w6 = tl.load(in_1_ptr + 6 * stride_in1_out + k_offsets * stride_in1_k, mask=k_mask, other=0.0).to(tl.float32)
        w7 = tl.load(in_1_ptr + 7 * stride_in1_out + k_offsets * stride_in1_k, mask=k_mask, other=0.0).to(tl.float32)
        
        a0 += tl.sum(in3_vals * w0)
        a1 += tl.sum(in3_vals * w1)
        a2 += tl.sum(in3_vals * w2)
        a3 += tl.sum(in3_vals * w3)
        a4 += tl.sum(in3_vals * w4)
        a5 += tl.sum(in3_vals * w5)
        a6 += tl.sum(in3_vals * w6)
        a7 += tl.sum(in3_vals * w7)
    
    # Add bias
    l0 = a0 + b0
    l1 = a1 + b1
    l2 = a2 + b2
    l3 = a3 + b3
    l4 = a4 + b4
    l5 = a5 + b5
    l6 = a6 + b6
    l7 = a7 + b7

    # Sum halves
    sum_first = l0 + l1 + l2 + l3
    sum_second = l4 + l5 + l6 + l7

    # Sigmoid
    sig_first = tl.sigmoid(sum_first)
    sig_second = tl.sigmoid(sum_second)

    # GRU gating: sig_first * (sig_second * in_2 - 1.0) + 2.0
    result = sig_first * (sig_second * in_2_val - 1.0) + 2.0

    # Store result
    out_idx = pid_h * n_seq + pid_s
    tl.store(out_ptr + out_idx, result)


@torch.fx.wrap
def fused_linear_gru_dispatch(in_0, in_1, in_2, in_3, route):
    n_heads = in_3.shape[1]
    n_seq = in_3.shape[2]
    K = in_3.shape[3]

    out = torch.empty((1, n_heads, n_seq, 1), dtype=in_3.dtype, device=in_3.device)

    stride_in3 = in_3.stride()
    stride_in1 = in_1.stride()
    stride_in2 = in_2.stride()

    BLOCK_K = 16

    if route == "route_12" or route == "route_16":
        fused_linear_gru_kernel[(n_heads, n_seq)](
            in_3_ptr=in_3,
            in_1_ptr=in_1,
            in_0_ptr=in_0,
            in_2_ptr=in_2,
            out_ptr=out,
            n_heads=n_heads,
            n_seq=n_seq,
            K=K,
            stride_in3_batch=stride_in3[0],
            stride_in3_head=stride_in3[1],
            stride_in3_seq=stride_in3[2],
            stride_in3_k=stride_in3[3],
            stride_in1_out=stride_in1[0],
            stride_in1_k=stride_in1[1],
            stride_in2_head=stride_in2[1],
            BLOCK_K=BLOCK_K,
        )
    else:
        raise ValueError(f"Unknown route: {route}")

    return out