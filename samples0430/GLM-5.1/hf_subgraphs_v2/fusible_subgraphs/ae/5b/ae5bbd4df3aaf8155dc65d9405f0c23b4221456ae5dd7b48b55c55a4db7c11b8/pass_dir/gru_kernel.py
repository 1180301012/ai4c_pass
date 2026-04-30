import torch
import triton
import triton.language as tl


@triton.jit
def fused_gru_gate_kernel(
    input_ptr, weight_ptr, bias_ptr, const_ptr, out_ptr,
    K: tl.constexpr,
    stride_input_h, stride_input_t, stride_input_k,
    stride_weight_j, stride_weight_k,
    stride_const_h,
    stride_out_h, stride_out_t,
    BLOCK_K: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)

    k_offsets = tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    # Load input[0, h, t, :] and convert to float32
    input_vals = tl.load(
        input_ptr + pid_h * stride_input_h + pid_t * stride_input_t + k_offsets * stride_input_k,
        mask=k_mask, other=0.0
    ).to(tl.float32)

    # Compute weight sums for gate 0 (rows 0-3) and gate 1 (rows 4-7)
    w_sum0 = tl.zeros([BLOCK_K], dtype=tl.float32)
    w_sum1 = tl.zeros([BLOCK_K], dtype=tl.float32)

    for j in range(4):
        w = tl.load(
            weight_ptr + j * stride_weight_j + k_offsets * stride_weight_k,
            mask=k_mask, other=0.0
        ).to(tl.float32)
        w_sum0 += w
    for j in range(4, 8):
        w = tl.load(
            weight_ptr + j * stride_weight_j + k_offsets * stride_weight_k,
            mask=k_mask, other=0.0
        ).to(tl.float32)
        w_sum1 += w

    # Dot products (sum over K dimension)
    dot0 = tl.sum(input_vals * w_sum0)
    dot1 = tl.sum(input_vals * w_sum1)

    # Add bias sums
    b0 = (tl.load(bias_ptr + 0).to(tl.float32) +
          tl.load(bias_ptr + 1).to(tl.float32) +
          tl.load(bias_ptr + 2).to(tl.float32) +
          tl.load(bias_ptr + 3).to(tl.float32))
    b1 = (tl.load(bias_ptr + 4).to(tl.float32) +
          tl.load(bias_ptr + 5).to(tl.float32) +
          tl.load(bias_ptr + 6).to(tl.float32) +
          tl.load(bias_ptr + 7).to(tl.float32))

    dot0 += b0
    dot1 += b1

    # Sigmoid
    gate0 = tl.sigmoid(dot0)
    gate1 = tl.sigmoid(dot1)

    # Load const value for this head: in_2 is [1, H, 1, 1]
    const_val = tl.load(const_ptr + pid_h * stride_const_h).to(tl.float32)

    # Compute result: gate0 * (gate1 * const - 1) + 2
    result = gate0 * (gate1 * const_val - 1.0) + 2.0

    # Store result
    tl.store(out_ptr + pid_h * stride_out_h + pid_t * stride_out_t, result)


@torch.fx.wrap
def gru_gate_dispatch(bias, weight, const, input, route):
    if route == "12heads":
        H = 12
    elif route == "16heads":
        H = 16
    else:
        raise ValueError(f"Unknown route: {route}")

    T = 199
    K = 64
    BLOCK_K = 64

    out = torch.empty(1, H, T, 1, dtype=input.dtype, device=input.device)

    fused_gru_gate_kernel[(H, T)](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        const_ptr=const,
        out_ptr=out,
        K=K,
        stride_input_h=input.stride(1),
        stride_input_t=input.stride(2),
        stride_input_k=input.stride(3),
        stride_weight_j=weight.stride(0),
        stride_weight_k=weight.stride(1),
        stride_const_h=const.stride(1),
        stride_out_h=out.stride(1),
        stride_out_t=out.stride(2),
        BLOCK_K=BLOCK_K,
        num_warps=2,
    )

    return out