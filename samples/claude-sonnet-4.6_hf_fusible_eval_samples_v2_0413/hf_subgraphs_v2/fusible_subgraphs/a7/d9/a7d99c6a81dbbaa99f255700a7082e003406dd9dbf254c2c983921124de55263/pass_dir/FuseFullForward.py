import torch
import triton
import triton.language as tl


# ── Fused kernel: GEMV (value) + key copy ─────────────────────────────────────
# 512 programs, one per output feature:
#   - Each program computes value[pid] via GEMV (dot(hidden, weight[pid,:]) + bias[pid])
#   - AND copies one element of key_states (flat sequential access)
@triton.jit
def gemv_plus_copy_kernel(
    hidden_ptr, weight_ptr, bias_ptr,
    key_ptr,
    out_val_ptr, out_key_ptr,
    K: tl.constexpr,
):
    pid = tl.program_id(0)

    # Copy key state element (flat sequential, valid for seq_len=1 tensors)
    tl.store(out_key_ptr + pid, tl.load(key_ptr + pid))

    # GEMV for value state
    k_offs = tl.arange(0, K)
    x = tl.load(hidden_ptr + k_offs).to(tl.float32)
    w = tl.load(weight_ptr + pid * K + k_offs).to(tl.float32)
    b = tl.load(bias_ptr + pid).to(tl.float32)
    tl.store(out_val_ptr + pid, tl.sum(x * w, axis=0) + b)


@torch.fx.wrap
def fused_key_value(bias, weight, hidden_states, key_states):
    """
    Replaces both the key path (in_4→view→transpose→contiguous) AND
    the value path (linear→view→transpose→contiguous) with a single
    @torch.fx.wrap call.  One graph-break instead of two.

    Returns: (out_key [1,8,1,64], out_val [1,8,1,64])
             matching pattern output order (tmp_9, tmp_10).
    """
    K = 512
    weight = weight.to(device=hidden_states.device)
    bias = bias.to(device=hidden_states.device)

    out_val = torch.empty(1, 8, 1, 64, dtype=hidden_states.dtype, device=hidden_states.device)
    out_key = torch.empty(1, 8, 1, 64, dtype=key_states.dtype, device=key_states.device)

    gemv_plus_copy_kernel[(K,)](
        hidden_states, weight, bias, key_states,
        out_val, out_key,
        K=K,
    )

    return out_key, out_val


# ── Pass interface ────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_3, in_4):
    """
    in_0 = bias   [512]
    in_1 = weight [512, 512]
    in_3 = hidden_states [1, 1, 512]
    in_4 = key_states    [1, 1, 512]
    """
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    # Key path
    tmp_3 = in_4.view(1, 1, -1, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_9 = tmp_4.contiguous()
    # Value path
    tmp_5 = linear.view(1, 1, -1, 64)
    tmp_6 = tmp_5.transpose(1, 2)
    tmp_10 = tmp_6.contiguous()
    return tmp_9, tmp_10


def replacement_args(in_0, in_1, in_3, in_4):
    # bias, weight, hidden_states, key_states
    return (in_0, in_1, in_3, in_4)


def replacement_func():
    return fused_key_value