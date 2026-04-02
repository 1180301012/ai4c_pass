import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Simplified approach: fuse ONLY the element-wise tail (5 ops, no reductions).
#
# Pattern matched:
#   tmp_11 = tmp_9.sigmoid()          tmp_9  = LN(linear_out)
#   tmp_10 = in_9.sigmoid()           in_9   = raw input_gate
#   tmp_14 = tmp_12.unsqueeze(-2)     tmp_12 = LN(in_11) [B,D]→[B,1,D] view
#   tmp_15 = tmp_11 * tmp_14
#   tmp_16 = tmp_10 * tmp_13          tmp_13 = LN(in_10) [B,1,D]
#   output = tmp_15 + tmp_16
#
# All four inputs have 300×256 = 76 800 elements with identical flat strides
# ([B,1,D] and [B,D] both map flat index k → batch k//D, feat k%D).
#
# Kernel:  no reductions → pure memory-bandwidth-bound → very fast.
# Grid:    (B,) = (300,)   one program per batch row.
# Threads: num_warps=1 (32 threads, 8 elements/thread) for minimal overhead.
# ---------------------------------------------------------------------------

@triton.jit
def sigmoid_gate_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, out_ptr,
    D: tl.constexpr,
):
    batch_id = tl.program_id(0)
    d_range  = tl.arange(0, D)
    off      = batch_id * D

    a = tl.load(a_ptr + off + d_range)
    b = tl.load(b_ptr + off + d_range)
    c = tl.load(c_ptr + off + d_range)
    d = tl.load(d_ptr + off + d_range)

    a32 = a.to(tl.float32)
    b32 = b.to(tl.float32)
    result = tl.sigmoid(a32) * c.to(tl.float32) + tl.sigmoid(b32) * d.to(tl.float32)

    tl.store(out_ptr + off + d_range, result)


@torch.fx.wrap
def sigmoid_gate_fused(tmp_9, in_9, tmp_12, tmp_13):
    B   = tmp_9.shape[0]   # 300
    D   = tmp_9.shape[-1]  # 256
    out = torch.empty_like(tmp_9)
    sigmoid_gate_kernel[(B,)](
        tmp_9, in_9, tmp_12, tmp_13, out,
        D=D,
        num_warps=1,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern: the 6-op element-wise tail that follows the 3 layer-norms.
#
#   tmp_9,  tmp_12, tmp_13 are outputs of layer_norm (not pattern inputs).
#   in_9 is a raw graph input.
# ---------------------------------------------------------------------------

def pattern(tmp_9, in_9, tmp_12, tmp_13):
    tmp_11 = tmp_9.sigmoid()
    tmp_10 = in_9.sigmoid()
    tmp_14 = tmp_12.unsqueeze(-2)
    tmp_15 = tmp_11 * tmp_14
    tmp_16 = tmp_10 * tmp_13
    tmp_17 = tmp_15 + tmp_16
    return tmp_17


def replacement_args(tmp_9, in_9, tmp_12, tmp_13):
    return (tmp_9, in_9, tmp_12, tmp_13)


def replacement_func():
    return sigmoid_gate_fused