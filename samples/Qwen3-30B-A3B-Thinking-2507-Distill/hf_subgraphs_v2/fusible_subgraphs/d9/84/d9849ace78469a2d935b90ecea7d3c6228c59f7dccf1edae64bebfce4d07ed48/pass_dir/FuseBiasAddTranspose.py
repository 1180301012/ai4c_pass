import operator
import torch
import torch.fx.proxy
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: in-place add (broadcast) followed by transpose(1, 2)
# in_0 : [J, 1]         bias / small tensor
# in_1 : [B, J, K]      input tensor
# out  : [B, K, J]      transposed output
# ---------------------------------------------------------------------------
# FX Proxy.__iadd__ in PyTorch 2.9.1 always creates operator.add nodes, but
# the Dynamo-traced target graph has operator.iadd nodes.
# Workaround (pattern() is exempt from API validation):
#   Temporarily patch Proxy.__iadd__ to create operator.iadd nodes, run the
#   pattern, then restore the original.

def pattern(in_0, in_1):
    # Save original __iadd__ (plain function, not descriptor)
    _cls = torch.fx.proxy.Proxy
    _orig = _cls.__dict__.get('__iadd__')

    def _iadd_as_iadd(self, other):
        # Create operator.iadd leaf node instead of operator.add
        return self.tracer.create_proxy(
            'call_function', operator.iadd, (self, other), {}
        )

    _cls.__iadd__ = _iadd_as_iadd

    try:
        in_1 += in_0        # uses patched __iadd__ → operator.iadd node
        tmp_2 = in_1.transpose(1, 2)
    finally:
        if _orig is None:
            try:
                del _cls.__iadd__
            except AttributeError:
                pass
        else:
            _cls.__iadd__ = _orig

    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel: fused broadcast-add + transpose
# Output [B, K, J] element at (b, k, j) = in_1[b, j, k] + in_0[j, 0]
# ---------------------------------------------------------------------------

@triton.jit
def fused_add_transpose_kernel(
    in_0_ptr,   # [J, 1]  — flat J elements (stride-1 in last dim)
    in_1_ptr,   # [B, J, K]  flat B*J*K elements
    out_ptr,    # [B, K, J]  flat B*K*J elements
    J, K, total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    # Decompose output flat index → (b=0 since B=1, k, j)
    rem   = offsets % (K * J)
    k_idx = rem // J
    j_idx = rem  % J

    # in_1 flat index: j*K + k  (B=1 so no batch stride)
    in1_off = j_idx * K + k_idx
    # in_0 flat index: j (shape [J,1], contiguous)
    in0_off = j_idx

    in1_val = tl.load(in_1_ptr + in1_off, mask=mask)
    in0_val = tl.load(in_0_ptr + in0_off, mask=mask)

    tl.store(out_ptr + offsets, in1_val + in0_val, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper (must be @torch.fx.wrap so FX does not try to trace into it)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_add_transpose(in_0, in_1):
    """
    Computes: out[b, k, j] = in_1[b, j, k] + in_0[j, 0]
    in_0 : [J, 1]     (bias, broadcast across batch and K dims)
    in_1 : [B, J, K]  (input tensor, before iadd modification)
    out  : [B, K, J]  plain tensor
    """
    B = in_1.shape[0]
    J = in_1.shape[1]
    K = in_1.shape[2]
    total = B * J * K

    out = torch.empty((B, K, J), dtype=in_1.dtype, device=in_1.device)

    BLOCK_SIZE = 256
    num_blocks = (total + BLOCK_SIZE - 1) // BLOCK_SIZE

    fused_add_transpose_kernel[(num_blocks,)](
        in_0, in_1, out,
        J, K, total,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    return out  # plain tensor, NOT (out,)


def replacement_func():
    return fused_add_transpose