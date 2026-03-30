import torch
import triton
import triton.language as tl


def pattern(a, b, c):
    """
    Matches the 3-op chain from BiSeNetV2 branch 1:
      sig        = sigmoid(a)          # a = interp(in4) at 64x64
      mul_result = b * sig             # b = in3
      out        = mul_result + c      # c = interp(in2 * sigmoid(conv2d))
    Fuses sigmoid + multiply + add into a single Triton kernel.
    """
    sig = torch.sigmoid(a)
    mul_result = b * sig
    out = mul_result + c
    return out


def replacement_args(a, b, c):
    return (a, b, c)


@triton.jit
def sigmoid_mul_add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # evict_first: don't pollute L2 with streaming data we won't reuse
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0, eviction_policy='evict_first')
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0, eviction_policy='evict_first')
    c = tl.load(c_ptr + offsets, mask=mask, other=0.0, eviction_policy='evict_first')

    # Compute b * sigmoid(a) + c; cast a to fp32 for sigmoid
    sig_a = tl.sigmoid(a.to(tl.float32)).to(a.dtype)
    out = b * sig_a + c

    tl.store(out_ptr + offsets, out, mask=mask, eviction_policy='evict_first')


@torch.fx.wrap
def fused_sigmoid_mul_add(a, b, c):
    n_elements = a.numel()
    out = torch.empty_like(b)
    BLOCK_SIZE = 2048
    n_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    sigmoid_mul_add_kernel[(n_blocks,)](
        a, b, c, out, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    return out


def replacement_func():
    return fused_sigmoid_mul_add