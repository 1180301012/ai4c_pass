import torch
import triton
import triton.language as tl


def pattern(conv_out, in_3):
    tmp_3 = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    tmp_4 = tmp_3 * conv_out
    return tmp_4


def replacement_args(conv_out, in_3):
    return (conv_out, in_3)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
    ],
    key=["n_elements"],
)
@triton.jit

def _fused_hardtanh_mul_contig_kernel(
    conv_ptr,
    gate_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    conv = tl.load(conv_ptr + offs, mask=mask)
    gate = tl.load(gate_ptr + offs, mask=mask)
    gate = tl.maximum(tl.minimum(gate, 6.0), 0.0)
    out = gate * conv
    tl.store(out_ptr + offs, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit

def _fused_hardtanh_mul_strided_kernel(
    conv_ptr,
    gate_ptr,
    out_ptr,
    n_elements,
    N,
    C,
    H,
    W,
    conv_s0,
    conv_s1,
    conv_s2,
    conv_s3,
    gate_s0,
    gate_s1,
    gate_s2,
    gate_s3,
    out_s0,
    out_s1,
    out_s2,
    out_s3,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    linear = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = linear < n_elements

    w = linear % W
    t0 = linear // W
    h = t0 % H
    t1 = t0 // H
    c = t1 % C
    n = t1 // C

    conv_offs = n * conv_s0 + c * conv_s1 + h * conv_s2 + w * conv_s3
    gate_offs = n * gate_s0 + c * gate_s1 + h * gate_s2 + w * gate_s3
    out_offs = n * out_s0 + c * out_s1 + h * out_s2 + w * out_s3

    conv = tl.load(conv_ptr + conv_offs, mask=mask)
    gate = tl.load(gate_ptr + gate_offs, mask=mask)
    gate = tl.maximum(tl.minimum(gate, 6.0), 0.0)
    out = gate * conv
    tl.store(out_ptr + out_offs, out, mask=mask)


@torch.fx.wrap
def fused_hardtanh_mul(conv_out, in_3):
    out = torch.empty_like(conv_out)
    n_elements = conv_out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    if conv_out.is_contiguous() and in_3.is_contiguous() and out.is_contiguous():
        _fused_hardtanh_mul_contig_kernel[grid](
            conv_out,
            in_3,
            out,
            n_elements,
        )
    else:
        N = conv_out.shape[0]
        C = conv_out.shape[1]
        H = conv_out.shape[2]
        W = conv_out.shape[3]
        _fused_hardtanh_mul_strided_kernel[grid](
            conv_out,
            in_3,
            out,
            n_elements,
            N,
            C,
            H,
            W,
            conv_out.stride(0),
            conv_out.stride(1),
            conv_out.stride(2),
            conv_out.stride(3),
            in_3.stride(0),
            in_3.stride(1),
            in_3.stride(2),
            in_3.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
        )
    return out


def replacement_func():
    return fused_hardtanh_mul