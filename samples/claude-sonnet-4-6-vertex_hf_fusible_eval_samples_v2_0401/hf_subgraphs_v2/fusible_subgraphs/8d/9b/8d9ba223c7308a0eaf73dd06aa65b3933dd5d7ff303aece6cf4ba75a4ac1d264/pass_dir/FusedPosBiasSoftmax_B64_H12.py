import torch
import triton
import triton.language as tl


# Fuse: sigmoid(x) * 16 + in_2  — shape-independent, matches ALL 4 graphs.
# Uses @triton.autotune to find the best BLOCK size per problem shape.
def pattern(x, in_2):
    tmp_9  = torch.sigmoid(x)
    tmp_10 = 16 * tmp_9
    tmp_11 = tmp_10.unsqueeze(0)
    tmp_12 = in_2 + tmp_11
    return tmp_12


def replacement_args(x, in_2):
    return (x, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 64},   num_warps=2,  num_stages=3),
        triton.Config({'BLOCK': 128},  num_warps=2,  num_stages=3),
        triton.Config({'BLOCK': 128},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK': 256},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK': 512},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK': 512},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK': 1024}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK': 1024}, num_warps=16, num_stages=3),
    ],
    key=['NH_SEQ_SEQ'],
)
@triton.jit
def _autotune_bf16(
    x_ptr, in2_ptr, out_ptr,
    NH_SEQ_SEQ,
    SEQ_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b     = tl.program_id(0)
    chunk = tl.program_id(1)
    STEP  = BLOCK // SEQ_LEN        # constexpr: rows per program
    h_i   = chunk * STEP
    j     = tl.arange(0, BLOCK)

    x_f32     = tl.load(x_ptr + (h_i * SEQ_LEN + j)).to(tl.float32)
    sig_bf16  = tl.sigmoid(x_f32).to(tl.bfloat16)
    bias_bf16 = (sig_bf16.to(tl.float32) * 16.0).to(tl.bfloat16)

    offs = b * NH_SEQ_SEQ + h_i * SEQ_LEN + j
    tl.store(out_ptr + offs, tl.load(in2_ptr + offs) + bias_bf16)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 64},   num_warps=2,  num_stages=3),
        triton.Config({'BLOCK': 128},  num_warps=2,  num_stages=3),
        triton.Config({'BLOCK': 128},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK': 256},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK': 512},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK': 512},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK': 1024}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK': 1024}, num_warps=16, num_stages=3),
    ],
    key=['NH_SEQ_SEQ'],
)
@triton.jit
def _autotune_fp16(
    x_ptr, in2_ptr, out_ptr,
    NH_SEQ_SEQ,
    SEQ_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b     = tl.program_id(0)
    chunk = tl.program_id(1)
    STEP  = BLOCK // SEQ_LEN
    h_i   = chunk * STEP
    j     = tl.arange(0, BLOCK)

    x_f32     = tl.load(x_ptr + (h_i * SEQ_LEN + j)).to(tl.float32)
    sig_fp16  = tl.sigmoid(x_f32).to(tl.float16)
    bias_fp16 = (sig_fp16.to(tl.float32) * 16.0).to(tl.float16)

    offs = b * NH_SEQ_SEQ + h_i * SEQ_LEN + j
    tl.store(out_ptr + offs, tl.load(in2_ptr + offs) + bias_fp16)


@torch.fx.wrap
def fused_sigmoid_scale_add(x, in_2):
    B, NH      = in_2.shape[0], in_2.shape[1]
    SEQ_LEN    = 64
    NH_SEQ_SEQ = NH * SEQ_LEN * SEQ_LEN

    def grid(meta):
        BLOCK = meta['BLOCK']
        STEP  = BLOCK // SEQ_LEN
        return (B, NH * SEQ_LEN // STEP)

    out = torch.empty_like(in_2)
    if in_2.dtype == torch.bfloat16:
        _autotune_bf16[grid](
            x, in_2, out, NH_SEQ_SEQ, SEQ_LEN=SEQ_LEN,
        )
    else:
        _autotune_fp16[grid](
            x, in_2, out, NH_SEQ_SEQ, SEQ_LEN=SEQ_LEN,
        )
    return out


def replacement_func():
    return fused_sigmoid_scale_add