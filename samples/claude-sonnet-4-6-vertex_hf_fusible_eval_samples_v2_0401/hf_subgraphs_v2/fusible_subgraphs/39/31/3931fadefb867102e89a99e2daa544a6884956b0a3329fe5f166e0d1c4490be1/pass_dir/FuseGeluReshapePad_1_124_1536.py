import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0)
    tmp_1 = tmp_0.reshape(1, 124, 2, 768)
    tmp_2 = tmp_1.reshape(1, 248, 768)
    tmp_3 = torch.nn.functional.pad(tmp_2, (0, 0, 0, 1), 'constant', None)
    return tmp_3


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
        triton.Config({'BLOCK_SIZE': 8192}),
    ],
    key=['N_OUTPUT'],
)
@triton.jit
def gelu_reshape_pad_kernel(
    input_ptr,
    output_ptr,
    N_INPUT,
    N_OUTPUT,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    output_mask = offsets < N_OUTPUT
    input_mask = offsets < N_INPUT

    # Load; elements in the padding region (offsets >= N_INPUT) get other=0.0
    x = tl.load(input_ptr + offsets, mask=input_mask, other=0.0)

    # Upcast to fp32 for numerics
    x_fp32 = x.to(tl.float32)

    # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    # GELU(0) = 0, so padding elements are correctly zeroed.
    inv_sqrt2 = 0.7071067811865476
    gelu_out = x_fp32 * 0.5 * (1.0 + tl.math.erf(x_fp32 * inv_sqrt2))

    # Cast back to original dtype
    gelu_out = gelu_out.to(x.dtype)

    tl.store(output_ptr + offsets, gelu_out, mask=output_mask)


@torch.fx.wrap
def fused_gelu_reshape_pad(in_0):
    # in_0: [1, 124, 1536]  ->  gelu  ->  reshape [1,248,768]  ->  pad [1,249,768]
    N_INPUT = in_0.numel()   # 1 * 124 * 1536 = 190464
    N_OUTPUT = 249 * 768     # 191232

    out = torch.empty(1, 249, 768, dtype=in_0.dtype, device=in_0.device)

    def grid(meta):
        return ((N_OUTPUT + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    gelu_reshape_pad_kernel[grid](
        in_0,
        out,
        N_INPUT,
        N_OUTPUT,
    )

    return out


def replacement_func():
    return fused_gelu_reshape_pad