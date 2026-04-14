import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: full GELU approximation (tanh variant)
#   tmp_0 = 0.5 * in_0
#   tmp_1 = torch.pow(in_0, 3.0)
#   tmp_2 = 0.044715 * tmp_1
#   tmp_3 = in_0 + tmp_2
#   tmp_4 = 0.7978845608028654 * tmp_3
#   tmp_5 = torch.tanh(tmp_4)
#   tmp_6 = 1.0 + tmp_5
#   tmp_7 = tmp_0 * tmp_6
# ──────────────────────────────────────────────────────────────────────────────
def pattern(in_0):
    tmp_0 = 0.5 * in_0
    tmp_1 = torch.pow(in_0, 3.0)
    tmp_2 = 0.044715 * tmp_1
    tmp_3 = in_0 + tmp_2
    tmp_4 = 0.7978845608028654 * tmp_3
    tmp_5 = torch.tanh(tmp_4)
    tmp_6 = 1.0 + tmp_5
    tmp_7 = tmp_0 * tmp_6
    return tmp_7


def replacement_args(in_0):
    return (in_0,)


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel – fused GELU (tanh approximation)
# ──────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 2048},  num_warps=4,  num_stages=1),
        triton.Config({"BLOCK_SIZE": 2048},  num_warps=8,  num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096},  num_warps=4,  num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096},  num_warps=8,  num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096},  num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 8192},  num_warps=8,  num_stages=1),
        triton.Config({"BLOCK_SIZE": 8192},  num_warps=16, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def _gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load in original dtype, upcast to fp32 for stable computation
    # evict_first: streaming hint — each cache line is read once; free it early
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first")
    x_f32 = x.to(tl.float32)

    # GELU = 0.5 * x * (1 + tanh( 0.7978845608 * (x + 0.044715 * x^3) ))
    x3 = x_f32 * x_f32 * x_f32
    inner = 0.7978845608028654 * (x_f32 + 0.044715 * x3)
    # Use hardware-native tanh.approx.f32 PTX instruction (fast, ~4x vs libdevice)
    tanh_inner = tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;", "=f,f", [inner],
        dtype=tl.float32, is_pure=True, pack=1
    )
    out_f32 = 0.5 * x_f32 * (1.0 + tanh_inner)

    # Cast back to the original dtype and store (evict_first: output is write-once)
    tl.store(out_ptr + offsets, out_f32.to(x.dtype), mask=mask, eviction_policy="evict_first")


@torch.fx.wrap
def gelu_triton(in_0):
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)

    def grid(meta):
        return ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    _gelu_kernel[grid](in_0, out, n_elements)
    return out


def replacement_func():
    return gelu_triton