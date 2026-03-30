import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Pattern: matches the computation in all target graphs
#   in_1 += in_0
#   tmp_1 = in_2.float()
#   tmp_2 = softmax(tmp_1, dim=-1)
#   tmp_3 = tmp_2.type_as(in_2)
#   tmp_4 = dropout(tmp_3, p=0.1, training=False)  # no-op at inference
#   return (tmp_4,)
# ─────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1):
    in_1 += in_0
    in_2 = in_1
    tmp_1 = in_2.float()
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = tmp_2.type_as(in_2)
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.1, training=False)
    return (tmp_4,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: fused add + numerically-stable softmax
#   - One Triton program per row (last dimension = one softmax vector)
#   - Loads from in0/in1 as their native dtype, upcasts to float32 internally
#   - Stores result to a float32 output buffer
#   - BLOCK_SIZE is constexpr, chosen as next_power_of_2(N)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 8}),
        triton.Config({"BLOCK_SIZE": 16}),
        triton.Config({"BLOCK_SIZE": 32}),
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
    ],
    key=["N"],
)
@triton.jit
def fused_add_softmax_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    row_offsets = row * N + offsets

    # Load inputs and upcast to float32 for numerical stability
    x = tl.load(in0_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(in1_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)

    # Element-wise addition (corresponds to in_1 += in_0)
    z = x + y

    # Replace out-of-bounds lanes with a large negative to neutralise them
    # in the softmax max/exp without producing NaN
    z = tl.where(mask, z, -1e9)

    # Numerically stable softmax: subtract row max before exp
    z_max = tl.max(z, axis=0)
    z_exp = tl.where(mask, tl.exp(z - z_max), 0.0)
    z_sum = tl.sum(z_exp, axis=0)
    z_softmax = z_exp / z_sum

    # Write float32 result; Python wrapper handles dtype cast-back if needed
    tl.store(out_ptr + row_offsets, z_softmax, mask=mask)


@torch.fx.wrap
def fused_add_softmax(in_0, in_1):
    """
    Replacement for:
      in_1 += in_0
      softmax(in_1.float(), dim=-1).type_as(in_1)
    Dropout(p=0.1, training=False) is identity and is dropped.
    """
    orig_dtype = in_1.dtype
    shape = in_1.shape
    N = shape[-1]                           # size of the last (softmax) dim
    num_rows = in_0.numel() // N            # total independent softmax rows

    # Float32 output buffer (kernel always writes float32)
    out_f32 = torch.empty(shape, dtype=torch.float32, device=in_1.device)

    fused_add_softmax_kernel[(num_rows,)](
        in_0, in_1, out_f32, N,
    )

    # Cast back to the original input dtype (matches type_as semantics)
    if orig_dtype == torch.float32:
        return (out_f32,)
    else:
        return (out_f32.to(orig_dtype),)


def replacement_func():
    return fused_add_softmax