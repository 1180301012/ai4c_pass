import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_2 + in_3
    tmp_3 = tmp_2.reshape(-1, 768)
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (768,), in_1, in_0, 1e-05)
    return (tmp_3, tmp_4)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def fused_add_layer_norm_768_kernel(
    in2_ptr,
    in3_ptr,
    weight_ptr,
    bias_ptr,
    out_reshaped_ptr,
    out_normed_ptr,
    eps,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < HIDDEN_SIZE

    row_start = row_idx * HIDDEN_SIZE

    # Load inputs and compute element-wise sum
    x2 = tl.load(in2_ptr + row_start + col_offsets, mask=mask, other=0.0)
    x3 = tl.load(in3_ptr + row_start + col_offsets, mask=mask, other=0.0)
    x = x2 + x3

    # Store the reshaped (added) result — this is tmp_3
    tl.store(out_reshaped_ptr + row_start + col_offsets, x, mask=mask)

    # Compute mean in fp32 for numerical stability
    # Masked (padded) elements were loaded as 0.0 so they do not affect the sum
    x_fp32 = x.to(tl.float32)
    mean = tl.sum(x_fp32, axis=0) / HIDDEN_SIZE

    # Compute variance — zero out the padded elements so they don't bias the result
    diff = tl.where(mask, x_fp32 - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / HIDDEN_SIZE

    # Normalize
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = diff * inv_std

    # Apply weight and bias (loaded in fp32)
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
    bias_val = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    out = x_norm * weight + bias_val

    # Store the layer-norm result cast back to the input dtype — this is tmp_4
    tl.store(out_normed_ptr + row_start + col_offsets, out.to(x.dtype), mask=mask)


# ── Inner launcher (opaque to FX) ──────────────────────────────────────────────
# Returns a [2, N, 768] buffer: [0] = out_reshaped, [1] = out_normed.
# Returning a single tensor avoids the "1-node vs 2-returning-nodes" mismatch.
@torch.fx.wrap
def _run_fused_add_layer_norm_768(in_0, in_1, in_2, in_3):
    HIDDEN_SIZE = 768
    N = in_2.numel() // HIDDEN_SIZE
    # One allocation; buffer[0] and buffer[1] are contiguous views
    buffer = torch.empty((2, N, HIDDEN_SIZE), dtype=in_2.dtype, device=in_2.device)
    out_reshaped = buffer[0]   # [N, 768]
    out_normed   = buffer[1]   # [N, 768]
    fused_add_layer_norm_768_kernel[(N,)](
        in_2,
        in_3,
        in_1,           # weight
        in_0,           # bias
        out_reshaped,
        out_normed,
        1e-5,
        HIDDEN_SIZE=HIDDEN_SIZE,
        BLOCK_SIZE=1024,
        num_warps=4,
    )
    return buffer


# ── Outer replacement (FX traces INTO this, producing 2 getitem nodes) ─────────
# NOT wrapped with @torch.fx.wrap so FX sees two independent output nodes:
#   getitem(buffer, 0)  →  replaces tmp_3
#   getitem(buffer, 1)  →  replaces tmp_4
def fused_add_layer_norm_768(in_0, in_1, in_2, in_3):
    buffer = _run_fused_add_layer_norm_768(in_0, in_1, in_2, in_3)
    return buffer[0], buffer[1]


def replacement_func():
    return fused_add_layer_norm_768