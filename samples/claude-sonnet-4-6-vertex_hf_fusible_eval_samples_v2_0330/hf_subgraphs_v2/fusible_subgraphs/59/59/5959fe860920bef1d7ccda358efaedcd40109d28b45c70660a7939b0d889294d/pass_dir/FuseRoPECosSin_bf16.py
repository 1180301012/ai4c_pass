import torch
import triton
import triton.language as tl


@triton.jit
def _rope_cos_sin_bf16_kernel(
    in_ptr,
    cos_ptr,
    sin_ptr,
    H,
    BLOCK_H: tl.constexpr,
):
    """
    Each program processes one row of in_1 (shape [N_rows, H]).
    Writes cos and sin (duplicated) to out tensors of shape [N_rows, 2*H].
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < H

    # Load one row of in_1, cast to float32 for precision
    x = tl.load(in_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)

    c = tl.cos(x).to(tl.bfloat16)
    s = tl.sin(x).to(tl.bfloat16)

    # Write first copy (positions 0 .. H-1 in output row)
    out_base = row * 2 * H
    tl.store(cos_ptr + out_base + cols,     c, mask=mask)
    tl.store(sin_ptr + out_base + cols,     s, mask=mask)
    # Write second copy (positions H .. 2H-1 in output row)
    tl.store(cos_ptr + out_base + H + cols, c, mask=mask)
    tl.store(sin_ptr + out_base + H + cols, s, mask=mask)


@torch.fx.wrap
def _rope_cos_sin_bf16_exec(in_1):
    """Runs RoPE kernel, returns (cos_out, sin_out) in bfloat16."""
    orig_shape = in_1.shape          # e.g. [B, S, H]
    H = orig_shape[-1]
    N_rows = in_1.numel() // H

    out_shape = list(orig_shape[:-1]) + [2 * H]
    cos_out = torch.empty(out_shape, dtype=torch.bfloat16, device=in_1.device)
    sin_out = torch.empty(out_shape, dtype=torch.bfloat16, device=in_1.device)

    BLOCK_H = triton.next_power_of_2(H)

    _rope_cos_sin_bf16_kernel[(N_rows,)](
        in_1, cos_out, sin_out,
        H,
        BLOCK_H=BLOCK_H,
    )

    return cos_out, sin_out


# NOT @torch.fx.wrap — FX traces into this, creating getitem nodes,
# so the replacement graph has 2 separate returning nodes.
def fused_rope_cos_sin_bf16(in_1):
    result = _rope_cos_sin_bf16_exec(in_1)
    return result[0], result[1]


def pattern(in_1):
    tmp_1 = torch.cat((in_1, in_1), dim=-1)
    tmp_2 = tmp_1.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_4 = tmp_1.sin()
    tmp_5 = tmp_4 * 1.0
    tmp_6 = tmp_3.to(dtype=torch.bfloat16)
    tmp_7 = tmp_5.to(dtype=torch.bfloat16)
    return tmp_6, tmp_7


def replacement_args(in_1):
    return (in_1,)


def replacement_func():
    return fused_rope_cos_sin_bf16