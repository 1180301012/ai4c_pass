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
        triton.Config({'BLOCK_I': 4,  'BLOCK_J': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_I': 8,  'BLOCK_J': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_I': 16, 'BLOCK_J': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_I': 4,  'BLOCK_J': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_I': 8,  'BLOCK_J': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_I': 2,  'BLOCK_J': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_I': 1,  'BLOCK_J': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_I': 1,  'BLOCK_J': 512}, num_warps=4, num_stages=2),
    ],
    key=[],
)
@triton.jit
def fused_gelu_reshape_pad_kernel(
    input_ptr,
    output_ptr,
    N_I: tl.constexpr,      # 249 (output rows after padding)
    N_J: tl.constexpr,      # 768 (output columns)
    N_I_input: tl.constexpr, # 248 (input rows after two reshapes)
    N_CH: tl.constexpr,     # 768 (channels per row)
    N_COLS: tl.constexpr,   # 1536 (cols in original input [1,124,1536])
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    """
    Fused GELU + reshape + pad kernel.

    Output layout: [N_I, N_J] = [249, 768]
      - Rows 0..247:  gelu(input[row_orig, col])
      - Row  248:     0 (padding)

    Input layout:  [1, 124, 1536] contiguous → treat as [124, 1536]
    Mapping for output (i, j):
      row_orig = i * 2 + j // N_CH    (which of the 124 input rows)
      col      = j % N_CH             (column within that row, 0..767)
      flat_input_idx = row_orig * N_COLS + col

    Output flat index: i * N_J + j
    """
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    i_offsets = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)  # shape [BLOCK_I]
    j_offsets = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)  # shape [BLOCK_J]

    i_idx = i_offsets[:, None]  # [BLOCK_I, 1]
    j_idx = j_offsets[None, :]  # [1, BLOCK_J]

    # Flat output indices (contiguous [N_I, N_J] tensor)
    out_offsets = i_idx * N_J + j_idx  # [BLOCK_I, BLOCK_J]
    mask = (i_idx < N_I) & (j_idx < N_J)  # [BLOCK_I, BLOCK_J]

    is_pad = (i_idx >= N_I - 1)  # True for the last (padded) row

    # Safe input indices: clamp so we never load out-of-bounds (value used only when is_pad)
    j_mod = j_idx % N_CH          # [1, BLOCK_J] → column within 768
    j_div = j_idx // N_CH         # [1, BLOCK_J] → 0 or 1
    i_safe = tl.where(is_pad, 0, i_idx)  # [BLOCK_I, 1] safe clamp
    row_orig = i_safe * 2 + j_div[None, :]  # [BLOCK_I, BLOCK_J]
    col      = j_mod[None, :]                 # [BLOCK_I, BLOCK_J]
    in_offsets = row_orig * N_COLS + col      # [BLOCK_I, BLOCK_J]

    # Load input values (ignore masked-out elements; use 0.0 as safe placeholder)
    x = tl.load(input_ptr + in_offsets, mask=mask & ~is_pad, other=0.0)

    # GELU (exact formula: 0.5 * x * (1 + erf(x / sqrt(2))))
    x_f32 = x.to(tl.float32)
    gelu_out = 0.5 * x_f32 * (1.0 + tl.math.erf(x_f32 * 0.7071067811865476))

    # Cast back to the original dtype and zero-out the padded row
    gelu_cast = gelu_out.to(x.dtype)
    result = tl.where(is_pad, tl.zeros_like(gelu_cast), gelu_cast)

    tl.store(output_ptr + out_offsets, result, mask=mask)


@torch.fx.wrap
def fused_gelu_reshape_pad(in_0):
    N_I     = 249     # 248 + 1 padded row
    N_J     = 768
    N_I_input = 248
    N_CH    = 768
    N_COLS  = 1536

    output = torch.empty((1, N_I, N_J), dtype=in_0.dtype, device=in_0.device)

    grid = lambda meta: (
        triton.cdiv(N_I, meta['BLOCK_I']),
        triton.cdiv(N_J, meta['BLOCK_J']),
    )

    fused_gelu_reshape_pad_kernel[grid](
        in_0,
        output,
        N_I=N_I,
        N_J=N_J,
        N_I_input=N_I_input,
        N_CH=N_CH,
        N_COLS=N_COLS,
    )

    return output


def replacement_func():
    return fused_gelu_reshape_pad