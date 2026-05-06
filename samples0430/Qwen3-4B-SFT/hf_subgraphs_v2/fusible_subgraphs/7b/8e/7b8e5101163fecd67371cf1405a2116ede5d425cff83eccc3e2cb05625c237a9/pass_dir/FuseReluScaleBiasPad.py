import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    tmp_5 = torch.nn.functional.pad(tmp_4, (0, 1, 0, 1), 'constant', None)
    return (tmp_5,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def _fused_relu_scale_bias_pad_kernel(
    in0_ptr,   # bias, shape [1]
    in1_ptr,   # scale, shape [1]
    in2_ptr,   # input feature map [B, C, H_in, W_in]
    out_ptr,   # output, shape [B, C, H_in+2, W_in+2]
    N,         # B * C * H_in * W_in  -- total input elements
    total_out, # B * C * (H_in+2) * (W_in+2) -- total padded output elements
    C,         # channels
    H_in,      # input height
    W_in,      # input width
    H_out: tl.constexpr,  # = H_in + 2
    W_out: tl.constexpr,  # = W_in + 2
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: relu(in2) * in1 + in0, written into padded output.
    Padded positions (border) store 0.0.

    Key insight: the padded output [B, C, H_in+2, W_in+2] stores the
    computed result at positions [b,c,h,w] for h in [0,H_in), w in [0,W_in).
    All padded border cells contain 0.0.

    safe_idx (when masked=False) = (b*C + c)*H_in*W_in + h_in*W_in + w_in
                                   = the same range as the input, never OOB.
    When masked=True, safe_idx lands at the last valid input index even
    for the element immediately to the right of the rightmost column of
    the input region, which avoids any potential OOB read on broken GPUs.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_out
    out_of_bounds = ~mask

    # Decompose output linear index -> (b, c, h_out, w_out)
    # We'll use H_out, W_out as constexpr so the compiler can do strict modulo.
    hw_out      = offsets % (W_out * H_out)
    w_out       = hw_out % W_out
    h_out       = hw_out // W_out

    # Destination: [B, C, H_out, W_out] row-major
    HW_total    = H_out * W_out
    bc_out      = offsets // HW_total
    rem         = offsets % HW_total

    # Identify "padded" output positions (h_out == H_in or w_out == W_in)
    is_padded   = (h_out == H_in - 1) | (w_out == W_in - 1)
    mask_valid  = mask & ~is_padded          # only padded positions trigger OOB

    # Safe read index for non-padded positions (always a valid input address).
    # The padded-index byte for a given (h_out, w_out) is:
    #   dr = (h_out - H_in) * W_out + (w_out - W_in) = w_out - W_in + (H_in - h_out) * W_out
    #   so rem + dr = rem + w_out - W_in + (H_in - h_out) * W_out
    # For (H_in-1, W_in-1) this gives (H_in-1)*W_in + (W_in-1) = H_in*W_in - 1 (valid).
    # For any other padded cell (h_out=H_in, w_out<W_in): the rem wraps predictably.
    padded_d    = w_out - W_in + (H_in - h_out) * W_out
    safe_idx    = rem + padded_d

    in2_val = tl.load(in2_ptr + safe_idx, mask=mask_valid, other=0.0).to(tl.float32)
    relu_val = tl.maximum(in2_val, 0.0)

    in0_val = tl.load(in0_ptr).to(tl.float32)
    in1_val = tl.load(in1_ptr).to(tl.float32)

    result = relu_val * in1_val + in0_val

    tl.store(out_ptr + offsets, result.to(result.dtype), mask=mask & ~out_of_bounds)


@torch.fx.wrap
def fused_relu_scale_bias_pad(in_0, in_1, in_2):
    B, C, H_in, W_in = in_2.shape
    H_out = H_in + 2
    W_out = W_in + 2

    total_out = B * C * H_out * W_out
    N         = B * C * H_in * W_in

    out = torch.empty((B, C, H_out, W_out), dtype=in_2.dtype, device=in_2.device)

    grid = lambda meta: (triton.cdiv(total_out, meta['BLOCK_SIZE']),)

    _fused_relu_scale_bias_pad_kernel[grid](
        in_0, in_1, in_2, out,
        N, total_out,
        C, H_in, W_in,
        H_out, W_out,
    )

    return (out,)


def replacement_func():
    return fused_relu_scale_bias_pad