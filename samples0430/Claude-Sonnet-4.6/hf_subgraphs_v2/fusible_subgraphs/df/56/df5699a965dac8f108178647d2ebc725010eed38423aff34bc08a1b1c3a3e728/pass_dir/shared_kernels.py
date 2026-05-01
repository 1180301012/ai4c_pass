import torch
import triton
import triton.language as tl


# ─── Channel Shuffle kernel (kept for route "sh") ─────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 2048}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 512}),
    ],
    key=["total"],
)
@triton.jit
def _channel_shuffle_kernel(
    A_ptr, B_ptr, out_ptr,
    total, HW, in_CHW,
    BLOCK_SIZE: tl.constexpr,
):
    pid      = tl.program_id(0)
    offs     = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_CHW  = in_CHW * 2
    mask     = offs < total

    b_idx    = offs // out_CHW
    rem      = offs - b_idx * out_CHW
    c_out    = rem // HW
    hw_idx   = rem - c_out * HW

    from_B   = (c_out % 2) == 1
    src_c    = c_out >> 1
    src      = b_idx * in_CHW + src_c * HW + hw_idx

    a_val    = tl.load(A_ptr + src, mask=mask, other=0.0)
    b_val    = tl.load(B_ptr + src, mask=mask, other=0.0)
    tl.store(out_ptr + offs, tl.where(from_B, b_val, a_val), mask=mask)


def _channel_shuffle_impl(A, B_tensor):
    batch, C_in, H, W = A.shape
    HW      = H * W
    in_CHW  = C_in * HW
    total   = batch * 2 * C_in * HW

    out = torch.empty(batch, 2 * C_in, H, W, dtype=A.dtype, device=A.device)
    grid = lambda meta: ((total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _channel_shuffle_kernel[grid](A, B_tensor, out, total, HW, in_CHW)
    return out


# ─── Transpose-12-contiguous kernel (route "sc") ──────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 512}),
        triton.Config({"BLOCK_HW": 256}),
        triton.Config({"BLOCK_HW": 128}),
        triton.Config({"BLOCK_HW": 64}),
    ],
    key=["HW"],
)
@triton.jit
def _transpose_12_contiguous_kernel(
    in_ptr, out_ptr,
    D2, HW, D2_HW,
    BLOCK_HW: tl.constexpr,
):
    """
    2-D grid: axis-0 = (b, c) flattened, axis-1 = hw block.
    Input  layout: (B, 2, D2, HW)  – contiguous
    Output layout: (B, D2, 2, HW)  – contiguous

    For each (b, c) pair reads both g=0 and g=1 slices and writes them
    to the transposed positions.  No per-element integer division.
    """
    bc_pid = tl.program_id(0)   # linearised (b, c) → b*D2 + c
    hw_pid = tl.program_id(1)

    # Recover (b, c) — one division per BLOCK, not per element
    b = bc_pid // D2
    c = bc_pid % D2

    hw_offs = hw_pid * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offs < HW

    # In layout:  (B, 2, D2, HW)  – flat[b,g,c,hw] = b*2*D2_HW + g*D2_HW + c*HW + hw
    two_D2_HW  = 2 * D2_HW
    in_g0_base = b * two_D2_HW + c * HW           # g = 0
    in_g1_base = in_g0_base   + D2_HW             # g = 1

    # Out layout: (B, D2, 2, HW)  – flat[b,c,g,hw] = b*2*D2_HW + c*2*HW + g*HW + hw
    two_HW     = 2 * HW
    out_g0_base = b * two_D2_HW + c * two_HW      # g = 0
    out_g1_base = out_g0_base   + HW               # g = 1

    v0 = tl.load(in_ptr  + in_g0_base  + hw_offs, mask=hw_mask, other=0.0)
    v1 = tl.load(in_ptr  + in_g1_base  + hw_offs, mask=hw_mask, other=0.0)
    tl.store(out_ptr + out_g0_base + hw_offs, v0, mask=hw_mask)
    tl.store(out_ptr + out_g1_base + hw_offs, v1, mask=hw_mask)


def _strided_copy_impl(x):
    """
    Input  x : (B, 2, D2, H, W) contiguous  (result of view before transpose)
    Output   : (B, D2, 2, H, W) contiguous  (result of contiguous() after transpose)
    """
    B, D1, D2, H, W = x.shape      # D1 is always 2 here
    HW    = H * W
    D2_HW = D2 * HW

    out  = torch.empty(B, D2, D1, H, W, dtype=x.dtype, device=x.device)
    grid = lambda meta: (B * D2, (HW + meta['BLOCK_HW'] - 1) // meta['BLOCK_HW'])
    _transpose_12_contiguous_kernel[grid](x, out, D2, HW, D2_HW)
    return out


# ─── Fused Conv-Sigmoid-Scale kernel (route "css") ────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 512,  "BLOCK_CIN": 16}),
        triton.Config({"BLOCK_HW": 256,  "BLOCK_CIN": 16}),
        triton.Config({"BLOCK_HW": 128,  "BLOCK_CIN": 16}),
        triton.Config({"BLOCK_HW": 64,   "BLOCK_CIN": 16}),
    ],
    key=["B", "C_out", "HW"],
)
@triton.jit
def _fused_conv_sigmoid_scale_kernel(
    x_ptr, w_ptr, bias_ptr, s_ptr, out_ptr,
    B, C_in, C_out, HW,
    BLOCK_HW:  tl.constexpr,
    BLOCK_CIN: tl.constexpr,
):
    bc_pid    = tl.program_id(0)
    hw_pid    = tl.program_id(1)
    b_idx     = bc_pid // C_out
    co_idx    = bc_pid %  C_out

    c_in_offs = tl.arange(0, BLOCK_CIN)
    cin_mask  = c_in_offs < C_in
    x_vals  = tl.load(x_ptr + b_idx * C_in + c_in_offs, mask=cin_mask, other=0.0).to(tl.float32)
    w_vals  = tl.load(w_ptr + co_idx * C_in + c_in_offs, mask=cin_mask, other=0.0).to(tl.float32)
    acc     = tl.sum(x_vals * w_vals, axis=0)
    bias_val = tl.load(bias_ptr + co_idx).to(tl.float32)
    sig_val  = 1.0 / (1.0 + tl.exp(-(acc + bias_val)))

    hw_offs  = hw_pid * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask  = hw_offs < HW
    s_base   = b_idx * C_out * HW + co_idx * HW
    s_vals   = tl.load(s_ptr + s_base + hw_offs, mask=hw_mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + s_base + hw_offs, (s_vals * sig_val).to(s_vals.dtype), mask=hw_mask)


def _fused_conv_sigmoid_scale_impl(bias, weight, scale_input, conv_input):
    B, C_in, _, _ = conv_input.shape
    C_out         = weight.shape[0]
    _, _, H, W    = scale_input.shape
    HW            = H * W
    out  = torch.empty_like(scale_input)
    grid = lambda meta: (B * C_out, (HW + meta['BLOCK_HW'] - 1) // meta['BLOCK_HW'])
    _fused_conv_sigmoid_scale_kernel[grid](
        conv_input, weight, bias, scale_input, out, B, C_in, C_out, HW)
    return out


# ─── Unified dispatch (single replacement_func for ALL passes) ─────────────────
@torch.fx.wrap
def dispatch(a, b, c, d, route):
    """
    route 'css' → fused conv-sigmoid-scale  (a=bias, b=weight, c=scale, d=conv_in)
    route 'sh'  → channel-shuffle interleave (a=A, b=B_tensor; c,d unused)
    route 'sc'  → strided transpose-12-contiguous (a=x; b,c,d unused)
    All routes return a single tensor.
    """
    if route == "css":
        return _fused_conv_sigmoid_scale_impl(a, b, c, d)
    elif route == "sc":
        return _strided_copy_impl(a)
    else:
        return _channel_shuffle_impl(a, b)