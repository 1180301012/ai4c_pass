import torch
import triton
import triton.language as tl
import sys

# ===== Shared dispatch wrapper using sys.modules =====
_REGISTRY_KEY = '_ai4c_se_block_dispatch'

if _REGISTRY_KEY not in sys.modules:

    @triton.jit
    def fused_se_kernel(
        feature_ptr, input_ptr, weight_ptr, bias_ptr, output_ptr,
        N, C_in, C_out, H, W,
        add_val, div_val,
        feat_n_s, feat_c_s, feat_hw_s,
        in_n_s, in_c_s,
        w_co_s, w_ci_s,
        b_s,
        out_n_s, out_c_s, out_hw_s,
        BLOCK_HW: tl.constexpr,
        BLOCK_CI: tl.constexpr,
    ):
        """
        Fused SE block kernel: conv2d(1x1) + hardsigmoid + broadcast multiply.
        3D grid: (N, C_out, spatial_blocks) - no division/modulo for n and c indices.
        Scale value computed once per (n, c_out) and reused across spatial positions.
        """
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_hw = tl.program_id(2)

        # Compute scale value (conv2d + activation)
        bias_val = tl.load(bias_ptr + pid_c * b_s).to(tl.float32)
        acc = bias_val

        for ci_start in range(0, C_in, BLOCK_CI):
            ci_offsets = ci_start + tl.arange(0, BLOCK_CI)
            ci_mask = ci_offsets < C_in

            w_vals = tl.load(weight_ptr + pid_c * w_co_s + ci_offsets * w_ci_s,
                             mask=ci_mask, other=0.0).to(tl.float32)
            i_vals = tl.load(input_ptr + pid_n * in_n_s + ci_offsets * in_c_s,
                             mask=ci_mask, other=0.0).to(tl.float32)
            acc += tl.sum(w_vals * i_vals)

        # hardsigmoid: clamp((x + add_val) / div_val, 0, 1)
        scale_val = tl.clamp((acc + add_val) / div_val, 0.0, 1.0)

        # Broadcast multiply - compute spatial offsets
        hw_start = pid_hw * BLOCK_HW
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < H * W

        # Use flattened spatial access for contiguous tensors
        base_offset = pid_n * feat_n_s + pid_c * feat_c_s

        f_ptrs = feature_ptr + base_offset + hw_offsets * feat_hw_s
        f_vals = tl.load(f_ptrs, mask=hw_mask, other=0.0).to(tl.float32)

        o_vals = f_vals * scale_val

        o_ptrs = output_ptr + base_offset + hw_offsets * out_hw_s
        tl.store(o_ptrs, o_vals, mask=hw_mask)


    def _impl(bias, weight, feature, input_tensor, add_val, div_val):
        """Fused SE block: conv2d(1x1) + hardsigmoid + broadcast multiply."""
        N = input_tensor.shape[0]
        C_in = input_tensor.shape[1]
        C_out = weight.shape[0]
        H_feat = feature.shape[2]
        W_feat = feature.shape[3]

        output = torch.empty_like(feature)

        # 3D grid: (N, C_out, ceil(H*W / BLOCK_HW))
        BLOCK_HW = 256
        BLOCK_CI = 64
        hw_total = H_feat * W_feat
        num_hw_blocks = (hw_total + BLOCK_HW - 1) // BLOCK_HW

        grid = (N, C_out, max(1, num_hw_blocks))

        feat_hw_s = feature.stride(3)
        out_hw_s = output.stride(3)

        fused_se_kernel[grid](
            feature_ptr=feature,
            input_ptr=input_tensor,
            weight_ptr=weight,
            bias_ptr=bias,
            output_ptr=output,
            N=N, C_in=C_in, C_out=C_out, H=H_feat, W=W_feat,
            add_val=add_val, div_val=div_val,
            feat_n_s=feature.stride(0),
            feat_c_s=feature.stride(1),
            feat_hw_s=feat_hw_s,
            in_n_s=input_tensor.stride(0),
            in_c_s=input_tensor.stride(1) if input_tensor.dim() > 1 else 0,
            w_co_s=weight.stride(0),
            w_ci_s=weight.stride(1) if weight.dim() > 1 else 0,
            b_s=bias.stride(0) if bias.dim() > 0 else 1,
            out_n_s=output.stride(0),
            out_c_s=output.stride(1),
            out_hw_s=out_hw_s,
            BLOCK_HW=BLOCK_HW,
            BLOCK_CI=BLOCK_CI,
        )

        return output


    @torch.fx.wrap
    def fused_se_block_dispatch(bias, weight, feature, input_tensor, route):
        if route == "route_1_2":
            return _impl(bias, weight, feature, input_tensor, 1.0, 2.0)
        elif route == "route_3_6":
            return _impl(bias, weight, feature, input_tensor, 3.0, 6.0)
        else:
            raise ValueError(f"Unknown route: {route}")

    # Store in sys.modules for sharing across pass files
    sys.modules[_REGISTRY_KEY] = type('module', (), {
        'fused_se_block_dispatch': fused_se_block_dispatch,
    })

# Retrieve the shared dispatch wrapper
fused_se_block_dispatch = sys.modules[_REGISTRY_KEY].fused_se_block_dispatch


# ===== Pattern-specific definitions for (x+3)/6 HardSigmoid =====

def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d + 3.0
    tmp_4 = tmp_3 / 6.0
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    tmp_6 = in_2 * tmp_5
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "route_3_6")


def replacement_func():
    return fused_se_block_dispatch