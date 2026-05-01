"""
Shared Triton kernels and dispatch wrapper.
Both FuseDropoutScaleAdd and FuseBatchNormInference import `fused_dispatch` from
here, ensuring set_g_replacement_func(fused_dispatch) is the SAME object across
all passes (satisfying the identity-check in the pass backend).

Optimisations vs. the naïve 1-D approach