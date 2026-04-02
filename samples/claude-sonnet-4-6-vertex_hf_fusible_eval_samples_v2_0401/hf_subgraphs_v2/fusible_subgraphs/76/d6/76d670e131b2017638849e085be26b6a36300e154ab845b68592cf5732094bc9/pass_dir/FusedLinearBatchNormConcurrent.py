"""
FusedLinearBatchNormConcurrent
==============================
Matches the full sub-graph (linear + batch_norm in inference mode) and replaces
it with a concurrent execution strategy:

  Stream 1: torch.nn.functional.linear   (cuBLAS — fastest possible)
  Stream 2: Triton scale+offset kernel    (x*scale + offset, precomputed once)

The two ops share no data, so they can run in parallel.  For typical shapes
(linear on [128, 384] x [384, 1000]) the linear kernel occupies most of the
GPU; the tiny BN kernel hides inside its shadow, reducing overall latency.

BN optimisation detail