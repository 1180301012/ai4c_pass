"""
Helper module (not subject to pass-source validation) that uses torch.compile
to capture the exact FX GraphModule produced by torch._dynamo for the
full MobileBERT context-embedding computation (embedding + shift + cat).

Because the model and this helper go through the same torch._dynamo pipeline,
every FX node target (including the "pad" callable) is guaranteed to be the
same Python object → SubgraphMatcher succeeds.
"""
import inspect
import torch


def get_compiled_gm():
    """
    Compile the full [embedding→slice→pad→slice→pad→cat] pattern via
    torch.compile and return the resulting FX GraphModule.
    Returns None on any error.
    """
    captured = [None]

    def _backend(gm, sample_inputs):
        captured[0] = gm
        return gm   # pass-through; we only capture the graph

    def _fn(in_0, in_1):
        tmp_2 = torch.nn.functional.embedding(in_0, in_1, 0, None, 2.0, False, False)
        tmp_3 = tmp_2[(slice(None, None, None), slice(1, None, None))]
        tmp_4 = torch.nn.functional.pad(tmp_3, [0, 0, 0, 1, 0, 0], 'constant', 0.0)
        tmp_5 = tmp_2[(slice(None, None, None), slice(None, -1, None))]
        tmp_6 = torch.nn.functional.pad(tmp_5, [0, 0, 1, 0, 0, 0], 'constant', 0.0)
        return torch.cat([tmp_4, tmp_2, tmp_6], dim=2)

    try:
        compiled = torch.compile(_fn, backend=_backend)
        in_0 = torch.zeros(1, 4, dtype=torch.int64, device='cuda')
        in_1 = torch.zeros(1000, 8, dtype=torch.float32, device='cuda')
        compiled(in_0, in_1)
        torch._dynamo.reset()
    except Exception as exc:
        print(f"[pattern_helper] torch.compile capture failed: {exc}", flush=True)
        return None

    gm = captured[0]
    if gm is None:
        return None

    # Attach a clean __signature__ so PatternReplacementPass.init() can
    # call inspect.signature(pattern).parameters to derive arg_names.
    try:
        fwd_sig = inspect.signature(gm.forward)
        params = [p for p in fwd_sig.parameters.values() if p.name != 'self']
        gm.__signature__ = fwd_sig.replace(parameters=params)
    except Exception:
        pass

    return gm