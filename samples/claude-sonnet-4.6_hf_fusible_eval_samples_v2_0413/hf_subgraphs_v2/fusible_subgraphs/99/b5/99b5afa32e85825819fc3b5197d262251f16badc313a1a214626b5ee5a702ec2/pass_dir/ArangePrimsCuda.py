import torch
import triton
import triton.language as tl
from torch import device

from pass_dir.arange_shared import dispatch_arange


# ── Pattern: torch.arange(1, device=device_arg) ───────────────────────────────
# pattern() is exempt from API validation.
# device_arg is a placeholder so the get_attr node for the device constant
# stays alive (has a user) after replacement — prevents _tensor_constantN
# from being pruned and causing an AttributeError.
def pattern(device_arg):
    import inspect
    tracer = None
    frame = inspect.currentframe()
    try:
        while frame is not None:
            obj = frame.f_locals.get('self', None)
            if obj is not None and hasattr(obj, 'create_proxy') and hasattr(obj, 'graph'):
                tracer = obj
                break
            frame = frame.f_back
    finally:
        del frame  # avoid reference cycle

    if tracer is not None:
        # Build call_function[torch.arange](1, device=device_arg)
        # device_arg is a Proxy (placeholder), keeping the get_attr node alive.
        return tracer.create_proxy(
            'call_function',
            torch.arange,
            (1,),
            {'device': device_arg},
        )

    # Fallback: execute concretely (no active tracer found)
    return torch.arange(1, device=device_arg)


def replacement_args(device_arg):
    # Pass device_arg as first arg so the get_attr[_tensor_constantN] node
    # (which holds the device object) is used by dispatch_arange → stays alive.
    return (device_arg,)


def replacement_func():
    return dispatch_arange