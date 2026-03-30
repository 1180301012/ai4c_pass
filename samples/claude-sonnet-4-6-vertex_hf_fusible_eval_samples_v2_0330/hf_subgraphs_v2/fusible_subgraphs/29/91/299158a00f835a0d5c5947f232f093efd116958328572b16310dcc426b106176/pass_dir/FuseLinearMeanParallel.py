import torch


def pattern(in_0, in_1, in_2, in_3):
    """
    Match both independent operations together:
      - linear(in_2, in_1, in_0)
      - in_3.mean(-2)
    Both are returned from the model, so both must appear here.
    """
    linear_out = torch.nn.functional.linear(in_2, in_1, in_0)
    mean_out = in_3.mean(-2)
    return linear_out, mean_out


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# Pre-allocated side stream (lazy init to avoid CUDA-not-initialized errors at import time)
_side_stream = None


def _get_side_stream():
    global _side_stream
    if _side_stream is None:
        _side_stream = torch.cuda.Stream()
    return _side_stream


@torch.fx.wrap
def parallel_linear_and_mean(in_0, in_1, in_2, in_3):
    """
    Run linear and mean concurrently on separate CUDA streams.
    They share no data, so the GPU can overlap their execution.
    """
    main_stream = torch.cuda.current_stream()
    side_stream = _get_side_stream()

    # ── Side stream: submit the larger mean reduction ──────────────────────
    side_stream.wait_stream(main_stream)   # respects prior main-stream work
    with torch.cuda.stream(side_stream):
        mean_out = in_3.mean(-2)

    # ── Main stream: submit linear (runs concurrently with mean) ───────────
    linear_out = torch.nn.functional.linear(in_2, in_1, in_0)

    # ── Sync: main stream waits for mean to finish ─────────────────────────
    main_stream.wait_stream(side_stream)

    return linear_out, mean_out


def replacement_func():
    return parallel_linear_and_mean