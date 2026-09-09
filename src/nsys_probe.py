"""Tiny four-worker CUDA Graph capture to verify tracing before loading weights."""

from __future__ import annotations


def worker(rank: int, barrier: object) -> None:
    import torch

    torch.cuda.set_device(rank)
    x = torch.ones(1024, device=f"cuda:{rank}")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            y = x + 1
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = x + 1
    torch.cuda.synchronize()
    barrier.wait(timeout=120)
    torch.cuda.profiler.start()
    barrier.wait(timeout=120)
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    barrier.wait(timeout=120)
    torch.cuda.profiler.stop()
    assert y[0].item() == 2
    print(f"probe rank={rank} graph replay complete", flush=True)


def main() -> None:
    import multiprocessing as mp

    context = mp.get_context("spawn")
    barrier = context.Barrier(4)
    children = [context.Process(target=worker, args=(rank, barrier)) for rank in range(4)]
    try:
        for child in children:
            child.start()
        for child in children:
            child.join(timeout=180)
        if any(child.exitcode != 0 for child in children):
            raise RuntimeError(f"probe failures: {[child.exitcode for child in children]}")
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(timeout=5)
                if child.is_alive():
                    child.kill()
                    child.join()


if __name__ == "__main__":
    main()
