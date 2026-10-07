"""Cold H2D copies on one stream and on LMCACHE_H2D_STREAMS streams.

Matches ``_queue_h2d_copies``: chunk i is queued on stream i % K, then the
main stream waits for the other streams before synchronize.
"""

from __future__ import annotations

import os
import time

import torch


def queue_copies(streams: list[torch.npu.Stream], destination: torch.Tensor, sources: list[torch.Tensor]) -> float:
    """Queue every copy, wait for side streams, and return the wall time in microseconds."""
    main = streams[0]
    started = time.perf_counter()
    for index, source in enumerate(sources):
        with torch.npu.stream(streams[index % len(streams)]):
            destination.copy_(source, non_blocking=True)
    for side in streams[1:]:
        main.wait_stream(side)
    main.synchronize()
    return (time.perf_counter() - started) * 1e6


def main() -> None:
    devices = [int(item) for item in os.environ.get("H2D_DEVICES", "4,5,6,7").split(",") if item]
    window_bytes = int(os.environ.get("H2D_BYTES", str(32 * 1024 * 1024)))
    count = int(os.environ.get("H2D_N", "32"))
    stream_count = max(1, int(os.environ.get("LMCACHE_H2D_STREAMS", "4")))

    sources_one = [torch.empty(window_bytes, dtype=torch.uint8, pin_memory=True) for _ in range(count)]
    sources_many = [torch.empty(window_bytes, dtype=torch.uint8, pin_memory=True) for _ in range(count)]
    for tensor in sources_one + sources_many:
        tensor[0] = 1
        tensor[-1] = 1

    print(
        f"BEGIN devices={devices} bytes={window_bytes} count={count} streams={stream_count} "
        f"one_first={hex(sources_one[0].data_ptr())} many_first={hex(sources_many[0].data_ptr())}",
        flush=True,
    )
    for device in devices:
        torch.npu.set_device(device)
        destination = torch.empty(window_bytes, dtype=torch.uint8, device=f"npu:{device}")
        one = [torch.npu.Stream(device=device)]
        many = [torch.npu.Stream(device=device) for _ in range(stream_count)]
        one_us = queue_copies(one, destination, sources_one)
        many_us = queue_copies(many, destination, sources_many)
        print(
            f"DEVICE {device} one_stream_us={one_us:.1f} streams_{stream_count}_us={many_us:.1f}",
            flush=True,
        )
        del destination
        torch.npu.synchronize(device)
    print("END", flush=True)


if __name__ == "__main__":
    main()
