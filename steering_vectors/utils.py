from collections.abc import Generator, Sequence
from typing import TypeVar

import torch
from tqdm import tqdm
from contextlib import contextmanager
from steering_vectors.core.patterns import Singleton

T = TypeVar("T")


def batchify(
    data: Sequence[T],
    batch_size: int,
    show_progress: bool = False,
    tqdm_desc: str | None = None,
) -> Generator[Sequence[T], None, None]:
    """Generate batches from data. If show_progress is True, display a progress bar."""

    for i in tqdm(
        range(0, len(data), batch_size),
        total=(len(data) // batch_size + (len(data) % batch_size != 0)),
        disable=not show_progress,
        desc=tqdm_desc,
    ):
        yield data[i : i + batch_size]


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # Parse the PyTorch version to check if it's below version 2.0
        major_version = int(torch.__version__.split(".")[0])
        if major_version >= 2:
            return "mps"
    else:
        return "cpu"

    raise RuntimeError("Should not reach here!")


@Singleton
class DeviceManager:
    device: str

    def __init__(self):
        self.device = get_default_device()

    def get_device(self) -> str:
        return self.device

    def set_device(self, device: str) -> None:
        self.device = device

    @contextmanager
    def use_device(self, device: str):
        old_device = self.get_device()
        self.set_device(device)
        yield
        self.set_device(old_device)
