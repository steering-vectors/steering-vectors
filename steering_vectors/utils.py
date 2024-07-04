from collections.abc import Generator, Sequence
from typing import TypeVar

from tqdm import tqdm

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
