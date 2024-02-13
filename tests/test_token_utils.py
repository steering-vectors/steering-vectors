import torch

from steering_vectors.token_utils import (
    adjust_read_indices_for_padding,
    find_attention_start_and_end_positions,
)


def test_find_attention_start_and_end_positions_right_padding() -> None:
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
        ]
    )
    start_positions, end_positions = find_attention_start_and_end_positions(
        attention_mask
    )
    assert start_positions.tolist() == [0, 0]
    assert end_positions.tolist() == [2, 3]


def test_find_attention_start_and_end_positions_left_padding() -> None:
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
        ]
    )
    start_positions, end_positions = find_attention_start_and_end_positions(
        attention_mask
    )
    assert start_positions.tolist() == [2, 3]
    assert end_positions.tolist() == [5, 5]


def test_adjust_read_indices_for_padding_with_right_padding() -> None:
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
        ]
    )
    read_indices = torch.tensor([1, -2])
    adjusted_indices = adjust_read_indices_for_padding(read_indices, attention_mask)
    assert adjusted_indices.tolist() == [1, 2]


def test_adjust_read_indices_for_padding_with_left_padding() -> None:
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
        ]
    )
    read_indices = torch.tensor([1, -2])
    adjusted_indices = adjust_read_indices_for_padding(read_indices, attention_mask)
    assert adjusted_indices.tolist() == [3, 4]
