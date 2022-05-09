import torch
import pytest
import sys

sys.path.insert(0, "src")

from model import MultilabelClassifier


@pytest.mark.parametrize(
    "ids, mask", [([[101, 1188, 1110, 102, 0, 0, 0]], [[1, 1, 1, 1, 0, 0, 0]])]
)
def test_forward(ids, mask):
    n_labels = 100
    batch = 1
    expected_shape = (batch, n_labels)

    model = MultilabelClassifier(n_labels)

    ids = torch.tensor(ids, dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.long)
    actual = model.forward(ids=ids, mask=mask)

    assert expected_shape == actual.shape
