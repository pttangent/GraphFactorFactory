import pytest

pytest.importorskip("qlib")
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


def test_real_qlib_classes_import():
    assert DataHandlerLP is not None
    assert DatasetH is not None
