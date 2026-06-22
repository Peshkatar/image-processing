# pyright: reportArgumentType=false

import pytest
from torchvision.datasets import MNIST

from eigen.data import load_dataset


@pytest.fixture(scope="module")
def train_data() -> MNIST:
    return MNIST(root="data", train=True, download=True)


@pytest.fixture
def test_data() -> MNIST:
    return MNIST(root="data", train=False, download=True)


def test_load_dataset_raises_when_split_sum_is_not_one(train_data: MNIST) -> None:
    with pytest.raises(
        ValueError,
        match=r"`train_split` \+ `test_split` \+ `val_split` must be equal to 1",
    ):
        load_dataset(train_data, train_split=0.5, test_split=0.5, val_split=0.5)


def test_load_dataset_raises_when_test_data_provided_but_test_split_not_zero(
    train_data: MNIST, test_data: MNIST
) -> None:
    with pytest.raises(
        ValueError, match=r"`test_data` is provided but `test_split` is not 0"
    ):
        load_dataset(train_data, test_data=test_data, test_split=0.5)


def test_load_dataset_raises_when_test_data_provided_but_val_split_not_zero(
    train_data: MNIST, test_data: MNIST
) -> None:
    with pytest.raises(
        ValueError, match=r"`val_data` is provided but `val_split` is not 0"
    ):
        load_dataset(train_data, val_data=test_data, val_split=0.5)


def test_load_dataset_with_train_test(train_data: MNIST, test_data: MNIST) -> None:
    # train and test
    train_loader, test_loader = load_dataset(train_data, test_data)
    assert len(train_loader.dataset) + len(test_loader.dataset) == len(
        train_data
    ) + len(test_data)


def test_load_dataset_with_train_test_and_val_split(
    train_data: MNIST, test_data: MNIST
) -> None:
    # train, test, val (from train_data)
    train_loader, test_loader, val_loader = load_dataset(
        train_data, test_data, val_split=0.1
    )
    print(train_loader, test_loader, val_loader)
    assert len(train_loader.dataset) + len(test_loader.dataset) + len(
        val_loader.dataset
    ) == len(train_data) + len(test_data)


def test_load_dataset_with_train_test_split(train_data: MNIST) -> None:
    # train, test (from train_data)
    train_loader, test_loader = load_dataset(train_data, test_split=0.1)
    assert len(train_loader.dataset) + len(test_loader.dataset) == len(train_data)


def test_load_dataset_with_train_val_split(train_data: MNIST) -> None:
    # train, val (from train_data)
    train_loader, val_loader = load_dataset(train_data, val_split=0.1)
    assert len(train_loader.dataset) + len(val_loader.dataset) == len(train_data)
