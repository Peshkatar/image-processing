from torch.utils.data import DataLoader, Dataset, random_split


def load_dataset(
    train_data: Dataset,
    test_data: Dataset | None = None,
    val_data: Dataset | None = None,
    batch_size: int = 64,
    train_split: float = 0,
    test_split: float = 0,
    val_split: float = 0,
) -> tuple[DataLoader, ...]:
    train_split = train_split if train_split > 0 else 1 - test_split - val_split

    if train_split + test_split + val_split != 1:
        raise ValueError(
            "`train_split` + `test_split` + `val_split` must be equal to 1"
        )

    if test_data is not None and test_split != 0:
        raise ValueError("`test_data` is provided but `test_split` is not 0")

    if val_data is not None and val_split != 0:
        raise ValueError("`val_data` is provided but `val_split` is not 0")

    # both provided for splitting
    if test_data is None and val_data is None and test_split > 0 and val_split > 0:
        train_data, test_data, val_data = random_split(
            train_data, [train_split, test_split, val_split]
        )
    # test split only
    elif test_data is None and test_split > 0:
        train_data, test_data = random_split(train_data, [train_split, test_split])
    # val split only
    elif val_data is None and val_split > 0:
        train_data, val_data = random_split(train_data, [train_split, val_split])

    return tuple(
        DataLoader(d, batch_size=batch_size, shuffle=True)
        for d in [train_data, test_data, val_data]
        if d is not None
    )
