import numpy as numpy
from pathlib import Path 
BASE = Path("har_raw/UCI HAR Dataset")

SIGNALS = [
    "total_acc_x", "total_acc_y", "total_acc_z",
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
]

LABEL_MAP = {
    1: "walking",
    2: "walking_upstairs",
    3: "walking_downstairs",
    4: "sitting",
    5: "standing",
    6: "laying",
}

VAL_SUBJECTS = {1, 8, 21, 27}

def load_split(split):
    signal_dir = BASE / split / "Inertial Signals"

    channel_arrays = []
    for signal_name in SIGNALS:
        file_path = signal_dir / f"{signal_name}_{split}.txt"
        arr=numpy.loadtxt(file_path)
        channel_arrays.append(arr)

    X = numpy.stack(channel_arrays, axis = -1)

    y_raw = numpy.loadtxt(BASE / split / f"y_{split}.txt").astype(int)
    y = y_raw - 1

    subjects = numpy.loadtxt(BASE / split / f"subject_{split}.txt").astype(int)

    return X, y, subjects

def load_train_val_test():
    X_train_full, y_train_full, subj_train_full = load_split("train")
    X_test, y_test, _ = load_split("test")

    val_mask = numpy.isin(subj_train_full, list(VAL_SUBJECTS))
    train_mask = ~val_mask

    X_train, y_train = X_train_full[train_mask], y_train_full[train_mask]
    X_val, y_val = X_train_full[val_mask], y_train_full[val_mask]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

if __name__ == "__main__":
    # X_train, y_train, subj_train = load_split("train")
    # X_test, y_test, subj_test = load_split("test")

    # print("Train X shape:", X_train.shape)
    # print("Train y shape:", y_train.shape)
    # print("Unique labels:", sorted(set(y_train)))
    # print("Unique train subjects:", sorted(set(subj_train)))
    # print("Test X shape:", X_test.shape)
    # print("Unique test subjects:", sorted(set(subj_test)))

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_train_val_test()

    print("Train:", X_train.shape, y_train.shape)
    print("Val: ", X_val.shape, y_val.shape)
    print("Test: ", X_test.shape, y_test.shape)

    import collections
    print("Train label counts:", collections.Counter(y_train.tolist()))
    print("Val label counts:", collections.Counter(y_val.tolist()))

