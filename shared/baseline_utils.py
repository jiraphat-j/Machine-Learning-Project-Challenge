from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image


CLASS_MAP = {"7-11k": 0, "fiveN": 1, "freefried": 2, "pube": 3, "roblock": 4}


def image_features(path):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        pixels = np.asarray(rgb.resize((24, 12)), dtype=np.float32).ravel() / 255
        small = np.asarray(rgb.resize((64, 32)))
        hist = np.concatenate([np.histogram(small[:, :, channel], bins=16, range=(0, 256))[0]
                               for channel in range(3)]).astype(np.float32)
        hist /= 64 * 32
    return np.concatenate([pixels, hist])


def extract_images(paths):
    # Preserve CSV row order. Do not sort filenames before prediction.
    with ThreadPoolExecutor(max_workers=4) as pool:
        return np.vstack(list(pool.map(image_features, paths)))


def anomaly_features(frame):
    result = frame.drop(columns="player_id").copy()
    for name in frame:
        if name.endswith("_1") and name[:-2] + "_4" in frame:
            prefix = name[:-2]
            columns = [f"{prefix}_{period}" for period in range(1, 5)]
            result[prefix + "_delta"] = frame[columns[-1]] - frame[columns[0]]
            result[prefix + "_std"] = frame[columns].std(axis=1)
    return result


def validate_submission(submission, sample, test_lengths):
    assert list(submission.columns) == ["id", "task1", "task2", "task3", "task4", "task5"]
    assert len(submission) == len(sample) == 25889
    assert all(length == len(sample) for length in test_lengths)
    assert submission.id.equals(sample.id), "Submission IDs or row order changed."
    assert submission.id.is_unique
    assert submission.notna().all().all(), "Missing prediction."
    assert np.isfinite(submission.drop(columns="id").to_numpy()).all()
    for column, allowed in {"task1": {0, 1}, "task2": {0, 1, 2, 3},
                            "task4": {0, 1, 2, 3, 4}, "task5": {0, 1}}.items():
        assert set(submission[column].unique()).issubset(allowed), column
        assert (submission[column] == submission[column].astype(int)).all(), column
    assert submission.task3.ge(0).all()
    return pd.DataFrame({"missing": submission.isna().sum(),
                         "unique": submission.nunique(), "dtype": submission.dtypes.astype(str)})
