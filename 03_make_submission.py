# %% [markdown]
# # รวม prediction เป็น submission
#
# อ่าน prediction จากทั้ง 5 งาน ตรวจ row order และ schema แล้วบันทึก submission โดยไม่มีการฝึกโมเดล
# ตั้ง DATA_ROOT และ PREDICTION_FILES ให้ตรงกับ Input บน Kaggle
# หากแยก Kaggle notebook แต่ละงาน ต้องแนบ output ของทั้ง 5 งานก่อนรวม
# การจับคู่แถวยังคงใช้ positional mapping ตาม baseline เดิม ต้องยืนยันกติกากับผู้จัด

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

if "__file__" in globals():
    PROJECT_ROOT = Path(__file__).resolve().parent
else:
    PROJECT_ROOT = next((path for path in [Path.cwd(), *Path.cwd().parents]
                         if (path / "sample_submission.csv").exists()), Path.cwd())
# Set DATA_ROOT and PREDICTION_FILES to the actual Kaggle Input paths.
DATA_ROOT = PROJECT_ROOT
PREDICTION_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else PROJECT_ROOT / "artifacts"
OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else PROJECT_ROOT / "submissions"
PREDICTION_FILES = {task: PREDICTION_DIR / f"task{task}_predictions.csv" for task in range(1, 6)}
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

# %% [markdown]
# ## ตรวจและรวมไฟล์

# %%
sample = pd.read_csv(DATA_ROOT / "sample_submission.csv")
submission = sample[["id"]].copy()
lengths = []
for task, path in PREDICTION_FILES.items():
    predictions = pd.read_csv(path)
    assert predictions.columns.tolist() == ["row", "prediction"], f"Task {task}: unexpected columns"
    assert len(predictions) == len(sample), f"Task {task}: wrong row count"
    assert np.array_equal(predictions.row.to_numpy(), np.arange(len(sample))), f"Task {task}: row order changed"
    submission[f"task{task}"] = predictions.prediction.to_numpy()
    lengths.append(len(predictions))
display(validate_submission(submission, sample, lengths))
submission_path = OUTPUT_DIR / "baseline_submission.csv"
submission.to_csv(submission_path, index=False)
saved = pd.read_csv(submission_path)
validate_submission(saved, sample, lengths)
display(saved.head())
print("Saved:", submission_path.name)
print("Rows:", len(saved))

# %% [markdown]
# ไฟล์สำเร็จต้องมี 25,889 แถว ไม่มี missing และคง ANS ID ตาม sample submission
# การรวมไฟล์ไม่ได้บอกว่าคะแนนดีขึ้น คะแนน 0.20131 เป็นผล baseline ที่ผู้ใช้รายงานไว้ก่อนแยกไฟล์
