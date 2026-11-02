# %% [markdown]
# # Task 5: Anomaly detection
#
# ไม่มี train หรือ label จึง fit IsolationForest บนข้อมูล unlabeled ที่ให้มา
# เพิ่ม delta และ std ระหว่าง 4 ช่วงเวลา เติม missing ด้วย median และ missing indicator
# threshold กำหนดล่วงหน้าที่ top 10% ของ anomaly score เป็นสมมติฐานเพื่อสร้าง submission
# **10% ไม่ใช่ anomaly prevalence ที่วัดจากข้อมูล** และ score ไม่ใช่ probability
# การใช้ test เพื่อ fit เป็น transductive baseline ต้องตรวจว่ากติกาการแข่งขันอนุญาตก่อนอัปโหลด
#
# ใช้ baseline เดิม แยกให้รันเฉพาะงานนี้ได้ ไม่มีการเปลี่ยนโมเดลในขั้นตอนแยกไฟล์
# ไฟล์นี้ยังไม่ได้รันใหม่ ให้ตั้ง DATA_ROOT เป็นโฟลเดอร์ข้อมูลบน Kaggle ก่อน Run All
# ผลลัพธ์หลักคือ task5_predictions.csv โดยคงลำดับแถว test เดิม

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

# Set DATA_ROOT to the dataset folder shown in Kaggle's Input panel.
if "__file__" in globals():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
else:
    PROJECT_ROOT = next((path for path in [Path.cwd(), *Path.cwd().parents]
                         if (path / "sample_submission.csv").exists()), Path.cwd())
DATA_ROOT = PROJECT_ROOT
OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else PROJECT_ROOT / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
scores = []

def anomaly_features(frame):
    result = frame.drop(columns="player_id").copy()
    for name in frame:
        if name.endswith("_1") and name[:-2] + "_4" in frame:
            prefix = name[:-2]
            columns = [f"{prefix}_{period}" for period in range(1, 5)]
            result[prefix + "_delta"] = frame[columns[-1]] - frame[columns[0]]
            result[prefix + "_std"] = frame[columns].std(axis=1)
    return result

# %% [markdown]
# ## คำนวณ anomaly score

# %%
test5 = pd.read_csv(DATA_ROOT / '05_anomaly_detection/test.csv')
X5 = anomaly_features(test5)
model5 = make_pipeline(
    SimpleImputer(strategy='median', add_indicator=True),
    IsolationForest(n_estimators=250, max_samples=2048, contamination=.10,
                    n_jobs=4, random_state=SEED),
)
model5.fit(X5)
imputed5 = model5[0].transform(X5)
anomaly_score = -model5[1].score_samples(imputed5)
threshold5 = np.quantile(anomaly_score, .90)
prediction = (anomaly_score >= threshold5).astype(int)
anomaly_output = pd.DataFrame({'row': np.arange(len(test5)), 'anomaly_score': anomaly_score,
                               'is_anomaly': prediction})
anomaly_output.to_csv(OUTPUT_DIR / 'anomaly_scores.csv', index=False)
display(anomaly_output.describe())
print('Threshold:', threshold5)
print('Flagged fraction:', prediction.mean())

# %% [markdown]
# **การอ่านผล:** สัดส่วนที่ถูก flag เป็นผลจาก threshold ที่เลือก จึงใช้ยืนยันความแม่นยำไม่ได้
# ต้องมี labeled validation หรือการตรวจสอบจากผู้เชี่ยวชาญก่อนสรุปว่าแถวใดผิดปกติจริง

# %% [markdown]
# ## บันทึกผลสำหรับรวม submission

# %%
prediction_frame = pd.DataFrame({"row": np.arange(len(test5)), "prediction": prediction})
assert prediction_frame.prediction.notna().all()
assert np.isfinite(prediction_frame.prediction).all()
prediction_path = OUTPUT_DIR / "task5_predictions.csv"
prediction_frame.to_csv(prediction_path, index=False)
validation = pd.DataFrame(scores, columns=["task", "metric", "value"])
validation.to_csv(OUTPUT_DIR / "task5_validation_scores.csv", index=False)
display(validation)
print("Saved:", prediction_path.name)
print("Rows:", len(prediction_frame))

# %% [markdown]
# นำไฟล์ prediction นี้ไปใช้กับ `03_make_submission.ipynb`
# หากรันคนละ Kaggle notebook ต้อง Save Version แล้วแนบ output ของแต่ละงานเป็น Input ใน notebook รวม
