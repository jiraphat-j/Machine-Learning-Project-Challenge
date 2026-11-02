# %% [markdown]
# # Task 1: Anti-cheat
#
# ตัดเฉพาะแถวที่ target หายออกจาก supervised training ไม่เติม target
# ไม่ใช้ player_id เป็น feature และให้ HistGradientBoosting จัดการ NaN ใน feature โดยตรง
# threshold 0.5 เป็นค่าเริ่มต้นที่กำหนดไว้ก่อนดู validation
#
# ใช้ baseline เดิม แยกให้รันเฉพาะงานนี้ได้ ไม่มีการเปลี่ยนโมเดลในขั้นตอนแยกไฟล์
# ไฟล์นี้ยังไม่ได้รันใหม่ ให้ตั้ง DATA_ROOT เป็นโฟลเดอร์ข้อมูลบน Kaggle ก่อน Run All
# ผลลัพธ์หลักคือ task1_predictions.csv โดยคงลำดับแถว test เดิม

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

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

# %% [markdown]
# ## ฝึกและตรวจ validation

# %%
train1 = pd.read_csv(DATA_ROOT / '01_binary_classification/train.csv')
test1 = pd.read_csv(DATA_ROOT / '01_binary_classification/test.csv')
labeled1 = train1.dropna(subset=['is_cheater'])
X1 = labeled1.drop(columns=['player_id', 'is_cheater'])
y1 = labeled1.is_cheater.astype(int)
X_train, X_valid, y_train, y_valid = train_test_split(
    X1, y1, test_size=.2, stratify=y1, random_state=SEED,
)
model1 = HistGradientBoostingClassifier(
    max_iter=180, learning_rate=.08, max_leaf_nodes=31,
    l2_regularization=1, random_state=SEED,
)
model1.fit(X_train, y_train)
prob1 = model1.predict_proba(X_valid)[:, 1]
valid1 = (prob1 >= .5).astype(int)
scores.append({'task': 1, 'metric': 'F1', 'value': f1_score(y_valid, valid1)})
scores.append({'task': 1, 'metric': 'ROC-AUC', 'value': roc_auc_score(y_valid, prob1)})
scores.append({'task': 1, 'metric': 'accuracy', 'value': accuracy_score(y_valid, valid1)})
display(pd.DataFrame(scores))
display(pd.DataFrame(confusion_matrix(y_valid, valid1),
                     index=['actual 0', 'actual 1'], columns=['pred 0', 'pred 1']))
model1.fit(X1, y1)
prediction = model1.predict(test1[X1.columns]).astype(int)

# %% [markdown]
# **การอ่านผล:** F1 วัดผลที่ threshold 0.5 ส่วน ROC-AUC วัดการเรียงความเสี่ยง
# การแบ่งแบบ stratified รักษาสัดส่วนคลาส; ยังประเมินการใช้งานข้ามเวลาไม่ได้เพราะไม่มี timestamp
# หลังวัดผลจึง refit ด้วยข้อมูลที่มี label ทั้งหมด

# %% [markdown]
# ## บันทึกผลสำหรับรวม submission

# %%
prediction_frame = pd.DataFrame({"row": np.arange(len(test1)), "prediction": prediction})
assert prediction_frame.prediction.notna().all()
assert np.isfinite(prediction_frame.prediction).all()
prediction_path = OUTPUT_DIR / "task1_predictions.csv"
prediction_frame.to_csv(prediction_path, index=False)
validation = pd.DataFrame(scores, columns=["task", "metric", "value"])
validation.to_csv(OUTPUT_DIR / "task1_validation_scores.csv", index=False)
display(validation)
print("Saved:", prediction_path.name)
print("Rows:", len(prediction_frame))

# %% [markdown]
# นำไฟล์ prediction นี้ไปใช้กับ `03_make_submission.ipynb`
# หากรันคนละ Kaggle notebook ต้อง Save Version แล้วแนบ output ของแต่ละงานเป็น Input ใน notebook รวม
