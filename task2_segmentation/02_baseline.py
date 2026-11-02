# %% [markdown]
# # Task 2: Player segmentation
#
# เข้ารหัส category ด้วย OrdinalEncoder ภายใน Pipeline โดย fit เฉพาะ training fold
# ระบุ categorical_features ให้โมเดลเพื่อไม่ตีความรหัสหมวดหมู่เป็นลำดับตัวเลข
# category ที่ไม่เคยพบจะเป็น NaN และตัด random_metric_1 ถึง random_metric_3 ตาม data card
#
# ใช้ baseline เดิม แยกให้รันเฉพาะงานนี้ได้ ไม่มีการเปลี่ยนโมเดลในขั้นตอนแยกไฟล์
# ไฟล์นี้ยังไม่ได้รันใหม่ ให้ตั้ง DATA_ROOT เป็นโฟลเดอร์ข้อมูลบน Kaggle ก่อน Run All
# ผลลัพธ์หลักคือ task2_predictions.csv โดยคงลำดับแถว test เดิม

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

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
train2 = pd.read_csv(DATA_ROOT / '02_multiclass_classification/train.csv')
test2 = pd.read_csv(DATA_ROOT / '02_multiclass_classification/test.csv')
noise = ['random_metric_1', 'random_metric_2', 'random_metric_3']
X2 = train2.drop(columns=['player_id', 'segment'] + noise)
y2 = train2.segment.astype(int)
categorical = X2.select_dtypes(include=['object', 'str']).columns.tolist()
numeric = [column for column in X2 if column not in categorical]
encoder = ColumnTransformer([
    ('numeric', 'passthrough', numeric),
    ('category', OrdinalEncoder(handle_unknown='use_encoded_value',
                                unknown_value=np.nan, encoded_missing_value=np.nan), categorical),
])
category_mask = [False] * len(numeric) + [True] * len(categorical)
model2 = make_pipeline(encoder, HistGradientBoostingClassifier(
    categorical_features=category_mask, max_iter=180, learning_rate=.08,
    l2_regularization=1, random_state=SEED,
))
X_train, X_valid, y_train, y_valid = train_test_split(
    X2, y2, test_size=.2, stratify=y2, random_state=SEED,
)
model2.fit(X_train, y_train)
valid2 = model2.predict(X_valid)
scores.append({'task': 2, 'metric': 'macro F1', 'value': f1_score(y_valid, valid2, average='macro')})
scores.append({'task': 2, 'metric': 'accuracy', 'value': accuracy_score(y_valid, valid2)})
display(pd.DataFrame(scores).query('task == 2'))
display(pd.DataFrame(confusion_matrix(y_valid, valid2)))
model2.fit(X2, y2)
prediction = model2.predict(test2[X2.columns]).astype(int)

# %% [markdown]
# **การอ่านผล:** macro F1 ให้น้ำหนักแต่ละคลาสเท่ากัน จึงควรอ่านร่วมกับ confusion matrix
# ตัวแปร player_type_tag, engagement_level และ loyalty_tier อาจถูกสร้างจากพฤติกรรมเดียวกับ target
# การมีความสัมพันธ์ยังไม่ยืนยัน leakage ต้องตรวจเวลาที่สร้าง feature ก่อนนำไปใช้งานจริง

# %% [markdown]
# ## บันทึกผลสำหรับรวม submission

# %%
prediction_frame = pd.DataFrame({"row": np.arange(len(test2)), "prediction": prediction})
assert prediction_frame.prediction.notna().all()
assert np.isfinite(prediction_frame.prediction).all()
prediction_path = OUTPUT_DIR / "task2_predictions.csv"
prediction_frame.to_csv(prediction_path, index=False)
validation = pd.DataFrame(scores, columns=["task", "metric", "value"])
validation.to_csv(OUTPUT_DIR / "task2_validation_scores.csv", index=False)
display(validation)
print("Saved:", prediction_path.name)
print("Rows:", len(prediction_frame))

# %% [markdown]
# นำไฟล์ prediction นี้ไปใช้กับ `03_make_submission.ipynb`
# หากรันคนละ Kaggle notebook ต้อง Save Version แล้วแนบ output ของแต่ละงานเป็น Input ใน notebook รวม
