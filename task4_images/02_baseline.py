# %% [markdown]
# # Task 4: Images
#
# ใช้ภาพ train ที่มีไฟล์จริงและอ่านได้เท่านั้น แปลงเป็น pixel 24×12 และ color histogram
# แล้วเทียบ ExtraTrees กับ SVM บน CPU ไม่ใช้ pretrained model หรือดาวน์โหลดภาพเพิ่ม
# mapping อิงชื่อเกมใน data card: 7-11k=0, fiveN=1, freefried=2, pube=3, roblock=4
# คะแนน validation สะท้อนเฉพาะภาพที่มีอยู่ ซึ่งอาจไม่เป็นตัวแทนของภาพที่ขาด
#
# ใช้ baseline เดิม แยกให้รันเฉพาะงานนี้ได้ ไม่มีการเปลี่ยนโมเดลในขั้นตอนแยกไฟล์
# ไฟล์นี้ยังไม่ได้รันใหม่ ให้ตั้ง DATA_ROOT เป็นโฟลเดอร์ข้อมูลบน Kaggle ก่อน Run All
# ผลลัพธ์หลักคือ task4_predictions.csv โดยคงลำดับแถว test เดิม

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display
import hashlib
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

CLASS_MAP = {"7-11k": 0, "fiveN": 1, "freefried": 2, "pube": 3, "roblock": 4}

def image_inventory(base, frame, split):
    rows = []
    for image_id in frame.id:
        path = base / split / f"{image_id}.jpg"
        row = {"id": image_id, "exists": path.exists(), "readable": False,
               "width": np.nan, "height": np.nan, "mode": "", "error": "", "sha256": ""}
        if row["exists"]:
            try:
                with Image.open(path) as image:
                    image.load()
                    row.update(readable=True, width=image.width, height=image.height, mode=image.mode)
                row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            except (OSError, ValueError) as error:
                row["error"] = type(error).__name__
        rows.append(row)
    return pd.DataFrame(rows)

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

# %% [markdown]
# ## ฝึกและตรวจ validation

# %%
image_base = DATA_ROOT / '04_image_classification/task4_image_classification'
train4 = pd.read_csv(image_base / 'train.csv')
test4 = pd.read_csv(image_base / 'test.csv')
inventory4 = image_inventory(image_base, train4, 'train')
available_ids = set(inventory4.loc[inventory4.readable, 'id'])
available4 = train4.loc[train4.id.isin(available_ids)].reset_index(drop=True)
assert available4['class'].isin(CLASS_MAP).all()
assert set(available4['class']) == set(CLASS_MAP)
print(f'Available train images: {len(available4):,} / {len(train4):,}')
display(pd.crosstab(available4['class'], columns='available images'))
train_paths = [image_base / 'train' / f'{image_id}.jpg' for image_id in available4.id]
test_paths = [image_base / 'test' / f'{image_id}.jpg' for image_id in test4.id]
X4 = extract_images(train_paths)
y4 = available4['class'].map(CLASS_MAP).to_numpy()
X_train, X_valid, y_train, y_valid = train_test_split(
    X4, y4, test_size=.2, stratify=y4, random_state=SEED,
)
image_candidates = {
    'ExtraTrees': ExtraTreesClassifier(n_estimators=200, max_features=.4,
                                       class_weight='balanced', n_jobs=4, random_state=SEED),
    'SVM': make_pipeline(StandardScaler(), SVC(C=10, class_weight='balanced')),
}
image_scores = []
for name, candidate in image_candidates.items():
    candidate.fit(X_train, y_train)
    candidate_prediction = candidate.predict(X_valid)
    image_scores.append({'model': name,
                         'macro_f1': f1_score(y_valid, candidate_prediction, average='macro'),
                         'accuracy': accuracy_score(y_valid, candidate_prediction)})
image_scores = pd.DataFrame(image_scores)
display(image_scores)
image_scores.to_csv(OUTPUT_DIR / 'image_model_scores.csv', index=False)
best_image_model = image_scores.loc[image_scores.macro_f1.idxmax(), 'model']
model4 = image_candidates[best_image_model]
print('Selected by validation macro F1:', best_image_model)
valid4 = model4.predict(X_valid)
scores.append({'task': 4, 'metric': 'macro F1', 'value': f1_score(y_valid, valid4, average='macro')})
scores.append({'task': 4, 'metric': 'accuracy', 'value': accuracy_score(y_valid, valid4)})
display(pd.DataFrame(scores).query('task == 4'))
display(pd.DataFrame(confusion_matrix(y_valid, valid4)))
model4.fit(X4, y4)
prediction = model4.predict(extract_images(test_paths)).astype(int)

# %% [markdown]
# **การอ่านผล:** นี่เป็น image baseline จาก feature พื้นฐาน ยังไม่ได้ใช้ transfer learning
# เลือกโมเดลด้วย macro F1 บน validation ชุดเดิม คะแนนของโมเดลที่เลือกอาจสูงเกินข้อมูลใหม่เล็กน้อย
# random split อาจให้คะแนนสูงเกินจริงหากมี screenshot คล้ายกันจากคลิปเดียวกัน
# รอบถัดไปควรกู้ภาพที่ขาด ตรวจ near duplicates และแบ่ง validation ตามแหล่งภาพเมื่อมีข้อมูล

# %% [markdown]
# ## บันทึกผลสำหรับรวม submission

# %%
prediction_frame = pd.DataFrame({"row": np.arange(len(test4)), "prediction": prediction})
assert prediction_frame.prediction.notna().all()
assert np.isfinite(prediction_frame.prediction).all()
prediction_path = OUTPUT_DIR / "task4_predictions.csv"
prediction_frame.to_csv(prediction_path, index=False)
validation = pd.DataFrame(scores, columns=["task", "metric", "value"])
validation.to_csv(OUTPUT_DIR / "task4_validation_scores.csv", index=False)
display(validation)
print("Saved:", prediction_path.name)
print("Rows:", len(prediction_frame))

# %% [markdown]
# นำไฟล์ prediction นี้ไปใช้กับ `03_make_submission.ipynb`
# หากรันคนละ Kaggle notebook ต้อง Save Version แล้วแนบ output ของแต่ละงานเป็น Input ใน notebook รวม
