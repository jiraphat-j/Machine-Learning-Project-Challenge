from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))


md("""
# Baseline และ submission

เป้าหมายคือสร้างคำตอบครบทั้ง 5 งานภายใน 1 ชั่วโมง โดยใช้โมเดล CPU และ seed 42
Notebook นี้แยกจาก EDA: การฝึกโมเดลทั้งหมดอยู่ที่นี่

ยังไม่พบ evaluation metric และกติกาการรวมคะแนนทางการในไฟล์ที่ได้รับ
จึงใช้ F1/ROC-AUC สำหรับ Task 1, macro F1 สำหรับ Task 2 และ Task 4,
RMSE/MAE สำหรับ Task 3 เป็น **local proxy** ไม่ใช่คะแนน Kaggle
แบ่ง train/validation 80/20 ก่อน fit preprocessing และไม่ใช้ test เลือกโมเดล
Task 5 ไม่มี label และเป็น unsupervised fit บนชุด test ที่ให้มา จึงไม่มี validation score

การจับคู่คำตอบใช้แถวที่ i ของ test แต่ละงานตรงกับแถวที่ i ของ sample submission
ไฟล์ sample ใช้ ANS ID ซึ่งไม่เชื่อมกับ player_id/image id จึงต้องยืนยันกติกานี้กับผู้จัด
ห้าม merge test ข้ามงานด้วย player_id เพราะลำดับข้อมูลต่างกัน
""")
code("""
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    ExtraTreesClassifier, IsolationForest,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, mean_absolute_error,
    root_mean_squared_error, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
            if (path / 'sample_submission.csv').exists())
sys.path.insert(0, str(ROOT))
from shared.eda_utils import setup_plot, image_inventory
from shared.baseline_utils import (
    CLASS_MAP, extract_images, anomaly_features, validate_submission,
)

setup_plot()
SEED = 42
OUTPUT = ROOT / 'submissions'
ARTIFACTS = ROOT / 'artifacts'
OUTPUT.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)
scores = []
predictions = {}
test_lengths = []
started = time.perf_counter()
""")
md("""
## Task 1: Anti-cheat

ตัดเฉพาะแถวที่ target หายออกจาก supervised training ไม่เติม target
ไม่ใช้ player_id เป็น feature และให้ HistGradientBoosting จัดการ NaN ใน feature โดยตรง
threshold 0.5 เป็นค่าเริ่มต้นที่กำหนดไว้ก่อนดู validation
""")
code("""
train1 = pd.read_csv(ROOT / '01_binary_classification/train.csv')
test1 = pd.read_csv(ROOT / '01_binary_classification/test.csv')
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
predictions[1] = model1.predict(test1[X1.columns]).astype(int)
test_lengths.append(len(test1))
""")
md("""
**การอ่านผล:** F1 วัดผลที่ threshold 0.5 ส่วน ROC-AUC วัดการเรียงความเสี่ยง
การแบ่งแบบ stratified รักษาสัดส่วนคลาส; ยังประเมินการใช้งานข้ามเวลาไม่ได้เพราะไม่มี timestamp
หลังวัดผลจึง refit ด้วยข้อมูลที่มี label ทั้งหมด
""")
md("""
## Task 2: Player segmentation

เข้ารหัส category ด้วย OrdinalEncoder ภายใน Pipeline โดย fit เฉพาะ training fold
ระบุ categorical_features ให้โมเดลเพื่อไม่ตีความรหัสหมวดหมู่เป็นลำดับตัวเลข
category ที่ไม่เคยพบจะเป็น NaN และตัด random_metric_1 ถึง random_metric_3 ตาม data card
""")
code("""
train2 = pd.read_csv(ROOT / '02_multiclass_classification/train.csv')
test2 = pd.read_csv(ROOT / '02_multiclass_classification/test.csv')
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
predictions[2] = model2.predict(test2[X2.columns]).astype(int)
test_lengths.append(len(test2))
""")
md("""
**การอ่านผล:** macro F1 ให้น้ำหนักแต่ละคลาสเท่ากัน จึงควรอ่านร่วมกับ confusion matrix
ตัวแปร player_type_tag, engagement_level และ loyalty_tier อาจถูกสร้างจากพฤติกรรมเดียวกับ target
การมีความสัมพันธ์ยังไม่ยืนยัน leakage ต้องตรวจเวลาที่สร้าง feature ก่อนนำไปใช้งานจริง
""")
md("""
## Task 3: Spending

ใช้ HistGradientBoostingRegressor บน target หน่วย THB โดยตรง เพื่อสร้าง RMSE baseline
เปรียบเทียบกับค่าคงที่ mean ของ training fold และบังคับ prediction ไม่ต่ำกว่า 0
ยังไม่เลือก log target เพราะไม่ทราบ metric ทางการ
""")
code("""
train3 = pd.read_csv(ROOT / '03_regression/train.csv')
test3 = pd.read_csv(ROOT / '03_regression/test.csv')
X3 = train3.drop(columns=['player_id', 'spending_30d'])
y3 = train3.spending_30d
X_train, X_valid, y_train, y_valid = train_test_split(
    X3, y3, test_size=.2, random_state=SEED,
)
model3 = HistGradientBoostingRegressor(
    loss='squared_error', max_iter=250, learning_rate=.06,
    max_leaf_nodes=31, l2_regularization=5, random_state=SEED,
)
model3.fit(X_train, y_train)
valid3 = np.maximum(model3.predict(X_valid), 0)
scores.append({'task': 3, 'metric': 'RMSE (THB)', 'value': root_mean_squared_error(y_valid, valid3)})
scores.append({'task': 3, 'metric': 'MAE (THB)', 'value': mean_absolute_error(y_valid, valid3)})
scores.append({'task': 3, 'metric': 'mean baseline RMSE',
               'value': root_mean_squared_error(y_valid, np.full(len(y_valid), y_train.mean()))})
display(pd.DataFrame(scores).query('task == 3'))
display(pd.DataFrame({'actual': y_valid, 'prediction': valid3}).describe())
model3.fit(X3, y3)
predictions[3] = np.maximum(model3.predict(test3[X3.columns]), 0)
test_lengths.append(len(test3))
""")
md("""
**การอ่านผล:** RMSE ไวต่อผู้เล่นที่ใช้จ่ายสูง ส่วน MAE แสดงขนาดความผิดพลาดเฉลี่ย
ควรตรวจแยกกลุ่มไม่ใช้จ่ายและกลุ่มใช้จ่ายสูงในการพัฒนารอบถัดไป
segment ที่มีในไฟล์เป็น input เดิม ไม่แทนด้วย Task 2 prediction เพราะไม่ได้ยืนยันว่าเป็น snapshot เดียวกัน
""")
md("""
## Task 4: Images

ใช้ภาพ train ที่มีไฟล์จริงและอ่านได้เท่านั้น แปลงเป็น pixel 24×12 และ color histogram
แล้วเทียบ ExtraTrees กับ SVM บน CPU ไม่ใช้ pretrained model หรือดาวน์โหลดภาพเพิ่ม
mapping อิงชื่อเกมใน data card: 7-11k=0, fiveN=1, freefried=2, pube=3, roblock=4
คะแนน validation สะท้อนเฉพาะภาพที่มีอยู่ ซึ่งอาจไม่เป็นตัวแทนของภาพที่ขาด
""")
code("""
image_base = ROOT / '04_image_classification/task4_image_classification'
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
image_scores.to_csv(ARTIFACTS / 'image_model_scores.csv', index=False)
best_image_model = image_scores.loc[image_scores.macro_f1.idxmax(), 'model']
model4 = image_candidates[best_image_model]
print('Selected by validation macro F1:', best_image_model)
valid4 = model4.predict(X_valid)
scores.append({'task': 4, 'metric': 'macro F1', 'value': f1_score(y_valid, valid4, average='macro')})
scores.append({'task': 4, 'metric': 'accuracy', 'value': accuracy_score(y_valid, valid4)})
display(pd.DataFrame(scores).query('task == 4'))
display(pd.DataFrame(confusion_matrix(y_valid, valid4)))
model4.fit(X4, y4)
predictions[4] = model4.predict(extract_images(test_paths)).astype(int)
test_lengths.append(len(test4))
""")
md("""
**การอ่านผล:** นี่เป็น image baseline จาก feature พื้นฐาน ยังไม่ได้ใช้ transfer learning
เลือกโมเดลด้วย macro F1 บน validation ชุดเดิม คะแนนของโมเดลที่เลือกอาจสูงเกินข้อมูลใหม่เล็กน้อย
random split อาจให้คะแนนสูงเกินจริงหากมี screenshot คล้ายกันจากคลิปเดียวกัน
รอบถัดไปควรกู้ภาพที่ขาด ตรวจ near duplicates และแบ่ง validation ตามแหล่งภาพเมื่อมีข้อมูล
""")
md("""
## Task 5: Anomaly detection

ไม่มี train หรือ label จึง fit IsolationForest บนข้อมูล unlabeled ที่ให้มา
เพิ่ม delta และ std ระหว่าง 4 ช่วงเวลา เติม missing ด้วย median และ missing indicator
threshold กำหนดล่วงหน้าที่ top 10% ของ anomaly score เป็นสมมติฐานเพื่อสร้าง submission
**10% ไม่ใช่ anomaly prevalence ที่วัดจากข้อมูล** และ score ไม่ใช่ probability
การใช้ test เพื่อ fit เป็น transductive baseline ต้องตรวจว่ากติกาการแข่งขันอนุญาตก่อนอัปโหลด
""")
code("""
test5 = pd.read_csv(ROOT / '05_anomaly_detection/test.csv')
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
predictions[5] = (anomaly_score >= threshold5).astype(int)
test_lengths.append(len(test5))
anomaly_output = pd.DataFrame({'player_id': test5.player_id, 'anomaly_score': anomaly_score,
                               'is_anomaly': predictions[5]})
anomaly_output.to_csv(ARTIFACTS / 'anomaly_scores.csv', index=False)
display(anomaly_output.describe())
print('Threshold:', threshold5)
print('Flagged fraction:', predictions[5].mean())
""")
md("""
**การอ่านผล:** สัดส่วนที่ถูก flag เป็นผลจาก threshold ที่เลือก จึงใช้ยืนยันความแม่นยำไม่ได้
ต้องมี labeled validation หรือการตรวจสอบจากผู้เชี่ยวชาญก่อนสรุปว่าแถวใดผิดปกติจริง
""")
md("""
## รวมไฟล์และตรวจคำตอบ

รักษา ANS ID และลำดับแถวเดิม ตรวจจำนวนแถว, missing, finite values,
ช่วงของ class และ spending ที่ไม่ติดลบ บันทึก local validation แยกจาก prediction
""")
code("""
sample = pd.read_csv(ROOT / 'sample_submission.csv')
submission = sample.copy()
for task in range(1, 6):
    submission[f'task{task}'] = predictions[task]
    pd.DataFrame({'row': np.arange(len(sample)), 'prediction': predictions[task]}).to_csv(
        ARTIFACTS / f'task{task}_predictions.csv', index=False,
    )
display(validate_submission(submission, sample, test_lengths))
submission_path = OUTPUT / 'baseline_submission.csv'
submission.to_csv(submission_path, index=False)
saved = pd.read_csv(submission_path)
validate_submission(saved, sample, test_lengths)
validation = pd.DataFrame(scores)
validation.to_csv(ARTIFACTS / 'validation_scores.csv', index=False)
display(validation)
display(saved.head())
print('Saved:', submission_path)
print(f'Elapsed: {(time.perf_counter() - started) / 60:.2f} minutes')
""")
md("""
ไฟล์ผลลัพธ์อยู่ที่ `submissions/baseline_submission.csv` และผ่านการอ่านกลับเพื่อตรวจ schema
ยังไม่ได้อัปโหลดขึ้น Kaggle จึงยังไม่มี public/private leaderboard score
การตรวจโครงสร้างไม่สามารถยืนยันกติกา row mapping, metric และ transductive learning แทนผู้จัดได้
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(notebook, ROOT / '02_baseline_submission.ipynb')
print('Created 02_baseline_submission.ipynb')
