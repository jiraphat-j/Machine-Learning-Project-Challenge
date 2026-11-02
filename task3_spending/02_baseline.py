# %% [markdown]
# # Task 3: Spending
#
# ใช้ HistGradientBoostingRegressor บน target หน่วย THB โดยตรง เพื่อสร้าง RMSE baseline
# เปรียบเทียบกับค่าคงที่ mean ของ training fold และบังคับ prediction ไม่ต่ำกว่า 0
# ยังไม่เลือก log target เพราะไม่ทราบ metric ทางการ
#
# ใช้ baseline เดิม แยกให้รันเฉพาะงานนี้ได้ ไม่มีการเปลี่ยนโมเดลในขั้นตอนแยกไฟล์
# ไฟล์นี้ยังไม่ได้รันใหม่ ให้ตั้ง DATA_ROOT เป็นโฟลเดอร์ข้อมูลบน Kaggle ก่อน Run All
# ผลลัพธ์หลักคือ task3_predictions.csv โดยคงลำดับแถว test เดิม

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
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
train3 = pd.read_csv(DATA_ROOT / '03_regression/train.csv')
test3 = pd.read_csv(DATA_ROOT / '03_regression/test.csv')
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
prediction = np.maximum(model3.predict(test3[X3.columns]), 0)

# %% [markdown]
# **การอ่านผล:** RMSE ไวต่อผู้เล่นที่ใช้จ่ายสูง ส่วน MAE แสดงขนาดความผิดพลาดเฉลี่ย
# ควรตรวจแยกกลุ่มไม่ใช้จ่ายและกลุ่มใช้จ่ายสูงในการพัฒนารอบถัดไป
# segment ที่มีในไฟล์เป็น input เดิม ไม่แทนด้วย Task 2 prediction เพราะไม่ได้ยืนยันว่าเป็น snapshot เดียวกัน

# %% [markdown]
# ## บันทึกผลสำหรับรวม submission

# %%
prediction_frame = pd.DataFrame({"row": np.arange(len(test3)), "prediction": prediction})
assert prediction_frame.prediction.notna().all()
assert np.isfinite(prediction_frame.prediction).all()
prediction_path = OUTPUT_DIR / "task3_predictions.csv"
prediction_frame.to_csv(prediction_path, index=False)
validation = pd.DataFrame(scores, columns=["task", "metric", "value"])
validation.to_csv(OUTPUT_DIR / "task3_validation_scores.csv", index=False)
display(validation)
print("Saved:", prediction_path.name)
print("Rows:", len(prediction_frame))

# %% [markdown]
# นำไฟล์ prediction นี้ไปใช้กับ `03_make_submission.ipynb`
# หากรันคนละ Kaggle notebook ต้อง Save Version แล้วแนบ output ของแต่ละงานเป็น Input ใน notebook รวม
