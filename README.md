# CPE342 Player Intelligence

โปรเจกต์ EDA และ baseline สำหรับ 5 งาน ใช้โค้ด Python แบบตรงไปตรงมา และเขียนข้อค้นพบภาษาไทยไว้ใน notebook
การฝึกโมเดลแยกจาก EDA ทุกงาน ไม่มีไฟล์รายงาน findings แยก

## เปิดไฟล์ไหนก่อน

1. [00_data_audit.ipynb](00_data_audit.ipynb): ตรวจข้อมูลทุกงาน แผน 1 ชั่วโมง และสรุปข้ามงาน
2. เลือก task ที่ต้องการพัฒนา แต่ละงานมี EDA, Python code และ notebook แยกกัน:

| งาน | EDA | Python code | Notebook |
|---|---|---|---|
| 1 Anti-cheat | [01_eda.ipynb](task1_anti_cheat/01_eda.ipynb) | [02_baseline.py](task1_anti_cheat/02_baseline.py) | [02_baseline.ipynb](task1_anti_cheat/02_baseline.ipynb) |
| 2 Segmentation | [01_eda.ipynb](task2_segmentation/01_eda.ipynb) | [02_baseline.py](task2_segmentation/02_baseline.py) | [02_baseline.ipynb](task2_segmentation/02_baseline.ipynb) |
| 3 Spending | [01_eda.ipynb](task3_spending/01_eda.ipynb) | [02_baseline.py](task3_spending/02_baseline.py) | [02_baseline.ipynb](task3_spending/02_baseline.ipynb) |
| 4 Images | [01_eda.ipynb](task4_images/01_eda.ipynb) | [02_baseline.py](task4_images/02_baseline.py) | [02_baseline.ipynb](task4_images/02_baseline.ipynb) |
| 5 Anomaly | [01_eda.ipynb](task5_anomaly/01_eda.ipynb) | [02_baseline.py](task5_anomaly/02_baseline.py) | [02_baseline.ipynb](task5_anomaly/02_baseline.ipynb) |

3. [03_make_submission.py](03_make_submission.py) หรือ [03_make_submission.ipynb](03_make_submission.ipynb): อ่าน prediction ของทั้ง 5 งาน ตรวจและรวมไฟล์ โดยไม่มีการฝึกโมเดล
4. [submissions/baseline_submission.csv](submissions/baseline_submission.csv): ไฟล์คำตอบครบทั้ง 5 งาน

แต่ละ task รันได้แยกกันและไม่มีการ import โค้ดจาก task อื่น ผลลัพธ์ชื่อ `task1_predictions.csv` ถึง `task5_predictions.csv`
ไฟล์ `.py` เป็นโค้ดหลักและแบ่ง cell ด้วย `# %%` ส่วน `.ipynb` สร้างจากโค้ดเดียวกัน
แก้ `.py` แล้วรัน `scripts/create_baseline.py` เพื่ออัปเดต notebook

[02_baseline_submission.ipynb](02_baseline_submission.ipynb) เก็บโค้ดและผล baseline รอบแรกไว้สำหรับอ้างอิง
ผู้ใช้รายงาน Kaggle score **0.20131** รอบการแยกไฟล์นี้ไม่ได้ฝึกโมเดลใหม่หรือปรับคะแนน

## แผนรอบ baseline แรกภายใน 1 ชั่วโมง

| เวลา | งาน |
|---|---|
| 0–10 นาที | ตรวจ schema, missing label, ไฟล์ภาพ และ submission |
| 10–25 นาที | EDA คุณภาพข้อมูล distribution และความสัมพันธ์ |
| 25–45 นาที | ฝึก baseline บน CPU และวัด local validation |
| 45–60 นาที | ตรวจ output, class mapping, จำนวนแถว และไฟล์ที่บันทึก |

## โมเดลและ validation

| งาน | Baseline | Local validation |
|---|---|---|
| 1 | HistGradientBoostingClassifier | stratified 80/20; F1, ROC-AUC, accuracy |
| 2 | OrdinalEncoder + categorical HistGradientBoostingClassifier | stratified 80/20; macro F1, accuracy |
| 3 | HistGradientBoostingRegressor | random 80/20; RMSE, MAE เทียบ mean predictor |
| 4 | pixel 24×12 + color histogram; เทียบ ExtraTrees กับ scaled SVM | stratified 80/20 บนภาพที่อ่านได้; เลือกด้วย macro F1 |
| 5 | delta/std + median imputation + IsolationForest | ไม่มี label จึงยังวัดความแม่นยำไม่ได้ |

ใช้ seed 42, fit preprocessing หลังแบ่ง validation และ refit ด้วย train ที่ใช้ได้หลังวัดผล
ยังไม่พบ metric ทางการในไฟล์ที่ให้มา คะแนนที่บันทึกจึงเป็น local proxy
ผลวัดรอบแรกอยู่ใน `02_baseline_submission.ipynb` และ `artifacts/validation_scores.csv`
เมื่อรันไฟล์แยก จะได้ `task{n}_validation_scores.csv` ของงานนั้น; Task 5 มีเฉพาะหัวตารางเพราะไม่มี label
Task 4 เทียบโมเดลบน holdout ชุดเดียว จึงใช้คะแนนนี้เลือก baseline เท่านั้น ยังไม่ใช่การประเมินอิสระรอบสุดท้าย

## วิธีรัน

แนะนำ Python 3.12 เลือก interpreter ที่ติดตั้ง packages ใน `requirements.txt` แล้ว Run All ใน notebook
ไม่ต้องดาวน์โหลดโมเดลหรือใช้ GPU

หลัง clone repo ให้ติดตั้ง dependency ด้วย `python -m pip install -r requirements.txt`
ข้อมูลดิบและภาพขนาดใหญ่ไม่อยู่ใน Git ให้นำข้อมูลจากชุดโจทย์มาไว้ตามโครงสร้างนี้ก่อนรันใหม่:

```text
01_binary_classification/{train.csv,test.csv}
02_multiclass_classification/{train.csv,test.csv}
03_regression/{train.csv,test.csv}
04_image_classification/task4_image_classification/{train.csv,test.csv}
04_image_classification/task4_image_classification/train/{id}.jpg
04_image_classification/task4_image_classification/test/{id}.jpg
05_anomaly_detection/test.csv
```

EDA และ notebook baseline เดิมมี output และ findings อยู่แล้ว จึงเปิดอ่านผลได้โดยไม่ต้องมีข้อมูลดิบ
baseline ที่แยกใหม่ยังไม่ได้ฝึกซ้ำและยังไม่มี output การฝึกใหม่
ไฟล์ `sample_submission.csv`, data cards, คะแนน validation และ baseline submission อยู่ใน repo
inventory ภาพ, player IDs, image hashes และ prediction รายแถวใน artifacts เก็บไว้ในเครื่อง
output ตัวอย่างใน EDA ไม่แสดง player/image identifiers

สำหรับเครื่องนี้มี Python 3.12 และ dependency ที่จำเป็นอยู่แล้ว รันจากโฟลเดอร์โปรเจกต์ด้วย PowerShell:

```powershell
$projectPython = 'C:\Users\Feast\AppData\Local\Programs\Python\Python312\python.exe'
& $projectPython scripts/execute_notebooks.py 00_data_audit.ipynb task1_anti_cheat/01_eda.ipynb task2_segmentation/01_eda.ipynb task3_spending/01_eda.ipynb task4_images/01_eda.ipynb task5_anomaly/01_eda.ipynb
& $projectPython task1_anti_cheat/02_baseline.py
& $projectPython task2_segmentation/02_baseline.py
& $projectPython task3_spending/02_baseline.py
& $projectPython task4_images/02_baseline.py
& $projectPython task5_anomaly/02_baseline.py
& $projectPython 03_make_submission.py
& $projectPython scripts/verify_outputs.py
```

เลือกเรียกเฉพาะ task ที่ต้องการแก้ได้ โดยใช้ prediction เดิมของ task อื่นในเครื่อง
ทุกไฟล์รวมต้องมาจากข้อมูล test รุ่นเดียวกันและลำดับแถวเดิม

## การรันบน Kaggle

1. นำ `02_baseline.ipynb` ของ task นั้นไปเปิดบน Kaggle หรือคัดลอกโค้ดจาก `.py`
2. เพิ่มข้อมูลเป็น Input แล้วตั้ง `DATA_ROOT` ใน cell แรกเป็น path จริง เช่น `Path("/kaggle/input/your-dataset")` โดยโฟลเดอร์นี้ต้องมีโครงสร้างข้อมูลตามด้านบน
3. Run All ของงานนั้น ผลลัพธ์จะอยู่ใน `/kaggle/working/task{n}_predictions.csv`
4. หากแยกเป็นหลาย Kaggle notebooks ให้ Save Version ของแต่ละงาน และแนบ output ทั้ง 5 งานเป็น Input ใน `03_make_submission.ipynb`
5. ตั้ง `DATA_ROOT` และ `PREDICTION_FILES` ในไฟล์รวมให้ตรงกับ path จริงของแต่ละ Input แล้ว Run All

`/kaggle/working` ของคนละ notebook ไม่ได้ใช้ร่วมกัน จึงต้องแนบ output หรือกำหนด path ให้ครบก่อนรวม
ไฟล์รวมตรวจจำนวนและลำดับ `row` ก่อนเขียน CSV หากแถวหาย ซ้ำ สลับ หรือ class ผิดช่วงจะหยุดด้วย error

ตัวรันใช้ IPython รัน cell จริงตามลำดับ เริ่ม namespace ใหม่ต่อ notebook และเก็บตาราง/PNG output ลง `.ipynb`
วิธีนี้ไม่ต้องสร้าง local Jupyter socket ซึ่งค้างในสภาพแวดล้อมที่ใช้เตรียมงานครั้งนี้
Notebook ยังเปิดและ Run All ใน Jupyter หรือ VS Code ได้ตามปกติ

ถ้าข้อมูลเปลี่ยน ให้สร้าง Markdown findings จาก snapshot ใหม่ก่อนรัน notebook:

```powershell
& $projectPython scripts/audit_images.py
& $projectPython scripts/create_eda.py
& $projectPython scripts/create_baseline.py
```

จากนั้นรัน notebooks ตามคำสั่งด้านบน สคริปต์ `create_eda.py` สร้างรายงาน EDA ใหม่ ส่วน `create_baseline.py` แปลงไฟล์ `.py` ที่แยกแล้วเป็น `.ipynb` และล้าง output เดิมของ notebook ที่สร้าง
`02_baseline_submission.ipynb` ที่เก็บผลรอบแรกจะไม่ถูกสร้างทับ
ค่าที่อยู่ใน Markdown เป็นข้อค้นพบของ snapshot ที่สร้างรายงาน ไม่อัปเดตอัตโนมัติจาก Run All เพียงอย่างเดียว

## ข้อจำกัดที่ต้องอ่านก่อนส่ง

- Task 1: target หาย 13,184 จาก 104,000 แถว ฝึกเฉพาะ 90,816 แถวที่มี label
- Task 4: มีภาพ train อ่านได้ 2,811 จาก 31,546 รายการ แต่ภาพ test มีครบ 25,889 ภาพ คะแนน validation อิงเฉพาะภาพที่มีอยู่
- Task 5: ไม่มี train/label การ fit บน test เป็น transductive baseline และ top 10% เป็น threshold ที่สมมติไว้ ไม่ใช่ anomaly prevalence ที่วัดได้ ต้องตรวจว่ากติกาอนุญาต
- Sample submission ใช้ ANS ID โดยไม่มีคอลัมน์เชื่อมกับ player/image ID การรวมคำตอบจึงใช้แถวที่ i ของ test งานนั้นตรงกับแถวที่ i ของ sample ตามสมมติฐาน positional mapping ต้องยืนยันกับผู้จัด
- ผู้ใช้รายงาน Kaggle score รอบแรก 0.20131 แต่ยังไม่ทราบ metric และวิธีรวมคะแนน จึงยังระบุไม่ได้ว่างานไหนฉุดคะแนน

## ไฟล์ประกอบ

`shared/eda_utils.py` มีฟังก์ชันตรวจข้อมูลและกราฟ ส่วน `shared/baseline_utils.py` มี image features, anomaly features และตัวตรวจ submission
`artifacts/` เก็บ inventory ภาพ, anomaly scores, prediction แต่ละงาน และ local validation สำหรับตรวจซ้ำในเครื่อง
Git เก็บเฉพาะ `validation_scores.csv` และ `image_model_scores.csv` จากโฟลเดอร์นี้
ข้อมูลต้นฉบับอยู่ในโฟลเดอร์ `01_binary_classification` ถึง `05_anomaly_detection`
