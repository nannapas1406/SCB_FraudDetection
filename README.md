# Getting Started
หลังจากโคลน git นี้แล้วให้เข้าไปที่ folder ของ git นี้ จากนั้นให้สร้างและ activate ตัว python environment โดยการรันคำสั้งต่อไปนี้ใน terminal
``` 
conda env create -f environment.yml
conda activate fraud-detect
```
หากต้องการทำการ train แบบจำลองใหม่ใน `/notebooks/fraud_detection.ipynb` ให้โหลด dataset จาก https://scbpocseasta001stdsbx.z23.web.core.windows.net/ เข้ามาไว้ที่ folder `/data`

# Repository Structures
```
SCB_FraudDetection
├── api
│ ├── pycache/
│ ├── fraud_db.sqlite
│ └── main.py # สคริปต์หลักของ FastAPI สำหรับเรียกโมเดลตรวจจับ fraud
│
├── data
│ └── scbpocseasta001stdsbx.z23.web.core.windows.net # dataset
│
├── diagrams
│ └── Fraud Detection Diagram.drawio # architecture for the fraud detection system
│
├── model
│ ├── col_transformer.pkl # ตัวแปลงข้อมูล (ColumnTransformer)
│ └── xgb_model. # โมเดล XGBoost ที่เทรนแล้ว
│
├── notebooks
│ ├── eda.ipynb # โน้ตบุ๊กสำหรับการวิเคราะห์ข้อมูลเบื้องต้น (EDA) พร้อมทั้งการวิเคราะห์ข้อมูลอย่างละเอียด
│ ├── environment.yml # รายชื่อ dependencies สำหรับสร้าง environment
│ └── fraud_detection.ipynb # โน้ตบุ๊กหลักสำหรับการเทรนและประเมินโมเดล พร้อมทั้งสรุปผลการเทรนและการเลือกโมเดล
│
├── .gitattributes
├── README.md # คำอธิบายโปรเจกต์และวิธีใช้งาน
└── take_home_exam.pdf # โจทย์
```
