# 🩺 Diabetes Dataset — Phân Tích & Tiền Xử Lý Dữ Liệu cho XAI

## 📋 Tổng Quan Dataset

**Nguồn:** Pima Indians Diabetes Database (NIDDK — National Institute of Diabetes and Digestive and Kidney Diseases)
**Mục tiêu:** Dự đoán khả năng mắc bệnh tiểu đường dựa trên các chỉ số y tế
**Kích thước:** 768 mẫu × 9 đặc trưng
**Task:** Binary Classification (XAI — Explainable AI)

---

## 🏷️ Mô Tả Các Đặc Trưng (Features)

| STT | Tên Cột | Kiểu | Đơn Vị | Mô Tả |
|-----|---------|------|---------|-------|
| 1 | `Pregnancies` | int | lần | Số lần mang thai |
| 2 | `Glucose` | int | mg/dL | Nồng độ glucose huyết tương sau 2 giờ trong xét nghiệm dung nạp glucose đường uống |
| 3 | `BloodPressure` | int | mmHg | Huyết áp tâm trương |
| 4 | `SkinThickness` | int | mm | Độ dày nếp gấp da cơ tam đầu |
| 5 | `Insulin` | int | mu U/ml | Nồng độ insulin huyết thanh 2 giờ |
| 6 | `BMI` | float | kg/m² | Chỉ số khối cơ thể (Cân nặng / Chiều cao²) |
| 7 | `DiabetesPedigreeFunction` | float | — | Hàm phả hệ tiểu đường (điểm di truyền từ tiền sử gia đình) |
| 8 | `Age` | int | năm | Tuổi |
| 9 | **`Outcome`** | int | 0/1 | **Biến mục tiêu**: 1 = mắc tiểu đường, 0 = không mắc |

---

## 📊 Thống Kê Mô Tả

| Feature | Min | Mean | Median | Max | Std |
|---------|-----|------|--------|-----|-----|
| Pregnancies | 0 | 3.85 | 3 | 17 | 3.37 |
| Glucose | 0 | 120.89 | 117 | 199 | 31.97 |
| BloodPressure | 0 | 69.11 | 72 | 122 | 19.36 |
| SkinThickness | 0 | 20.54 | 23 | 99 | 15.95 |
| Insulin | 0 | 79.80 | 30.5 | 846 | 115.24 |
| BMI | 0 | 31.99 | 32 | 67.1 | 7.88 |
| DiabetesPedigreeFunction | 0.078 | 0.47 | 0.37 | 2.42 | 0.33 |
| Age | 21 | 33.24 | 29 | 81 | 11.76 |

### Phân Bố Nhãn (Class Distribution)
- **Outcome = 0** (không tiểu đường): 500 mẫu — **65.1%**
- **Outcome = 1** (tiểu đường): 268 mẫu — **34.9%**
- ⚠️ **Mất cân bằng nhãn nhẹ** — cần xem xét khi đánh giá model

---

## 🚨 Vấn Đề Chất Lượng Dữ Liệu

### Giá Trị Zero Không Hợp Lệ (Missing Encoded as 0)
Trong các đặc trưng y tế, giá trị `0` là **không thể xảy ra về mặt sinh học** — đây là giá trị thiếu bị mã hóa thành 0:

| Feature | Số lượng Zero | Tỷ lệ | Mức độ ảnh hưởng |
|---------|--------------|-------|-----------------|
| `Glucose` | 5 | 0.7% | 🟡 Thấp |
| `BloodPressure` | 35 | 4.6% | 🟡 Trung bình |
| `BMI` | 11 | 1.4% | 🟡 Thấp |
| `SkinThickness` | 227 | **29.6%** | 🔴 Nghiêm trọng |
| `Insulin` | 374 | **48.7%** | 🔴 Nghiêm trọng |

> ⚠️ `Pregnancies = 0` là hợp lệ (chưa từng mang thai), không cần xử lý.

---

## 🛠️ Chiến Lược Xử Lý Dữ Liệu

### Bước 1 — Thay Thế Zero → NaN
Chuyển các zero không hợp lệ thành `NaN` để xử lý đúng cách.

### Bước 2 — Imputation (Điền Giá Trị Thiếu)
Chiến lược được chọn theo tỷ lệ missing:

| Feature | Phương pháp | Lý do |
|---------|------------|-------|
| `Glucose` | **Median** theo `Outcome` | Glucose phân phối lệch, median robust hơn mean |
| `BloodPressure` | **Median** theo `Outcome` | Tương tự |
| `BMI` | **Median** theo `Outcome` | Tương tự |
| `SkinThickness` | **KNN Imputer** (k=5) | Missing nhiều (29.6%), cần học từ đặc trưng khác |
| `Insulin` | **KNN Imputer** (k=5) | Missing rất nhiều (48.7%), tương quan cao với Glucose |

### Bước 3 — Xử Lý Outlier
Dùng **IQR method** để phát hiện và winsorize (clip) outlier thay vì xóa (giữ nguyên số lượng mẫu).

### Bước 4 — Feature Scaling
Dùng **StandardScaler** (Z-score normalization) — phù hợp cho các model ML và XAI methods (SHAP, LIME).

### Bước 5 — Feature Engineering (Gợi ý thêm)
Thêm các đặc trưng mới có ý nghĩa y học:
- `BMI_Category`: gầy / bình thường / thừa cân / béo phì
- `Glucose_Category`: bình thường / tiền tiểu đường / tiểu đường
- `Age_Group`: nhóm tuổi
- `Insulin_Glucose_Ratio`: tỷ lệ insulin/glucose

---

## 📁 Cấu Trúc Thư Mục

```
Explainability-in-Simple-Healthcare-Prediction-Models/
├── dataset/
│   ├── diabetes.csv            # Dữ liệu gốc Pima Indians Diabetes
│   └── diabetes_cleaned.csv      # Dữ liệu sau khi xử lý

├── dataset_processing/
│   ├── diabetes_analysis.ipynb   # Jupyter Notebook phân tích & tiền xử lý
│   ├── README.md                 # Tài liệu chi tiết về dataset & pipeline
│   └── requirements.txt          # Danh sách thư viện cần cài đặt
└── README.md                     # Tổng quan dự án
```
---

## 🚀 Hướng Dẫn Chạy Nhanh

**1. Clone repository**
```bash
git clone https://github.com/<your-username>/Explainability-in-Simple-Healthcare-Prediction-Models.git
cd Explainability-in-Simple-Healthcare-Prediction-Models
```

**2. Cài đặt thư viện**
```bash
pip install -r notebook/requirements.txt
```

**3. Chạy notebook**
```bash
jupyter notebook notebook/diabetes_analysis.ipynb
```

---
---

## 🔬 Pipeline Phân Tích

```
1. Import & Load Data
2. Exploratory Data Analysis (EDA)
   ├── Thống kê mô tả
   ├── Phân phối từng feature
   ├── Correlation heatmap
   └── Class distribution
3. Data Cleaning
   ├── Xử lý zero values → NaN
   ├── Median imputation (low missing)
   └── KNN imputation (high missing)
4. Outlier Detection & Treatment
5. Feature Engineering
6. Feature Scaling
7. Export cleaned data
8. [Đề xuất] Kiểm tra lại sau xử lý
```

---

## 💡 Lưu Ý Cho XAI

Khi áp dụng XAI methods trên dataset này:
- **SHAP Values**: sử dụng trên scaled data, interpret trên original scale
- **LIME**: cần specify feature names đầy đủ
- **Feature Importance**: `Glucose`, `BMI`, `Age` thường là top features
- **Class Imbalance**: cân nhắc dùng `class_weight='balanced'` hoặc SMOTE trước khi train

---

## 📚 Tài Liệu Tham Khảo

- [Pima Indians Diabetes Dataset — Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- [NIDDK Diabetes Information](https://www.niddk.nih.gov/health-information/diabetes)
