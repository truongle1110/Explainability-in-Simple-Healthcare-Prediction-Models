# 🔧 KNNImputer — Điền Giá Trị Thiếu Bằng K Hàng Xóm Gần Nhất

> **Mục đích:** Điền giá trị NaN bằng cách học từ các hàng dữ liệu tương tự nhất, thay vì dùng một giá trị cố định (mean/median). Hiệu quả hơn khi tỷ lệ missing cao.

---

## Import

```python
from sklearn.impute import KNNImputer
```

---

## Nguyên Lý Hoạt Động

**K-Nearest Neighbors Imputation:**

1. Với mỗi hàng có giá trị NaN, tìm **k hàng gần nhất** (neighbors) dựa trên các cột không bị missing.
2. Lấy **trung bình có trọng số** của k hàng đó để điền vào chỗ trống.
3. Hàng càng gần (khoảng cách Euclidean nhỏ) → trọng số càng cao.

**Ví dụ trực quan:**

```
Bệnh nhân A: Glucose=150, BMI=35, SkinThickness=NaN
                                         ↑ cần điền

Tìm 5 bệnh nhân gần nhất (dựa trên Glucose, BMI...):
  Bệnh nhân 1: Glucose=145, BMI=34, SkinThickness=32  (gần nhất)
  Bệnh nhân 2: Glucose=152, BMI=36, SkinThickness=28
  Bệnh nhân 3: Glucose=148, BMI=33, SkinThickness=35
  Bệnh nhân 4: Glucose=155, BMI=37, SkinThickness=30
  Bệnh nhân 5: Glucose=142, BMI=32, SkinThickness=33

→ SkinThickness của A ≈ weighted_avg(32, 28, 35, 30, 33) ≈ 31.5
```

---

## Cú Pháp & Tham Số

```python
knn_imputer = KNNImputer(
    n_neighbors=5,         # số hàng xóm dùng để tính
    weights='distance',    # cách tính trọng số
    metric='nan_euclidean' # cách tính khoảng cách khi có NaN
)
```

| Tham số | Giá trị dùng | Các lựa chọn | Ý nghĩa |
|---------|-------------|-------------|---------|
| `n_neighbors` | `5` | Số nguyên dương | Số hàng xóm. k nhỏ → nhạy cảm với noise; k lớn → smooth hơn nhưng chậm |
| `weights` | `'distance'` | `'uniform'`, `'distance'` | `'distance'`: hàng gần hơn có trọng số cao hơn (tốt hơn) |
| `metric` | `'nan_euclidean'` | mặc định | Khoảng cách Euclidean bỏ qua NaN khi tính |

---

## Ứng Dụng Trong Dự Án

```python
# KNN Imputer dùng cho SkinThickness (29.6% missing) và Insulin (48.7% missing)
# Vì tỷ lệ missing quá cao → median imputation không đủ tốt

feature_cols = [c for c in df_clean.columns if c != 'Outcome']

knn_imputer = KNNImputer(n_neighbors=5, weights='distance')

# fit_transform: học từ dữ liệu hiện có và điền NaN
df_imputed = knn_imputer.fit_transform(df_clean[feature_cols])

# Chuyển lại thành DataFrame
df_clean[feature_cols] = pd.DataFrame(df_imputed, columns=feature_cols)
```

> 💡 KNNImputer xử lý **tất cả cột cùng lúc** — không cần loop từng cột. Điều này giúp nó tận dụng mối quan hệ giữa các features để tính khoảng cách tốt hơn.

---

## So Sánh Các Phương Pháp Imputation

| Phương pháp | Khi nào dùng | Ưu điểm | Nhược điểm |
|-------------|-------------|---------|-----------|
| **Mean Imputation** | Missing < 5%, không có outlier | Đơn giản, nhanh | Bóp méo phân phối, phá vỡ correlation |
| **Median Imputation** | Missing < 10%, có outlier | Robust hơn mean | Vẫn dùng 1 giá trị cố định cho tất cả |
| **KNN Imputation** | Missing 10–50%, features có tương quan | Học từ dữ liệu, giữ correlation | Chậm hơn, nhạy cảm với scale |
| **Iterative Imputer** | Missing > 50%, cần độ chính xác cao | Chính xác nhất | Rất chậm |

**Chiến lược trong dự án:**

```
Glucose, BloodPressure, BMI  (< 5% missing)  → Median theo nhóm Outcome
SkinThickness                (29.6% missing) → KNN Imputer (k=5)
Insulin                      (48.7% missing) → KNN Imputer (k=5)
```

---

## Tại Sao Không Dùng Mean/Median Cho Insulin?

```
Insulin trước imputation:
  - 374/768 = 48.7% là NaN (sau khi thay 0 → NaN)
  - Phân phối lệch phải rất mạnh (mean=130, median=94)
  - Tương quan cao với Glucose (r ≈ 0.33)

Nếu dùng median imputation:
  → 374 điểm đều nhận giá trị = 94
  → Tạo spike nhân tạo tại 94, phá vỡ phân phối tự nhiên
  → Mất đi thông tin từ mối quan hệ Glucose-Insulin

Nếu dùng KNN Imputation:
  → Mỗi điểm nhận giá trị khác nhau dựa trên Glucose, BMI, Age...
  → Giữ được phân phối và correlation tự nhiên hơn
```

---

## Lưu Ý Quan Trọng

```python
# ⚠️ Scale trước khi dùng KNN (nếu features có scale khác nhau)
# KNN dùng khoảng cách Euclidean → feature scale lớn sẽ dominates

# Thứ tự đúng trong pipeline:
# 1. Replace 0 → NaN
# 2. KNN Impute  (metric='nan_euclidean' tự xử lý NaN)
# 3. Outlier treatment
# 4. StandardScaler

# ⚠️ KNNImputer chỉ nhận numpy array hoặc DataFrame không có NaN ở output
# Nếu 1 cột NaN hoàn toàn → lỗi. Kiểm tra trước:
print(df_clean.isnull().mean())  # không cột nào được 100% NaN
```

---

## 📝 Tóm Tắt Nhanh

```
Vấn đề:   Insulin 48.7% missing → không thể dùng 1 giá trị cố định

Giải pháp: KNNImputer(n_neighbors=5, weights='distance')
           → Tìm 5 bệnh nhân tương tự → lấy weighted average Insulin của họ

Kết quả:  Mỗi NaN được điền bằng giá trị khác nhau, phù hợp ngữ cảnh
           → Giữ được phân phối và tương quan tự nhiên giữa features
```
