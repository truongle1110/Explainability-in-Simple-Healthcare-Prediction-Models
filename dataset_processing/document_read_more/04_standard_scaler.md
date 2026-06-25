# ⚖️ StandardScaler — Chuẩn Hoá Dữ Liệu (Z-score Normalization)

> **Mục đích:** Đưa tất cả features về cùng một thang đo, tránh feature có giá trị lớn (như Insulin: 0–846) lấn át feature có giá trị nhỏ (như DiabetesPedigreeFunction: 0–2.4) trong quá trình train model.

---

## Import

```python
from sklearn.preprocessing import StandardScaler
```

---

## Công Thức

$$z = \frac{x - \mu}{\sigma}$$

| Ký hiệu | Ý nghĩa |
|---------|---------|
| `x` | Giá trị gốc |
| `μ` (mu) | Mean của cột (tính từ tập train) |
| `σ` (sigma) | Std của cột (tính từ tập train) |
| `z` | Giá trị sau khi scale |

**Kết quả:** Mỗi cột sau scale có **mean = 0** và **std = 1**.

**Ví dụ với Glucose:**
```
Giá trị gốc:  44, 99, 117, 140, 199  (mean=120.9, std=31.9)
Sau scale:   -2.4, -0.7, -0.1, 0.6, 2.5
             (mean≈0, std≈1)
```

---

## Cú Pháp & Workflow

```python
scaler = StandardScaler()

# Chỉ scale features liên tục, không scale target và categorical
SCALE_COLS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
              'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

df_scaled = df.copy()
df_scaled[SCALE_COLS] = scaler.fit_transform(df[SCALE_COLS])
```

### fit_transform vs transform

```python
# Trên tập TRAIN: fit (học mean/std) + transform (áp dụng)
X_train_scaled = scaler.fit_transform(X_train)

# Trên tập TEST: chỉ transform (dùng mean/std đã học từ train)
X_test_scaled = scaler.transform(X_test)

# ❌ KHÔNG làm: fit_transform trên test set
# → Sẽ bị data leakage (test set ảnh hưởng ngược lại scaler)
```

> ⚠️ **Quy tắc vàng:** `fit` chỉ trên **train set**. `transform` áp dụng cho cả train và test.

---

## Kiểm Tra Kết Quả

```python
# Sau khi scale: mean ≈ 0, std ≈ 1
print(df_scaled[SCALE_COLS].describe().loc[['mean', 'std']].round(3))

#           Glucose  BloodPressure   BMI   ...
# mean        0.000          0.000  0.000
# std         1.000          1.000  1.000
```

---

## So Sánh Với Các Phương Pháp Scale Khác

| Phương pháp | Công thức | Kết quả | Dùng khi |
|-------------|-----------|---------|---------|
| **StandardScaler** | `(x - mean) / std` | mean=0, std=1 | Dữ liệu gần phân phối chuẩn, có outlier vừa |
| MinMaxScaler | `(x - min) / (max - min)` | Về khoảng [0, 1] | Cần giới hạn rõ ràng, không có outlier |
| RobustScaler | `(x - median) / IQR` | Robust với outlier | Dữ liệu có nhiều outlier (như Insulin) |

**Trong dự án chọn StandardScaler vì:**
- Tương thích tốt với SHAP và LIME
- Hầu hết features gần phân phối chuẩn sau khi impute
- Là lựa chọn mặc định phổ biến trong ML pipeline

---

## Lưu Mean & Std Để Tái Sử Dụng

```python
# Sau khi fit, lưu lại tham số
print("Mean đã học:", scaler.mean_)
print("Std đã học: ", scaler.scale_)

# Inverse transform: đưa giá trị scaled về giá trị gốc
# Hữu ích khi giải thích SHAP trên thang đo thực
original = scaler.inverse_transform(df_scaled[SCALE_COLS])
```

---

## Lưu Ý Cho XAI

```python
# ✅ Dùng df_scaled để TRAIN model
model.fit(X_train_scaled, y_train)

# ✅ Dùng df_processed (CHƯA scale) để VISUALIZE SHAP
# → Giá trị trên biểu đồ SHAP sẽ ở thang đo thực (mg/dL, kg/m²...)
# → Dễ giải thích cho bác sĩ và bệnh nhân hơn
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled)
```

---

## 📝 Tóm Tắt Nhanh

```
Vấn đề:   Insulin (0–846) >> DiabetesPedigreeFunction (0–2.4)
           → Model bị bias về feature có scale lớn

Giải pháp: StandardScaler → đưa tất cả về mean=0, std=1

Quy tắc:  fit_transform(X_train) | transform(X_test)
           KHÔNG fit_transform(X_test) → data leakage!
```
