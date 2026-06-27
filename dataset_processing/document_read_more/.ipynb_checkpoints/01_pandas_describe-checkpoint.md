# 📊 pandas.describe() — Thống Kê Mô Tả

> **Mục đích:** Tóm tắt nhanh phân phối của từng cột số trong DataFrame.

---

## Cú pháp cơ bản

```python
df.describe()
df.describe(include='all')   # bao gồm cả cột object/category
df['Glucose'].describe()     # một cột cụ thể
```

---

## Các thông số đầu ra

```
       Glucose  BloodPressure    BMI   ...
count   768.0       768.0      768.0
mean    120.9        69.1       32.0
std      31.9        19.4        7.9
min       0.0         0.0        0.0
25%      99.0        62.0       27.3
50%     117.0        72.0       32.0
75%     140.2        80.0       36.6
max     199.0       122.0       67.1
```

| Thông số | Ý nghĩa |
|----------|---------|
| `count` | Số giá trị không null |
| `mean` | Giá trị trung bình |
| `std` | Độ lệch chuẩn |
| `min` | Giá trị nhỏ nhất |
| `25%` | Tứ phân vị Q1 — 25% dữ liệu nằm dưới ngưỡng này |
| `50%` | Tứ phân vị Q2 / Median — điểm giữa của dữ liệu |
| `75%` | Tứ phân vị Q3 — 75% dữ liệu nằm dưới ngưỡng này |
| `max` | Giá trị lớn nhất |

---

## 🔍 Hiểu Sâu: 25% / 50% / 75% (Tứ Phân Vị)

```
     Min    Q1      Q2      Q3     Max
      |------[======|========]------|
           25%    50%     75%
                  
           └──────IQR──────┘
```

- **Q1 (25%):** Sắp xếp dữ liệu tăng dần → lấy giá trị tại vị trí 25%. Ví dụ Glucose Q1 = 99 mg/dL nghĩa là 25% bệnh nhân có Glucose < 99.
- **Q2 (50%) = Median:** Điểm chính giữa. Không bị ảnh hưởng bởi outlier (khác với mean). Glucose median = 117 vs mean = 120.9 → phân phối lệch phải nhẹ.
- **Q3 (75%):** 75% bệnh nhân có Glucose < 140.2 mg/dL.
- **IQR = Q3 − Q1:** Khoảng tứ phân vị, đo độ phân tán của 50% dữ liệu giữa. Dùng để phát hiện outlier.

---

## 🔍 Hiểu Sâu: Độ Lệch Chuẩn (std)

**Công thức:**

$$\sigma = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n}(x_i - \bar{x})^2}$$

- Đo mức độ dữ liệu **phân tán** xung quanh mean.
- std nhỏ → dữ liệu tập trung gần mean.
- std lớn → dữ liệu trải rộng, biến động nhiều.

**Quy tắc 68-95-99.7** (với phân phối chuẩn):

```
mean ± 1σ  →  ~68% dữ liệu
mean ± 2σ  →  ~95% dữ liệu
mean ± 3σ  →  ~99.7% dữ liệu
```

**Ví dụ với Glucose:**
```
mean = 120.9,  std = 31.9
→ 68% bệnh nhân có Glucose trong khoảng [89, 152.8]
→ Giá trị > 120.9 + 3×31.9 = 216.6 là outlier cực đoan
```

---

## Phát Hiện Vấn Đề Dữ Liệu qua describe()

```python
# Phát hiện zero không hợp lệ
print(df.describe().loc['min'])
# → Glucose min = 0, BMI min = 0  ← bất hợp lệ!

# So sánh mean vs median (50%) để phát hiện skewness
# mean >> median  → phân phối lệch phải (right-skewed), có outlier cao
# Ví dụ Insulin: mean=79.8 vs median=30.5 → lệch phải rất mạnh
```

---

## Ứng Dụng Trong Dự Án

```python
# Làm tròn cho dễ đọc
df.describe().round(2)

# Chỉ xem các thống kê quan trọng
df.describe().loc[['mean', 'std', '25%', '50%', '75%']]

# Transpose để dễ đọc hơn khi nhiều cột
df.describe().T
```

---

## 📝 Tóm Tắt Nhanh

| Câu hỏi | Dùng thông số |
|---------|--------------|
| Dữ liệu tập trung ở đâu? | `mean`, `50%` |
| Dữ liệu biến động nhiều không? | `std`, IQR = `75% - 25%` |
| Có outlier không? | So sánh `max`/`min` với `mean ± 3*std` |
| Phân phối có lệch không? | So sánh `mean` vs `50%` |
| Có missing/zero không hợp lệ? | Kiểm tra `min = 0` ở cột y tế |
