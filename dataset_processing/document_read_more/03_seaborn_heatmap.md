# 🎨 Seaborn — Tổng Quan & Correlation Heatmap

> **Mục đích:** Thư viện visualization cấp cao xây dựng trên matplotlib. Tích hợp tốt với pandas DataFrame, tự động xử lý nhóm dữ liệu và tạo biểu đồ thống kê đẹp hơn với ít code hơn.

---

## Import & Cấu Hình

```python
import seaborn as sns

sns.set_style('whitegrid')    # nền lưới trắng, chuyên nghiệp
sns.set_palette('Set2')       # bảng màu mặc định dịu mắt
```

### Các Style Có Sẵn

| Style | Mô tả |
|-------|-------|
| `'whitegrid'` | Nền trắng + đường lưới — dùng trong dự án |
| `'darkgrid'` | Nền tối + đường lưới |
| `'white'` | Nền trắng, không lưới |
| `'ticks'` | Chỉ có tick marks |

---

## Tổng Quan Các Hàm Đã Sử Dụng

| Hàm | Mục đích | Ghi chú |
|-----|---------|---------|
| `sns.heatmap()` | Ma trận tương quan trực quan | Dùng nhiều nhất trong dự án |
| `sns.boxplot()` | Boxplot theo nhóm | Đẹp hơn matplotlib boxplot |
| `sns.set_style()` | Đặt theme toàn cục | Gọi 1 lần khi import |
| `sns.set_palette()` | Đặt bảng màu mặc định | Áp dụng cho tất cả chart sau đó |

---

## 🔥 Heatmap — Ma Trận Tương Quan

### Bước 1: Tính Correlation

```python
corr = df.corr()
```

`df.corr()` tính **Pearson Correlation Coefficient** (r) giữa mọi cặp cột số:

$$r = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i-\bar{x})^2 \cdot \sum(y_i-\bar{y})^2}}$$

**Đọc kết quả r:**

| Giá trị r | Ý nghĩa |
|-----------|---------|
| `r = 1.0` | Tương quan thuận hoàn toàn |
| `r > 0.7` | Tương quan thuận mạnh |
| `0.3 < r < 0.7` | Tương quan thuận vừa |
| `-0.3 < r < 0.3` | Tương quan yếu / không đáng kể |
| `r < -0.3` | Tương quan nghịch vừa |
| `r = -1.0` | Tương quan nghịch hoàn toàn |

> ⚠️ **Lưu ý:** Pearson chỉ đo quan hệ **tuyến tính**. Hai biến có quan hệ phi tuyến (cong, bậc 2...) có thể cho r ≈ 0 dù thực ra liên quan chặt.

### Bước 2: Vẽ Heatmap

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Heatmap toàn bộ ma trận ---
corr = df.corr()

# Tạo mask: chỉ hiển thị tam giác dưới (tránh trùng lặp)
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(
    corr,
    mask=mask,          # ẩn tam giác trên
    annot=True,         # hiển thị số trong mỗi ô
    fmt='.2f',          # định dạng 2 chữ số thập phân
    cmap='RdYlGn',      # màu: đỏ (âm) → vàng (0) → xanh (dương)
    center=0,           # đặt màu trung tính tại r=0
    ax=axes[0],
    linewidths=0.5,     # đường kẻ ngăn cách các ô
    cbar_kws={'label': 'Correlation'}
)
axes[0].set_title('Ma Trận Tương Quan (Pearson)', fontweight='bold')

# --- Heatmap tương quan với target ---
target_corr = df.corr()['Outcome'].drop('Outcome').sort_values()
colors = ['#e74c3c' if v > 0 else '#3498db' for v in target_corr.values]

axes[1].barh(target_corr.index, target_corr.values, color=colors, alpha=0.8)
axes[1].axvline(x=0, color='black', linewidth=0.8)
axes[1].set_title('Tương Quan với Outcome (Target)', fontweight='bold')
axes[1].set_xlabel('Pearson Correlation')

plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Giải Thích Các Tham Số `sns.heatmap()`

| Tham số | Giá trị dùng | Ý nghĩa |
|---------|-------------|---------|
| `data` | `corr` | DataFrame vuông (n×n) chứa giá trị r |
| `mask` | `np.triu(...)` | Ẩn tam giác trên — tránh hiển thị trùng |
| `annot` | `True` | Hiện số trong từng ô |
| `fmt` | `'.2f'` | Định dạng số: 2 chữ số thập phân |
| `cmap` | `'RdYlGn'` | Bảng màu phân kỳ: đỏ↔xanh, trắng ở 0 |
| `center` | `0` | Giá trị tại điểm giữa của cmap |
| `linewidths` | `0.5` | Độ dày đường kẻ giữa các ô |
| `cbar_kws` | `{'label': ...}` | Tùy chỉnh thanh màu bên phải |

### Tại Sao Dùng `mask = np.triu(...)`?

```
Ma trận đầy đủ (thừa):       Ma trận tam giác dưới (đủ dùng):
  A    B    C                   A    B    C
A [1.0  0.3  0.7]            A [   
B [0.3  1.0  0.5]            B [0.3       
C [0.7  0.5  1.0]            C [0.7  0.5    ]
                                              
→ Mỗi cặp xuất hiện 2 lần   → Mỗi cặp chỉ xuất hiện 1 lần
```

```python
# np.triu tạo ma trận True ở tam giác trên (bao gồm đường chéo)
mask = np.triu(np.ones_like(corr, dtype=bool))
# → sns.heatmap ẩn những ô có mask=True
```

### Kết Quả Correlation Trong Dự Án

```
Feature                  | r với Outcome | Mức độ
-------------------------|--------------|--------
Glucose                  |    0.47      | ⭐ Mạnh nhất
BMI                      |    0.29      | Vừa
Age                      |    0.24      | Vừa
Pregnancies              |    0.22      | Vừa
DiabetesPedigreeFunction |    0.17      | Yếu-vừa
Insulin                  |    0.13      | Yếu
SkinThickness            |    0.07      | Rất yếu
BloodPressure            |    0.07      | Rất yếu
```

---

## Các Cmap Phổ Biến Cho Heatmap

| Cmap | Dùng khi nào |
|------|-------------|
| `'RdYlGn'` | Giá trị có 2 chiều (âm/dương), ví dụ: correlation |
| `'coolwarm'` | Tương tự, màu xanh-đỏ rõ hơn |
| `'Reds'` | Chỉ hiển thị giá trị dương (ví dụ: % missing) |
| `'Blues'` | Chỉ giá trị dương, màu xanh |
| `'viridis'` | Dữ liệu liên tục, không phân kỳ |

---

## 📝 Tóm Tắt Nhanh

```python
# Pipeline cơ bản: tính → vẽ → đọc
corr = df.corr()                     # 1. tính Pearson r
mask = np.triu(np.ones_like(corr))   # 2. tạo mask tam giác
sns.heatmap(corr, mask=mask,         # 3. vẽ
            annot=True, fmt='.2f',
            cmap='RdYlGn', center=0)
# 4. đọc: ô xanh đậm = tương quan thuận mạnh
#         ô đỏ đậm   = tương quan nghịch mạnh
#         ô trắng/vàng = không tương quan
```
