# 📈 matplotlib.pyplot — Tổng Quan & Các Hàm Đã Dùng

> **Mục đích:** Thư viện vẽ biểu đồ cơ bản của Python. Mọi visualization trong dự án đều bắt đầu từ đây.

---

## Import & Cấu Hình

```python
import matplotlib.pyplot as plt

# Cấu hình mặc định dùng trong dự án
plt.rcParams['figure.figsize'] = (12, 6)   # kích thước mặc định (width, height) tính bằng inch
plt.rcParams['font.size'] = 11             # cỡ chữ mặc định
```

---

## Cấu Trúc Cơ Bản: Figure & Axes

```
Figure (toàn bộ khung vẽ)
└── Axes (1 ô biểu đồ)
    ├── Title
    ├── X-axis (xlabel, xticks)
    ├── Y-axis (ylabel, yticks)
    └── Plot elements (bars, lines, patches...)
```

```python
# Tạo 1 figure với nhiều ô (subplots)
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 8))
# axes là mảng 2D: axes[0][2] = hàng 0, cột 2
axes = axes.flatten()  # chuyển thành mảng 1D để dễ dùng vòng lặp
```

---

## Tổng Quan Các Hàm Đã Sử Dụng

| Hàm | Mục đích |
|-----|---------|
| `plt.subplots(rows, cols, figsize)` | Tạo figure với nhiều ô biểu đồ |
| `ax.hist()` | Vẽ histogram phân phối |
| `df.boxplot(ax=ax)` | Vẽ boxplot phát hiện outlier |
| `ax.bar()` | Biểu đồ cột (bar chart) |
| `ax.barh()` | Biểu đồ cột ngang |
| `ax.pie()` | Biểu đồ tròn |
| `ax.set_title()` | Đặt tiêu đề cho ô |
| `ax.set_xlabel/ylabel()` | Đặt nhãn trục |
| `ax.axvline/axhline()` | Vẽ đường dọc/ngang tham chiếu |
| `ax.text()` | Thêm text vào biểu đồ |
| `ax.legend()` | Hiển thị chú thích |
| `plt.suptitle()` | Tiêu đề chung cho cả figure |
| `plt.tight_layout()` | Tự động căn chỉnh khoảng cách |
| `plt.savefig()` | Lưu biểu đồ ra file |
| `plt.show()` | Hiển thị biểu đồ |

---

## 📊 Histogram — Phân Phối Dữ Liệu

> Histogram chia dữ liệu thành các khoảng (bins) và đếm số điểm rơi vào mỗi khoảng. Dùng để xem hình dạng phân phối.

### Cú pháp

```python
ax.hist(data, bins=25, color='blue', alpha=0.7, 
        density=False, label='label', edgecolor='white')
```

| Tham số | Ý nghĩa | Giá trị dùng trong dự án |
|---------|---------|--------------------------|
| `bins` | Số khoảng chia | `25` — đủ chi tiết, không quá rời rạc |
| `alpha` | Độ trong suốt (0=trong, 1=đặc) | `0.6` — cho 2 histogram chồng nhau thấy được cả hai |
| `density` | True = trục Y là xác suất (%), False = tần suất | `True` khi so sánh 2 nhóm khác cỡ mẫu |
| `color` | Màu | `'#2ecc71'` (xanh lá) cho Outcome=0, `'#e74c3c'` (đỏ) cho Outcome=1 |
| `edgecolor` | Màu viền mỗi bin | `'white'` — tạo khoảng cách rõ giữa các bin |
| `label` | Tên trong legend | `'Outcome=0'`, `'Outcome=1'` |

### Ví dụ Trong Dự Án: Histogram Theo Nhóm

```python
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

for i, col in enumerate(features):
    ax = axes[i]
    # Vẽ 2 histogram chồng nhau theo nhóm Outcome
    df[df['Outcome'] == 0][col].hist(ax=ax, alpha=0.6, color='#2ecc71',
                                      bins=25, label='Outcome=0', density=True)
    df[df['Outcome'] == 1][col].hist(ax=ax, alpha=0.6, color='#e74c3c',
                                      bins=25, label='Outcome=1', density=True)
    ax.set_title(col, fontweight='bold')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)

plt.suptitle('Phân Phối Đặc Trưng theo Nhãn', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Đọc Histogram

```
Phân phối chuẩn (normal):   hình chuông đối xứng
Lệch phải (right-skewed):   đuôi dài bên phải → mean > median (ví dụ: Insulin)
Lệch trái (left-skewed):    đuôi dài bên trái → mean < median
Bimodal:                     2 đỉnh → gợi ý 2 nhóm con trong dữ liệu
```

---

## 📦 Boxplot — Phát Hiện Outlier

> Boxplot tóm tắt phân phối qua 5 số: Min, Q1, Median, Q3, Max và hiển thị outlier riêng biệt.

### Cấu Trúc Boxplot

```
         ┌─────────────────────┐
 ─────────┤         │           ├───────── ○ ○ ○   ← outlier
         └─────────────────────┘
    ↑        ↑       ↑       ↑       ↑
   Q1-1.5×IQR  Q1   Median   Q3   Q3+1.5×IQR
   (whisker)                         (whisker)

Whisker = Q1 - 1.5×IQR  đến  Q3 + 1.5×IQR
Điểm nằm ngoài whisker = OUTLIER (hiển thị là ○)
```

### Cú pháp

```python
# Cách 1: dùng DataFrame.boxplot (dùng trong dự án)
df[feature_cols].boxplot(ax=ax)

# Cách 2: dùng plt trực tiếp
ax.boxplot(df['Glucose'])

# Cách 3: dùng seaborn (đẹp hơn)
sns.boxplot(data=df, x='Outcome', y='Glucose', ax=ax)
```

### Ví Dụ Trong Dự Án: So Sánh Trước/Sau Xử Lý Outlier

```python
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Trước khi xử lý
df_clean[feature_cols].boxplot(ax=axes[0])
axes[0].set_title('Boxplot Trước Khi Xử Lý Outlier', fontweight='bold')

# Sau khi winsorize
df_processed[feature_cols].boxplot(ax=axes[1])
axes[1].set_title('Boxplot Sau Khi Winsorize Outlier', fontweight='bold')

plt.tight_layout()
plt.savefig('outlier_treatment.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Đọc Boxplot

| Quan sát | Ý nghĩa |
|----------|---------|
| Box dài (Q3-Q1 lớn) | Dữ liệu phân tán, IQR lớn |
| Median lệch về Q1 | Phân phối lệch phải |
| Median lệch về Q3 | Phân phối lệch trái |
| Nhiều điểm ○ bên phải | Nhiều outlier cao (ví dụ: Insulin) |
| Whisker ngắn đều | Phân phối gần chuẩn |

---

## Các Hàm Phụ Trợ Quan Trọng

```python
# Thêm đường tham chiếu ngang
ax.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Ngưỡng 20%')

# Thêm text label lên bar
for bar in bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,  # vị trí x (giữa bar)
        bar.get_height() + 0.5,              # vị trí y (trên đầu bar)
        f'{bar.get_height():.1f}%',          # nội dung
        ha='center', va='bottom', fontweight='bold'
    )

# Lưu file chất lượng cao
plt.savefig('output.png', dpi=150, bbox_inches='tight')
# dpi=150: độ phân giải (150 dpi đủ cho báo cáo)
# bbox_inches='tight': cắt bỏ khoảng trắng thừa xung quanh
```

---

## 📝 Tóm Tắt Nhanh

| Muốn xem | Dùng |
|----------|------|
| Hình dạng phân phối | `hist()` |
| Outlier & tứ phân vị | `boxplot()` |
| So sánh số lượng theo nhóm | `bar()` |
| Tỷ lệ phần trăm | `pie()` |
| Nhiều biểu đồ cùng lúc | `subplots()` |
