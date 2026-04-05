# Graph Mining - Fake News Detection

Dự án phát hiện tin giả sử dụng GCN với cấu trúc đồ thị.

---

## 🚀 Cách chạy

### 1. Cài đặt thư viện
```bash
pip install pandas numpy scikit-learn scipy networkx matplotlib seaborn torch torch-geometric
```

### 2. Chuẩn bị dữ liệu
Đặt file CSV vào folder `data/raw/`:

**Option 1: Hai file riêng (Split mode)**
- `true.csv` - tin thật
- `fake.csv` - tin giả

**Option 2: Một file (Single mode)**
- `dataset.csv` với cột `label` (0=thật, 1=giả)

### 3. Chạy pipeline

```bash
cd scripts
python main.py                # Full pipeline

# Hoặc từng bước riêng lẻ
python main.py preprocess     # Bước 1: Preprocessing
python main.py graph          # Bước 2: Graph construction
python main.py visualize      # Bước 3: Visualization
```

---

## ⚙️ Thay đổi số lượng dữ liệu

Hiện tại mặc định xử lý **3000 dòng dữ liệu** để tiết kiệm bộ nhớ.

### Để thay đổi:

Mở file `scripts/main.py` (dòng ~38) tìm:
```python
preprocessor.run_pipeline(sample_size=3000, mode=mode)
```

**Thay đổi số 3000 thành:**
- `sample_size=1000` → 1000 dòng (nhanh, chiếm ít bộ nhớ)
- `sample_size=5000` → 5000 dòng (trung bình)
- `sample_size=-1` → Toàn bộ dữ liệu (lâu, tốn bộ nhớ)

**Ví dụ:**
```python
preprocessor.run_pipeline(sample_size=500, mode=mode)  # Dùng 500 dòng
```

---

## 📊 Output

Sau khi chạy xong, sẽ có:

| Thư mục | File | Mô tả |
|---------|------|-------|
| `data/processed/` | `nodes.csv` | Đặc trưng của các node |
| `data/processed/` | `edges.csv` | Kết nối giữa các node |
| `data/graph/` | `edge_index.npy` | Format cho PyTorch (dùng cho GCN model) |
| `output/visualizations/` | 4 file PNG | 4 biểu đồ phân tích |

---

## 📁 Cấu trúc thư mục

```
GraphMining/
├── scripts/
│   ├── main.py                      # Pipeline logic
│   ├── preprocessing.py             # Xử lý dữ liệu
│   ├── graph_construction.py        # Xây dựng đồ thị
│   └── visualization.py             # Vẽ biểu đồ
├── data/
│   ├── raw/                         # Đặt file CSV ở đây
│   ├── processed/                   # Output từ preprocessing
│   └── graph/                       # Output từ graph construction
└── output/
    └── visualizations/              # Các biểu đồ (PNG)
```

---

## 🔧 Troubleshooting

**"No data files found in data/raw"**
- ✓ Kiểm tra file CSV có trong `data/raw/` không
- ✓ Phải có đúng tên: `true.csv` + `fake.csv` hoặc `dataset.csv` với cột `label`

**"Run preprocessing first"**
- ✓ Chạy `python main.py preprocess` trước

**"Build graph first"**
- ✓ Chạy `python main.py graph` (sau preprocessing)

**Memory/quá lâu**
- ✓ Giảm `sample_size` xuống (ví dụ: 1000 thay vì 3000)

**EUDC font warning**
- ✓ Bỏ qua (Windows Qt warning, không ảnh hưởng output)

---

## 📈 Pipeline Flow

```
Raw CSV 
  ↓
Preprocessing (TF-IDF + KNN graph)
  ↓
nodes.csv + edges.csv
  ↓
Graph Construction (NetworkX → PyTorch format)
  ↓
edge_index.npy
  ↓
Visualization (4 plots)
  ↓
Completed! ✓
```
