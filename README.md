# Graph Mining - Fake News Detection

## Overview
Dự án xây dựng đồ thị lan truyền tin giả (propagation graph) từ dữ liệu tin tức. Gồm 2 bước xử lý:
- **Member 1**: Tiền xử lý dữ liệu → tạo nodes.csv + edges.csv
- **Member 2**: Xây dựng đồ thị → tạo edge_index.npy + visualizations

## Project Structure
```
data/
  ├── raw/              ← Input: fake.csv, true.csv (hoặc WELFake_Dataset.csv)
  ├── processed/        ← Output Member 1: nodes.csv, edges.csv
  └── graph/            ← Output Member 2: edge_index.npy

output/
  └── visualizations/   ← Output Member 2: PNG files (4 biểu đồ)
```

## 1. Setup

### Install Dependencies
```bash
pip install pandas numpy scikit-learn scipy networkx matplotlib seaborn
```

### Download Data

#### Option A: Sử dụng kagglehub (Easy - Recommended)
```bash
pip install kagglehub

# Chạy cell cuối cùng trong fakenews-create-node-edge.ipynb
# Hoặc chạy thủ công từ Python:
import kagglehub
kagglehub.dataset_download("bhavikjikadara/fake-news-detection")
# Files sẽ download và tự động lưu vào data/raw/
```

#### Option B: Download thủ công
1. Vào https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection
2. Click "Download"
3. Giải nén → copy `fake.csv` + `true.csv` vào `data/raw/`

#### Option C: Có sẵn WELFake_Dataset.csv
```
data/raw/WELFake_Dataset.csv đã có trong folder
```

## 2. Xử Lý Dữ Liệu (Member 1)

### Lấy toàn bộ dữ liệu
```bash
python preprocessing.py split
# Hoặc với single file
python preprocessing.py single --input data/raw/WELFake_Dataset.csv
```

### Lấy N sample
```bash
python preprocessing.py split --sample-size 6000
```

**Output**: `data/processed/nodes.csv` + `data/processed/edges.csv`

## 3. Xây Dựng Đồ Thị (Member 2)

```bash
python graph_construction.py 
```

**Output**:
- `data/graph/edge_index.npy` - Đồ thị ở định dạng PyTorch Geometric
- `output/visualizations/degree_distribution.png`
- `output/visualizations/indegree_outdegree.png`
- `output/visualizations/label_distribution.png`
- `output/visualizations/graph_visualization.png`

## 4. Pipeline Hoàn Chỉnh

```bash
# Bước 1: Preprocessing (Member 1)
python preprocessing.py 

# Bước 2: Graph Construction (Member 2)
python graph_construction.py

# Xong! Dữ liệu sẵn sàng cho GNN training
```

## Key Features

| Tính năng | Chi tiết |
|-----------|---------|
| Input modes | Split (fake.csv + true.csv) hoặc Single (1 file) |
| Auto-detection | Tự động chuyển sang single mode nếu split files không có |
| Sampling | Lấy toàn bộ (default) hoặc N mẫu |
| Graph construction | KNN graph (k=5) với cosine similarity |
| Output format | `.npy` (NumPy) thay vì `.pt` (PyTorch) |
| Visualizations | 4 biểu đồ phân tích đồ thị |

## Lưu Ý

- Default: lấy **toàn bộ** dữ liệu (sample_size = -1)
- Để lấy 6000 mẫu: `--sample-size 6000`
- TF-IDF features: 2000 features, max_features
- KNN neighbors: k=5 (cosine similarity)
- Graph: undirected (symmetrize edges)

## Troubleshooting

### Kaggle download fails
```
→ Download thủ công từ Kaggle hoặc dùng WELFake_Dataset.csv có sẵn
```

### Memory issue với toàn bộ dữ liệu
```
→ Giảm sample_size: python preprocessing.py split --sample-size 10000
```

### Files not found
```
→ Kiểm tra data/raw/ có fake.csv + true.csv (hoặc WELFake_Dataset.csv)
```


