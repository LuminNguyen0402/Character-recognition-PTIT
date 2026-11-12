# Demo Web - EMNIST Handwriting Recognition

Trang web demo cho bài toán nhận dạng ký tự viết tay trên tập **EMNIST ByMerge** (47 lớp: 0-9, A-Z, và 11 chữ thường a, b, d, e, f, g, h, n, q, r, t).

So sánh 2 model đã train sẵn:
- **MLP** (Multi-layer Perceptron) — accuracy ~88.6%
- **CNN** (Convolutional Neural Network) — accuracy ~90%

## Cài đặt

Yêu cầu đã có sẵn trong môi trường `emnist` (Anaconda):

```bash
conda activate emnist
pip install gradio
```

## Chạy demo

```bash
cd D:\10.Machine-Learning-prj\bt_cuoiky\demo
python app.py
```

Sau đó mở trình duyệt tại: **http://127.0.0.1:7860**

Lần đầu chạy mất khoảng **30-60 giây** để load model + dữ liệu test (116k mẫu).

## Các chức năng

### Tab 1: Preview mẫu
- Chọn số mẫu (1-100), nhóm lớp (tất cả / chỉ chữ số / chỉ chữ hoa / chỉ chữ thường), và seed.
- **Seed = 0** (mặc định): random thật, mỗi lần bấm sẽ ra mẫu khác nhau.
- **Seed khác 0** (ví dụ 42): tái lập kết quả — bấm nhiều lần vẫn ra cùng mẫu (hữu ích khi muốn demo lại).
- Bấm **"Lấy mẫu ngẫu nhiên"** để hiển thị grid các mẫu kèm label thật.

### Tab 2: Test model trên 1 mẫu
- Nhập **index** mẫu (0 → 116322) hoặc bấm **"Lấy ngẫu nhiên 1 mẫu"** (cũng tôn trọng seed).
- Bấm **"Chạy MLP + CNN"** để xem:
  - Ảnh input 28×28
  - Label thật
  - Dự đoán của MLP và CNN (kèm độ tự tin %)
  - Top-5 dự đoán có xác suất cao nhất

### Tab 3: So sánh MLP vs CNN
- Chọn số mẫu (1-50), nhóm lớp, và seed (cùng logic 0=random, khác 0=tái lập).
- Bấm **"So sánh"** để xem bảng so sánh:
  - **Xanh** = dự đoán đúng
  - **Đỏ** = dự đoán sai
- Thống kê: MLP đúng bao nhiêu, CNN đúng bao nhiêu, chỉ MLP đúng, chỉ CNN đúng, cả hai sai.

## Cấu trúc file

```
demo/
├── app.py              # File chính của demo
├── README.md           # File này
├── mlp_probs.npy       # (tuỳ chọn) Cache predictions của MLP
└── cnn_probs.npy       # (tuỳ chọn) Cache predictions của CNN
```

## Ghi chú kỹ thuật

- Ảnh trong EMNIST ByMerge được lưu dạng transpose/flip, notebook gốc **không** áp dụng `np.transpose`/`np.flip` để giữ đúng khớp với model đã train.
- Demo load dữ liệu test từ `../emnist-bymerge-test.csv` và model từ `../mlp_model.keras`, `../cnn_model.keras`.
- Predictions được tính **on-demand** (theo yêu cầu), không pre-compute toàn bộ — giúp khởi động nhanh.
- Nếu muốn pre-compute và cache predictions để test nhanh hơn, có thể bật lại trong code (đã comment hướng dẫn).

## Các cặp ký tự dễ nhầm lẫn

Do ảnh bị xoay/flip trong dataset gốc, model thường nhầm lẫn giữa:
- **O** ↔ **0**
- **Z** ↔ **2**
- **S** ↔ **5**
- **I** ↔ **l** ↔ **1**

Có thể quan sát trực tiếp các lỗi này qua Tab 3.
