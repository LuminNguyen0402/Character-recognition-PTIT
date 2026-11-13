"""
Demo web cho bài toán nhận dạng ký tự viết tay EMNIST ByMerge (47 lớp).
Ba chức năng:
  1. Preview mẫu: hiển thị grid các mẫu ngẫu nhiên từ tập test.
  2. Test model: chọn 1 mẫu, chạy qua MLP và CNN, hiển thị dự đoán + độ tự tin.
  3. So sánh MLP vs CNN trên nhiều mẫu ngẫu nhiên.

Chạy:
    python app.py
Sau đó mở trình duyệt tại http://127.0.0.1:7860

Lưu ý:
    - App khởi động nhanh (~30 giây để load model + data).
    - Predictions được tính theo yêu cầu (on-demand), không pre-compute.
    - Có thể lọc theo nhóm lớp: chữ số, chữ hoa, chữ thường.
"""

import os
import sys
import random
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf

# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
TEST_CSV = PROJECT_DIR / "emnist-bymerge-test.csv"
MLP_MODEL = PROJECT_DIR / "mlp_model.keras"
CNN_MODEL = PROJECT_DIR / "cnn_model.keras"

# Label dict giống trong notebook (47 lớp, ByMerge)
LABELS = {
    0: "0",  1: "1",  2: "2",  3: "3",  4: "4",  5: "5",  6: "6",  7: "7",
    8: "8",  9: "9",
    10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H",
    18: "I", 19: "J", 20: "K", 21: "L", 22: "M", 23: "N", 24: "O", 25: "P",
    26: "Q", 27: "R", 28: "S", 29: "T", 30: "U", 31: "V", 32: "W", 33: "X",
    34: "Y", 35: "Z",
    36: "a", 37: "b", 38: "d", 39: "e", 40: "f", 41: "g", 42: "h", 43: "n",
    44: "q", 45: "r", 46: "t",
}

# Nhóm lớp để filter
DIGIT_LABELS = list(range(0, 10))
UPPER_LABELS = list(range(10, 36))
LOWER_LABELS = list(range(36, 47))


def log(msg):
    """Print với flush để theo dõi tiến trình."""
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Load dữ liệu + model (1 lần khi khởi động)
# ---------------------------------------------------------------------------
def load_test_data():
    """Load test CSV, trả về (images_uint8 [N,28,28], labels [N])."""
    log(f"[INFO] Đang load test data từ {TEST_CSV} ...")
    df = pd.read_csv(TEST_CSV, header=None)
    labels = df.iloc[:, 0].to_numpy().astype(np.int32)
    pixels = df.iloc[:, 1:].to_numpy().astype(np.uint8).reshape(-1, 28, 28)
    log(f"[INFO] Đã load {len(labels)} mẫu test.")
    return pixels, labels


def load_models():
    """Load MLP và CNN, trả về (mlp, cnn)."""
    log(f"[INFO] Đang load MLP model từ {MLP_MODEL} ...")
    mlp = tf.keras.models.load_model(
        str(MLP_MODEL),
        custom_objects={"relu": tf.nn.relu, "softmax_v2": tf.nn.softmax},
    )
    log(f"[INFO] Đang load CNN model từ {CNN_MODEL} ...")
    cnn = tf.keras.models.load_model(
        str(CNN_MODEL),
        custom_objects={"relu": tf.nn.relu, "softmax_v2": tf.nn.softmax},
    )
    log("[INFO] Load model xong.")
    return mlp, cnn


log("=" * 60)
log("KHỞI ĐỘNG DEMO - EMNIST HANDWRITING RECOGNITION")
log("=" * 60)

TEST_IMAGES, TEST_LABELS = load_test_data()
MLP_MODEL_OBJ, CNN_MODEL_OBJ = load_models()

log("=" * 60)
log("SẴN SÀNG! Mở trình duyệt tại http://127.0.0.1:7860")
log("=" * 60)


# ---------------------------------------------------------------------------
# Hàm tiện ích
# ---------------------------------------------------------------------------
def get_indices_by_filter(filter_type: str):
    """Lọc index theo nhóm lớp."""
    if filter_type == "Tất cả (47 lớp)":
        return np.arange(len(TEST_LABELS))
    if filter_type == "Chỉ chữ số (0-9)":
        return np.where(np.isin(TEST_LABELS, DIGIT_LABELS))[0]
    if filter_type == "Chỉ chữ hoa (A-Z)":
        return np.where(np.isin(TEST_LABELS, UPPER_LABELS))[0]
    if filter_type == "Chỉ chữ thường (a,b,d,...)":
        return np.where(np.isin(TEST_LABELS, LOWER_LABELS))[0]
    return np.arange(len(TEST_LABELS))


def make_rng(seed):
    """Tạo RNG: seed=0 (hoặc None) thì dùng random thật, ngược lại dùng seed để tái lập."""
    if seed is None or int(seed) == 0:
        return np.random.default_rng()  # random thật, mỗi lần khác nhau
    return np.random.default_rng(int(seed))


def make_image_grid(indices, cols=10):
    """Tạo ảnh grid [rows*28, cols*28] từ danh sách index."""
    n = len(indices)
    rows = (n + cols - 1) // cols
    grid = np.full((rows * 28, cols * 28), 255, dtype=np.uint8)
    for i, idx in enumerate(indices):
        r, c = divmod(i, cols)
        img = TEST_IMAGES[idx]
        # EMNIST trong CSV bị xoay/flip theo nhiều nguồn, nhưng notebook gốc
        # KHÔNG áp dụng np.transpose/flip. Giữ nguyên để khớp với model đã train.
        grid[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = img
    return grid


def predict_mlp(images_uint8):
    """Predict MLP cho 1 hoặc nhiều ảnh uint8 [N,28,28]. Trả về probs [N,47]."""
    x = images_uint8.astype(np.float32) / 255.0
    x = x.reshape(-1, 28 * 28)
    return MLP_MODEL_OBJ.predict(x, verbose=0)


def predict_cnn(images_uint8):
    """Predict CNN cho 1 hoặc nhiều ảnh uint8 [N,28,28]. Trả về probs [N,47]."""
    x = images_uint8.astype(np.float32) / 255.0
    x = x.reshape(-1, 28, 28, 1)
    return CNN_MODEL_OBJ.predict(x, verbose=0)


# ---------------------------------------------------------------------------
# TAB 1: Preview mẫu
# ---------------------------------------------------------------------------
def preview_samples(num_samples, filter_type, seed):
    """Trả về (grid_image, info_text)."""
    rng = make_rng(seed)
    pool = get_indices_by_filter(filter_type)
    if len(pool) == 0:
        return np.full((28, 28), 255, dtype=np.uint8), "Không có mẫu nào."
    n = int(num_samples)
    n = max(1, min(n, 100, len(pool)))
    chosen = rng.choice(pool, size=n, replace=False)
    chosen = np.sort(chosen)

    grid = make_image_grid(chosen, cols=10)

    # Thông tin: liệt kê label của từng mẫu
    lines = [f"Đã hiển thị {n} mẫu (filter: {filter_type})."]
    lines.append("Label thật của từng mẫu (theo thứ tự trái -> phải, trên -> dưới):")
    for i, idx in enumerate(chosen):
        lines.append(f"  [{i + 1:2d}] idx={idx:6d}  label='{LABELS[int(TEST_LABELS[idx])]}'")
    info = "\n".join(lines)
    return grid, info


# ---------------------------------------------------------------------------
# TAB 2: Test model
# ---------------------------------------------------------------------------
def test_single_sample(index):
    """Test 1 mẫu với cả MLP và CNN."""
    idx = int(index)
    if idx < 0 or idx >= len(TEST_LABELS):
        return (
            np.full((28, 28), 255, dtype=np.uint8),
            "Index không hợp lệ.",
            [],
            [],
        )

    img = TEST_IMAGES[idx]
    true_label = LABELS[int(TEST_LABELS[idx])]

    # Predict
    mlp_probs = predict_mlp(img[None, ...])[0]
    cnn_probs = predict_cnn(img[None, ...])[0]

    mlp_pred = int(mlp_probs.argmax())
    cnn_pred = int(cnn_probs.argmax())
    mlp_conf = float(mlp_probs[mlp_pred])
    cnn_conf = float(cnn_probs[cnn_pred])
    mlp_correct = "ĐÚNG" if mlp_pred == TEST_LABELS[idx] else "SAI"
    cnn_correct = "ĐÚNG" if cnn_pred == TEST_LABELS[idx] else "SAI"

    info_lines = [
        f"Index: {idx}",
        f"Label thật (ground truth): '{true_label}'  (class {int(TEST_LABELS[idx])})",
        "",
        f"[MLP]  Dự đoán: '{LABELS[mlp_pred]}'  |  Độ tự tin: {mlp_conf * 100:.2f}%  |  {mlp_correct}",
        f"[CNN]  Dự đoán: '{LABELS[cnn_pred]}'  |  Độ tự tin: {cnn_conf * 100:.2f}%  |  {cnn_correct}",
    ]
    info_text = "\n".join(info_lines)

    # Bảng top-5
    def topk_table(probs, k=5):
        top_idx = probs.argsort()[-k:][::-1]
        return [(LABELS[int(j)], f"{float(probs[j]) * 100:.2f}") for j in top_idx]

    mlp_top5 = topk_table(mlp_probs)
    cnn_top5 = topk_table(cnn_probs)

    return img, info_text, mlp_top5, cnn_top5


def random_index(filter_type, seed):
    """Chọn ngẫu nhiên 1 index theo filter, trả về index + label."""
    rng = make_rng(seed)
    pool = get_indices_by_filter(filter_type)
    if len(pool) == 0:
        return 0, "Không có mẫu nào."
    idx = int(rng.choice(pool))
    return idx, f"Ngẫu nhiên: idx={idx}, label thật='{LABELS[int(TEST_LABELS[idx])]}'"


# ---------------------------------------------------------------------------
# TAB 3: So sánh MLP vs CNN
# ---------------------------------------------------------------------------
def compare_models(num_samples, filter_type, seed):
    """Chạy MLP và CNN trên nhiều mẫu, trả về bảng so sánh + thống kê."""
    rng = make_rng(seed)
    pool = get_indices_by_filter(filter_type)
    if len(pool) == 0:
        return np.full((28, 28), 255, dtype=np.uint8), "Không có mẫu nào."

    n = int(num_samples)
    n = max(1, min(n, 50, len(pool)))
    chosen = rng.choice(pool, size=n, replace=False)

    # Lấy ảnh và predict theo batch
    batch = TEST_IMAGES[chosen]
    mlp_probs = predict_mlp(batch)
    cnn_probs = predict_cnn(batch)
    mlp_preds = mlp_probs.argmax(axis=1)
    cnn_preds = cnn_probs.argmax(axis=1)

    # Thống kê
    mlp_correct = int((mlp_preds == TEST_LABELS[chosen]).sum())
    cnn_correct = int((cnn_preds == TEST_LABELS[chosen]).sum())
    both_correct = int(((mlp_preds == TEST_LABELS[chosen]) & (cnn_preds == TEST_LABELS[chosen])).sum())
    only_mlp = int(((mlp_preds == TEST_LABELS[chosen]) & (cnn_preds != TEST_LABELS[chosen])).sum())
    only_cnn = int(((mlp_preds != TEST_LABELS[chosen]) & (cnn_preds == TEST_LABELS[chosen])).sum())
    both_wrong = n - both_correct - only_mlp - only_cnn

    # Tạo bảng so sánh bằng PIL
    from PIL import Image, ImageDraw, ImageFont
    cell_w = 28
    label_w = 70
    row_h = 36
    header_h = 25
    total_w = cell_w + label_w * 4 + 20
    total_h = header_h + row_h * n + 10

    canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)
    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
        font_b = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
        font_b = font

    # Header
    x_img = 10
    x_true = x_img + cell_w + 10
    x_mlp = x_true + label_w
    x_cnn = x_mlp + label_w
    draw.text((x_img, 5), "Ảnh", fill="black", font=font_b)
    draw.text((x_true, 5), "True", fill="black", font=font_b)
    draw.text((x_mlp, 5), "MLP", fill="black", font=font_b)
    draw.text((x_cnn, 5), "CNN", fill="black", font=font_b)

    for i, idx in enumerate(chosen):
        idx = int(idx)
        true = int(TEST_LABELS[idx])
        mp = int(mlp_preds[i])
        cp = int(cnn_preds[i])
        y = header_h + i * row_h

        pil_img.paste(Image.fromarray(TEST_IMAGES[idx]), (x_img, y))
        draw.text((x_true, y + 8), f"'{LABELS[true]}'", fill="black", font=font)
        mlp_color = "green" if mp == true else "red"
        cnn_color = "green" if cp == true else "red"
        draw.text((x_mlp, y + 8), f"'{LABELS[mp]}'", fill=mlp_color, font=font)
        draw.text((x_cnn, y + 8), f"'{LABELS[cp]}'", fill=cnn_color, font=font)

    grid = np.array(pil_img)

    stats = (
        f"=== KẾT QUẢ TRÊN {n} MẪU (filter: {filter_type}) ===\n"
        f"MLP đúng       : {mlp_correct}/{n}  ({mlp_correct / n * 100:.1f}%)\n"
        f"CNN đúng       : {cnn_correct}/{n}  ({cnn_correct / n * 100:.1f}%)\n"
        f"Cả hai đúng    : {both_correct}/{n}\n"
        f"Chỉ MLP đúng   : {only_mlp}/{n}\n"
        f"Chỉ CNN đúng   : {only_cnn}/{n}\n"
        f"Cả hai sai     : {both_wrong}/{n}\n"
        f"\nChi tiết (xanh = đúng, đỏ = sai):\n"
    )
    for i, idx in enumerate(chosen):
        idx = int(idx)
        true = int(TEST_LABELS[idx])
        mp = int(mlp_preds[i])
        cp = int(cnn_preds[i])
        stats += (
            f"  [{i + 1:2d}] idx={idx:6d}  true='{LABELS[true]}'  "
            f"MLP='{LABELS[mp]}'  CNN='{LABELS[cp]}'\n"
        )

    return grid, stats


# ---------------------------------------------------------------------------
# Giao diện Gradio
# ---------------------------------------------------------------------------
with gr.Blocks(title="EMNIST Handwriting Recognition Demo") as demo:
    gr.Markdown(
        """
        # Demo nhận dạng ký tự viết tay - EMNIST ByMerge (47 lớp)
        Dataset gồm **chữ số (0-9)**, **chữ hoa (A-Z)** và **11 chữ thường** (a, b, d, e, f, g, h, n, q, r, t).
        Model đã train: **MLP** (accuracy ~88.6%) và **CNN** (accuracy ~90%).
        """
    )

    # ----- TAB 1: Preview mẫu -----
    with gr.Tab("1. Preview mẫu"):
        gr.Markdown(
            "Hiển thị grid các mẫu ngẫu nhiên từ tập test. Có thể lọc theo nhóm lớp."
        )
        with gr.Row():
            with gr.Column(scale=1):
                num_p = gr.Slider(1, 100, value=20, step=1, label="Số mẫu (tối đa 100)")
                filter_p = gr.Radio(
                    ["Tất cả (47 lớp)", "Chỉ chữ số (0-9)", "Chỉ chữ hoa (A-Z)",
                     "Chỉ chữ thường (a,b,d,...)"],
                    value="Tất cả (47 lớp)",
                    label="Lọc theo lớp",
                )
                seed_p = gr.Number(
                    value=0,
                    label="Seed (0 = random thật; nhập số khác để tái lập kết quả)",
                )
                btn_p = gr.Button("Lấy mẫu ngẫu nhiên", variant="primary")
            with gr.Column(scale=3):
                out_img_p = gr.Image(label="Grid các mẫu", type="numpy", height=400)
                out_info_p = gr.Textbox(label="Thông tin label", lines=10)

        btn_p.click(
            preview_samples,
            inputs=[num_p, filter_p, seed_p],
            outputs=[out_img_p, out_info_p],
        )

    # ----- TAB 2: Test model -----
    with gr.Tab("2. Test model trên 1 mẫu"):
        gr.Markdown(
            "Chọn 1 mẫu cụ thể (theo index) hoặc lấy ngẫu nhiên, sau đó xem dự đoán của MLP và CNN."
        )
        with gr.Row():
            with gr.Column(scale=1):
                idx_t = gr.Number(value=0, label="Index mẫu (0 - 116322)", precision=0)
                with gr.Row():
                    filter_t = gr.Radio(
                        ["Tất cả (47 lớp)", "Chỉ chữ số (0-9)", "Chỉ chữ hoa (A-Z)",
                         "Chỉ chữ thường (a,b,d,...)"],
                        value="Tất cả (47 lớp)",
                        label="Lọc khi lấy ngẫu nhiên",
                        scale=2,
                    )
                    seed_t = gr.Number(value=0, label="Seed (0=random)", scale=1)
                btn_rand = gr.Button("Lấy ngẫu nhiên 1 mẫu")
                rand_info = gr.Textbox(label="Index ngẫu nhiên", lines=1)
                btn_test = gr.Button("Chạy MLP + CNN", variant="primary")
            with gr.Column(scale=2):
                out_img_t = gr.Image(label="Ảnh input (28x28)", type="numpy", height=300)
                out_info_t = gr.Textbox(label="Kết quả dự đoán", lines=8)
                gr.Markdown("**Top-5 dự đoán (xác suất cao nhất)**")
                with gr.Row():
                    mlp_table = gr.Dataframe(
                        headers=["Ký tự", "Xác suất (%)"], label="MLP", wrap=True
                    )
                    cnn_table = gr.Dataframe(
                        headers=["Ký tự", "Xác suất (%)"], label="CNN", wrap=True
                    )

        btn_rand.click(
            random_index, inputs=[filter_t, seed_t], outputs=[idx_t, rand_info]
        )
        btn_test.click(
            test_single_sample,
            inputs=[idx_t],
            outputs=[out_img_t, out_info_t, mlp_table, cnn_table],
        )

    # ----- TAB 3: So sánh MLP vs CNN -----
    with gr.Tab("3. So sánh MLP vs CNN"):
        gr.Markdown(
            "Chọn nhiều mẫu ngẫu nhiên, xem MLP và CNN dự đoán như thế nào. "
            "**Màu xanh = đúng, màu đỏ = sai.**"
        )
        with gr.Row():
            with gr.Column(scale=1):
                num_c = gr.Slider(1, 50, value=10, step=1, label="Số mẫu (tối đa 50)")
                filter_c = gr.Radio(
                    ["Tất cả (47 lớp)", "Chỉ chữ số (0-9)", "Chỉ chữ hoa (A-Z)",
                     "Chỉ chữ thường (a,b,d,...)"],
                    value="Tất cả (47 lớp)",
                    label="Lọc theo lớp",
                )
                seed_c = gr.Number(value=0, label="Seed (0=random)")
                btn_c = gr.Button("So sánh", variant="primary")
            with gr.Column(scale=3):
                out_img_c = gr.Image(label="Bảng so sánh", type="numpy", height=500)
                out_info_c = gr.Textbox(label="Thống kê chi tiết", lines=15)

        btn_c.click(
            compare_models,
            inputs=[num_c, filter_c, seed_c],
            outputs=[out_img_c, out_info_c],
        )

    gr.Markdown(
        """
        ---
        **Ghi chú:**
        - Ảnh trong EMNIST ByMerge được lưu dạng transpose/flip, notebook gốc không áp dụng
          `np.transpose`/`np.flip` để giữ đúng khớp với model đã train.
        - Độ chính xác trên tập test: **MLP ~88.6%**, **CNN ~90%**.
        - Các cặp ký tự dễ nhầm lẫn: O/0, Z/2, S/5, I/l/1 (do ảnh bị xoay).
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
