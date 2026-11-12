"""
Demo web cho bai toan nhan dang ky tu viet tay EMNIST ByMerge (47 lop).
Ba chuc nang:
  1. Preview mau: hien thi grid cac mau ngau nhien tu tap test.
  2. Test model: chon 1 mau, chay qua MLP va CNN, hien thi du doan + do tu tin.
  3. So sanh MLP vs CNN tren nhieu mau ngau nhien.

Chay:
    python app.py
Sau do mo trinh duyet tai http://127.0.0.1:7860

Luu y:
    - App khoi dong nhanh (~30 giay de load model + data).
    - Predictions duoc tinh theo yeu cau (on-demand), khong pre-compute.
    - Co the loc theo nhom lop: chu so, chu hoa, chu thuong.
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
# Cau hinh
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
TEST_CSV = PROJECT_DIR / "emnist-bymerge-test.csv"
MLP_MODEL = PROJECT_DIR / "mlp_model.keras"
CNN_MODEL = PROJECT_DIR / "cnn_model.keras"

# Label dict giong trong notebook (47 lop, ByMerge)
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

# Nhom lop de filter
DIGIT_LABELS = list(range(0, 10))
UPPER_LABELS = list(range(10, 36))
LOWER_LABELS = list(range(36, 47))


def log(msg):
    """Print voi flush de theo doi tien trinh."""
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Load du lieu + model (1 lan khi khoi dong)
# ---------------------------------------------------------------------------
def load_test_data():
    """Load test CSV, tra ve (images_uint8 [N,28,28], labels [N])."""
    log(f"[INFO] Dang load test data tu {TEST_CSV} ...")
    df = pd.read_csv(TEST_CSV, header=None)
    labels = df.iloc[:, 0].to_numpy().astype(np.int32)
    pixels = df.iloc[:, 1:].to_numpy().astype(np.uint8).reshape(-1, 28, 28)
    log(f"[INFO] Da load {len(labels)} mau test.")
    return pixels, labels


def load_models():
    """Load MLP va CNN, tra ve (mlp, cnn)."""
    log(f"[INFO] Dang load MLP model tu {MLP_MODEL} ...")
    mlp = tf.keras.models.load_model(
        str(MLP_MODEL),
        custom_objects={"relu": tf.nn.relu, "softmax_v2": tf.nn.softmax},
    )
    log(f"[INFO] Dang load CNN model tu {CNN_MODEL} ...")
    cnn = tf.keras.models.load_model(
        str(CNN_MODEL),
        custom_objects={"relu": tf.nn.relu, "softmax_v2": tf.nn.softmax},
    )
    log("[INFO] Load model xong.")
    return mlp, cnn


log("=" * 60)
log("KHOI DONG DEMO - EMNIST HANDWRITING RECOGNITION")
log("=" * 60)

TEST_IMAGES, TEST_LABELS = load_test_data()
MLP_MODEL_OBJ, CNN_MODEL_OBJ = load_models()

log("=" * 60)
log("SAN SANG! Mo trinh duyet tai http://127.0.0.1:7860")
log("=" * 60)


# ---------------------------------------------------------------------------
# Ham tien ich
# ---------------------------------------------------------------------------
def get_indices_by_filter(filter_type: str):
    """Loc index theo nhom lop."""
    if filter_type == "Tat ca (47 lop)":
        return np.arange(len(TEST_LABELS))
    if filter_type == "Chi chu so (0-9)":
        return np.where(np.isin(TEST_LABELS, DIGIT_LABELS))[0]
    if filter_type == "Chi chu hoa (A-Z)":
        return np.where(np.isin(TEST_LABELS, UPPER_LABELS))[0]
    if filter_type == "Chi chu thuong (a,b,d,...)":
        return np.where(np.isin(TEST_LABELS, LOWER_LABELS))[0]
    return np.arange(len(TEST_LABELS))


def make_rng(seed):
    """Tao RNG: seed=0 (hoac None) thi dung random that, nguoc lai dung seed de tai lap."""
    if seed is None or int(seed) == 0:
        return np.random.default_rng()  # random that, moi lan khac nhau
    return np.random.default_rng(int(seed))


def make_image_grid(indices, cols=10):
    """Tao anh grid [rows*28, cols*28] tu danh sach index."""
    n = len(indices)
    rows = (n + cols - 1) // cols
    grid = np.full((rows * 28, cols * 28), 255, dtype=np.uint8)
    for i, idx in enumerate(indices):
        r, c = divmod(i, cols)
        img = TEST_IMAGES[idx]
        # EMNIST trong CSV bi xoay/flip theo nhieu nguon, nhung notebook goc
        # KHONG ap dung np.transpose/flip. Giu nguyen de khop voi model da train.
        grid[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = img
    return grid


def predict_mlp(images_uint8):
    """Predict MLP cho 1 hoac nhieu anh uint8 [N,28,28]. Tra ve probs [N,47]."""
    x = images_uint8.astype(np.float32) / 255.0
    x = x.reshape(-1, 28 * 28)
    return MLP_MODEL_OBJ.predict(x, verbose=0)


def predict_cnn(images_uint8):
    """Predict CNN cho 1 hoac nhieu anh uint8 [N,28,28]. Tra ve probs [N,47]."""
    x = images_uint8.astype(np.float32) / 255.0
    x = x.reshape(-1, 28, 28, 1)
    return CNN_MODEL_OBJ.predict(x, verbose=0)


# ---------------------------------------------------------------------------
# TAB 1: Preview mau
# ---------------------------------------------------------------------------
def preview_samples(num_samples, filter_type, seed):
    """Tra ve (grid_image, info_text)."""
    rng = make_rng(seed)
    pool = get_indices_by_filter(filter_type)
    if len(pool) == 0:
        return np.full((28, 28), 255, dtype=np.uint8), "Khong co mau nao."
    n = int(num_samples)
    n = max(1, min(n, 100, len(pool)))
    chosen = rng.choice(pool, size=n, replace=False)
    chosen = np.sort(chosen)

    grid = make_image_grid(chosen, cols=10)

    # Thong tin: liet ke label cua tung mau
    lines = [f"Da hien thi {n} mau (filter: {filter_type})."]
    lines.append("Label that cua tung mau (theo thu tu trai -> phai, tren -> duoi):")
    for i, idx in enumerate(chosen):
        lines.append(f"  [{i + 1:2d}] idx={idx:6d}  label='{LABELS[int(TEST_LABELS[idx])]}'")
    info = "\n".join(lines)
    return grid, info


# ---------------------------------------------------------------------------
# TAB 2: Test model
# ---------------------------------------------------------------------------
def test_single_sample(index):
    """Test 1 mau voi ca MLP va CNN."""
    idx = int(index)
    if idx < 0 or idx >= len(TEST_LABELS):
        return (
            np.full((28, 28), 255, dtype=np.uint8),
            "Index khong hop le.",
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
    mlp_correct = "DUNG" if mlp_pred == TEST_LABELS[idx] else "SAI"
    cnn_correct = "DUNG" if cnn_pred == TEST_LABELS[idx] else "SAI"

    info_lines = [
        f"Index: {idx}",
        f"Label that (ground truth): '{true_label}'  (class {int(TEST_LABELS[idx])})",
        "",
        f"[MLP]  Du doan: '{LABELS[mlp_pred]}'  |  Do tu tin: {mlp_conf * 100:.2f}%  |  {mlp_correct}",
        f"[CNN]  Du doan: '{LABELS[cnn_pred]}'  |  Do tu tin: {cnn_conf * 100:.2f}%  |  {cnn_correct}",
    ]
    info_text = "\n".join(info_lines)

    # Bang top-5
    def topk_table(probs, k=5):
        top_idx = probs.argsort()[-k:][::-1]
        return [(LABELS[int(j)], f"{float(probs[j]) * 100:.2f}") for j in top_idx]

    mlp_top5 = topk_table(mlp_probs)
    cnn_top5 = topk_table(cnn_probs)

    return img, info_text, mlp_top5, cnn_top5


def random_index(filter_type, seed):
    """Chon ngau nhien 1 index theo filter, tra ve index + label."""
    rng = make_rng(seed)
    pool = get_indices_by_filter(filter_type)
    if len(pool) == 0:
        return 0, "Khong co mau nao."
    idx = int(rng.choice(pool))
    return idx, f"Ngau nhien: idx={idx}, label that='{LABELS[int(TEST_LABELS[idx])]}'"


# ---------------------------------------------------------------------------
# TAB 3: So sanh MLP vs CNN
# ---------------------------------------------------------------------------
def compare_models(num_samples, filter_type, seed):
    """Chay MLP va CNN tren nhieu mau, tra ve bang so sanh + thong ke."""
    rng = make_rng(seed)
    pool = get_indices_by_filter(filter_type)
    if len(pool) == 0:
        return np.full((28, 28), 255, dtype=np.uint8), "Khong co mau nao."

    n = int(num_samples)
    n = max(1, min(n, 50, len(pool)))
    chosen = rng.choice(pool, size=n, replace=False)

    # Lay anh va predict theo batch
    batch = TEST_IMAGES[chosen]
    mlp_probs = predict_mlp(batch)
    cnn_probs = predict_cnn(batch)
    mlp_preds = mlp_probs.argmax(axis=1)
    cnn_preds = cnn_probs.argmax(axis=1)

    # Thong ke
    mlp_correct = int((mlp_preds == TEST_LABELS[chosen]).sum())
    cnn_correct = int((cnn_preds == TEST_LABELS[chosen]).sum())
    both_correct = int(((mlp_preds == TEST_LABELS[chosen]) & (cnn_preds == TEST_LABELS[chosen])).sum())
    only_mlp = int(((mlp_preds == TEST_LABELS[chosen]) & (cnn_preds != TEST_LABELS[chosen])).sum())
    only_cnn = int(((mlp_preds != TEST_LABELS[chosen]) & (cnn_preds == TEST_LABELS[chosen])).sum())
    both_wrong = n - both_correct - only_mlp - only_cnn

    # Tao bang so sanh bang PIL
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
    draw.text((x_img, 5), "Anh", fill="black", font=font_b)
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
        f"=== KET QUA TREN {n} MAU (filter: {filter_type}) ===\n"
        f"MLP dung       : {mlp_correct}/{n}  ({mlp_correct / n * 100:.1f}%)\n"
        f"CNN dung       : {cnn_correct}/{n}  ({cnn_correct / n * 100:.1f}%)\n"
        f"Ca hai dung    : {both_correct}/{n}\n"
        f"Chi MLP dung   : {only_mlp}/{n}\n"
        f"Chi CNN dung   : {only_cnn}/{n}\n"
        f"Ca hai sai     : {both_wrong}/{n}\n"
        f"\nChi tiet (xanh = dung, do = sai):\n"
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
# Giao dien Gradio
# ---------------------------------------------------------------------------
with gr.Blocks(title="EMNIST Handwriting Recognition Demo") as demo:
    gr.Markdown(
        """
        # Demo nhan dang ky tu viet tay - EMNIST ByMerge (47 lop)
        Dataset gom **chu so (0-9)**, **chu hoa (A-Z)** va **11 chu thuong** (a, b, d, e, f, g, h, n, q, r, t).
        Model da train: **MLP** (accuracy ~88.6%) va **CNN** (accuracy ~90%).
        """
    )

    # ----- TAB 1: Preview mau -----
    with gr.Tab("1. Preview mau"):
        gr.Markdown(
            "Hien thi grid cac mau ngau nhien tu tap test. Co the loc theo nhom lop."
        )
        with gr.Row():
            with gr.Column(scale=1):
                num_p = gr.Slider(1, 100, value=20, step=1, label="So mau (toi da 100)")
                filter_p = gr.Radio(
                    ["Tat ca (47 lop)", "Chi chu so (0-9)", "Chi chu hoa (A-Z)",
                     "Chi chu thuong (a,b,d,...)"],
                    value="Tat ca (47 lop)",
                    label="Loc theo lop",
                )
                seed_p = gr.Number(
                    value=0,
                    label="Seed (0 = random that; nhap so khac de tai lap ket qua)",
                )
                btn_p = gr.Button("Lay mau ngau nhien", variant="primary")
            with gr.Column(scale=3):
                out_img_p = gr.Image(label="Grid cac mau", type="numpy", height=400)
                out_info_p = gr.Textbox(label="Thong tin label", lines=10)

        btn_p.click(
            preview_samples,
            inputs=[num_p, filter_p, seed_p],
            outputs=[out_img_p, out_info_p],
        )

    # ----- TAB 2: Test model -----
    with gr.Tab("2. Test model tren 1 mau"):
        gr.Markdown(
            "Chon 1 mau cu the (theo index) hoac lay ngau nhien, sau do xem du doan cua MLP va CNN."
        )
        with gr.Row():
            with gr.Column(scale=1):
                idx_t = gr.Number(value=0, label="Index mau (0 - 116322)", precision=0)
                with gr.Row():
                    filter_t = gr.Radio(
                        ["Tat ca (47 lop)", "Chi chu so (0-9)", "Chi chu hoa (A-Z)",
                         "Chi chu thuong (a,b,d,...)"],
                        value="Tat ca (47 lop)",
                        label="Loc khi lay ngau nhien",
                        scale=2,
                    )
                    seed_t = gr.Number(value=0, label="Seed (0=random)", scale=1)
                btn_rand = gr.Button("Lay ngau nhien 1 mau")
                rand_info = gr.Textbox(label="Index ngau nhien", lines=1)
                btn_test = gr.Button("Chay MLP + CNN", variant="primary")
            with gr.Column(scale=2):
                out_img_t = gr.Image(label="Anh input (28x28)", type="numpy", height=300)
                out_info_t = gr.Textbox(label="Ket qua du doan", lines=8)
                gr.Markdown("**Top-5 du doan (xac suat cao nhat)**")
                with gr.Row():
                    mlp_table = gr.Dataframe(
                        headers=["Ky tu", "Xac suat (%)"], label="MLP", wrap=True
                    )
                    cnn_table = gr.Dataframe(
                        headers=["Ky tu", "Xac suat (%)"], label="CNN", wrap=True
                    )

        btn_rand.click(
            random_index, inputs=[filter_t, seed_t], outputs=[idx_t, rand_info]
        )
        btn_test.click(
            test_single_sample,
            inputs=[idx_t],
            outputs=[out_img_t, out_info_t, mlp_table, cnn_table],
        )

    # ----- TAB 3: So sanh MLP vs CNN -----
    with gr.Tab("3. So sanh MLP vs CNN"):
        gr.Markdown(
            "Chon nhieu mau ngau nhien, xem MLP va CNN du doan nhu the nao. "
            "**Mau xanh = dung, mau do = sai.**"
        )
        with gr.Row():
            with gr.Column(scale=1):
                num_c = gr.Slider(1, 50, value=10, step=1, label="So mau (toi da 50)")
                filter_c = gr.Radio(
                    ["Tat ca (47 lop)", "Chi chu so (0-9)", "Chi chu hoa (A-Z)",
                     "Chi chu thuong (a,b,d,...)"],
                    value="Tat ca (47 lop)",
                    label="Loc theo lop",
                )
                seed_c = gr.Number(value=0, label="Seed (0=random)")
                btn_c = gr.Button("So sanh", variant="primary")
            with gr.Column(scale=3):
                out_img_c = gr.Image(label="Bang so sanh", type="numpy", height=500)
                out_info_c = gr.Textbox(label="Thong ke chi tiet", lines=15)

        btn_c.click(
            compare_models,
            inputs=[num_c, filter_c, seed_c],
            outputs=[out_img_c, out_info_c],
        )

    gr.Markdown(
        """
        ---
        **Ghi chu:**
        - Anh trong EMNIST ByMerge duoc luu dang transpose/flip, notebook goc khong ap dung
          `np.transpose`/`np.flip` de giu dung khop voi model da train.
        - Do chinh xac tren tap test: **MLP ~88.6%**, **CNN ~90%**.
        - Cac cap ky tu de nham lan: O/0, Z/2, S/5, I/l/1 (do anh bi xoay).
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
