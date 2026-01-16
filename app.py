import io
import os
from pathlib import Path
import zipfile
import pandas as pd
import time



import streamlit as st
import torch
from PIL import Image

# Import from your main.py
from main import (
    Model1, Model2, Model3, Model4, Model5,
    Model6, Model7, Model8, Model9, Model10,
    preprocess_image_for_mnist
)
hidden_dim = 64
dropout_p = 0.25
MODEL_METADATA = {
    "hidden_dim": 64,
    "dropout_p": 0.25,
    "kernel_size": 5,
    "epochs": 10,
    "batch_size": 128,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "loss_function": "CrossEntropyLoss",
    "dataset": "MNIST",
    "input_shape": "1×28×28",
}

st.set_page_config(page_title="MNIST Model Tester", layout="wide")


@st.cache_resource
def load_model(model_number: int, device: str, hidden_dim: int = 64, dropout_p: float = 0.25):
    """Load selected model + weights once (cached)."""
    model_classes = [Model1, Model2, Model3, Model4, Model5, Model6, Model7, Model8, Model9, Model10]
    ModelClass = model_classes[model_number - 1]

    model = ModelClass(hidden_dim=hidden_dim, dropout_p=dropout_p).to(device)
    weights_path = Path(f"model_hidden_layers_{model_number}.pth")

    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file: {weights_path}. Train first with: python main.py (from terminal)")

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
    

def is_image_filename(name: str) -> bool:
    """Check if a filename has an image extension."""
    name = name.lower()
    return name.endswith((".png", ".jpg", ".jpeg", ".bmp"))


def extract_images_from_zip(zip_uploaded_file):
    """
    Takes a Streamlit UploadedFile (zip) and returns a list of in-memory UploadedFile-like objects.
    We will return tuples: (filename, bytes)
    """
    images = []
    zbytes = zip_uploaded_file.getvalue()

    with zipfile.ZipFile(io.BytesIO(zbytes), "r") as z:
        for name in z.namelist():
            # skip directories
            if name.endswith("/"):
                continue
            if is_image_filename(name):
                images.append((Path(name).name, z.read(name)))  # keep base name

    return images





def infer_true_label_from_name(filename: str):
    """
    If filename starts with a digit 0-9, return it as int. Else None.
    """
    return int(filename[0]) if filename and filename[0].isdigit() else None


@torch.no_grad()
def predict_from_bytes(model, device: str, filename: str, b: bytes):
    """
    Predict top-3 from raw image bytes.
    """
    img = Image.open(io.BytesIO(b)).convert("RGB")
    pred, conf, top3, _ = predict_top3(model, device, img)
    true_label = infer_true_label_from_name(filename)

    return {
        "filename": filename,
        "true": true_label if true_label is not None else "",
        "pred": pred,
        "confidence": conf,
        "top2": top3[1][0], "top2_prob": top3[1][1],
        "top3": top3[2][0], "top3_prob": top3[2][1],
        "correct": (pred == true_label) if true_label is not None else ""
    }


@torch.no_grad()
def predict_top3(model, device: str, image_pil: Image.Image):
    """
    Predict top-3 digits from a PIL image.
    Passes PIL image directly to preprocess_image_for_mnist to avoid file I/O issues.
    """
    x = preprocess_image_for_mnist(image_pil=image_pil).to(device)  # [1,1,28,28]
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]  # [10]

    top_probs, top_idx = torch.topk(probs, k=3)
    top3 = [(int(top_idx[i].item()), float(top_probs[i].item())) for i in range(3)]

    pred = top3[0][0]
    conf = top3[0][1]

    # Create 28x28 preview image from tensor (0..1)
    # x: [1,1,28,28]
    x_img = (x[0, 0].detach().cpu().numpy() * 255).clip(0, 255).astype("uint8")

    return pred, conf, top3, x_img


def run_all_models_on_dataset(device, hidden_dim, dropout_p, rows_source):
    """
    rows_source: list of dicts with at least:
      - filename
      - bytes  (raw image bytes)
    Returns:
      summary_df: per-model summary (accuracy, avg_conf, n_images, n_labeled, time)
      all_preds_df: per-image predictions for all models (optional for inspection)
    """
    model_numbers = list(range(1, 11))

    # Prepare data once: list of (filename, bytes, true_label)
    data = []
    for item in rows_source:
        fname = item["filename"]
        b = item["bytes"]
        true_label = infer_true_label_from_name(fname)
        data.append((fname, b, true_label))

    summaries = []
    all_rows = []

    for m in model_numbers:
        t0 = time.time()
        model = load_model(m, device, hidden_dim=hidden_dim, dropout_p=dropout_p)

        correct = 0
        labeled = 0
        conf_sum = 0.0

        for fname, b, true_label in data:
            img = Image.open(io.BytesIO(b)).convert("RGB")
            pred, conf, top3, _ = predict_top3(model, device, img)

            conf_sum += conf

            if true_label is not None:
                labeled += 1
                if pred == true_label:
                    correct += 1

            all_rows.append({
                "model": m,
                "filename": fname,
                "true": true_label if true_label is not None else "",
                "pred": pred,
                "confidence": conf,
                "top2": top3[1][0], "top2_prob": top3[1][1],
                "top3": top3[2][0], "top3_prob": top3[2][1],
                "correct": (pred == true_label) if true_label is not None else ""
            })

        elapsed = time.time() - t0
        avg_conf = conf_sum / len(data) if len(data) > 0 else 0.0
        acc = (correct / labeled) if labeled > 0 else None

        summaries.append({
            "model": m,
            "accuracy": acc if acc is not None else "",
            "avg_confidence": avg_conf,
            "n_images": len(data),
            "n_labeled": labeled,
            "time_s": elapsed
        })

    summary_df = pd.DataFrame(summaries)
    all_preds_df = pd.DataFrame(all_rows)
    return summary_df, all_preds_df



def main():
    st.markdown("<h1 style='text-align: center;'>MNIST Handwritten Digit – Model Tester</h1>", unsafe_allow_html=True)
    st.write("Pick a model, upload an image or dataset, and see accuracy and confidence results")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with st.sidebar:
        st.header("Settings")
        st.caption(f"Device: **{device}**")

        model_number = st.selectbox(
            "Choose model",
            options=list(range(1, 11)),
            index=2
        )

        st.markdown("---")
        st.subheader("Model & training configuration")

        for k, v in MODEL_METADATA.items():
            label = k.replace("_", " ").title()
            st.markdown(f"- **{label}:** `{v}`")



    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("1) Test input")
        tab_single, tab_dataset = st.tabs(["Single image", "Dataset (folder/zip)"])

        # -------------------
        # Tab 1: Single image
        # -------------------
        with tab_single:
            uploaded = st.file_uploader("Upload one image (png/jpg/jpeg/bmp)", type=["png", "jpg", "jpeg", "bmp"])

            if uploaded is not None:
                img = Image.open(uploaded).convert("RGB")
                st.image(img, caption="Uploaded image", use_container_width=True)

                show_processed = st.checkbox("Show preprocessed input (28×28)")

                if st.button("Predict (single image)"):
                    try:
                        model = load_model(model_number, device, hidden_dim=hidden_dim, dropout_p=dropout_p)
                        pred, conf, top3, x_img = predict_top3(model, device, img)

                        st.success(f"Prediction: **{pred}**   |   Confidence: **{conf*100:.2f}%**")
                        st.write("Top-3:")
                        st.table([{"digit": d, "probability": f"{p*100:.2f}%"} for d, p in top3])

                        if show_processed:
                            st.write("Model input (28×28 after preprocessing):")
                            # Convert numpy array to PIL Image for proper display
                            x_img_pil = Image.fromarray(x_img, mode="L")
                            st.image(x_img_pil, caption="28×28 input", width=200)

                    except Exception as e:
                        st.error(str(e))
            else:
                st.info("Upload an image to begin.")

        # -------------------
        # Tab 2: Dataset mode
        # -------------------
        with tab_dataset:
            run_all = st.checkbox("Run ALL models (1–10) and compare accuracy", value=False)
            st.write("Upload a dataset as either:")
            st.markdown("- **Many images** (drag & drop multiple files)\n- **A ZIP file** (best way to upload a whole folder)")

            multi_files = st.file_uploader(
                "Upload multiple images",
                type=["png", "jpg", "jpeg", "bmp"],
                accept_multiple_files=True
            )

            zip_file = st.file_uploader("Or upload a ZIP containing images", type=["zip"])

            run_dataset = st.button("Run dataset prediction")

            if run_dataset:
                model = load_model(model_number, device, hidden_dim=hidden_dim, dropout_p=dropout_p)

                rows = []

                # 1) Images uploaded directly
                if multi_files:
                    for uf in multi_files:
                        b = uf.getvalue()
                        rows.append(predict_from_bytes(model, device, uf.name, b))

                # 2) Images inside zip (folder upload)
                if zip_file is not None:
                    extracted = extract_images_from_zip(zip_file)
                    if not extracted:
                        st.warning("No image files found inside the ZIP.")
                    else:
                        for fname, b in extracted:
                            rows.append(predict_from_bytes(model, device, fname, b))

                    if not rows:
                        st.warning("Please upload multiple images or a ZIP file first.")
                    else:
                        # Build a unified list of dataset items: filename + bytes
                        dataset_items = []

                        # multi_files
                        if multi_files:
                            for uf in multi_files:
                                dataset_items.append({"filename": uf.name, "bytes": uf.getvalue()})

                        # zip files
                        if zip_file is not None:
                            extracted = extract_images_from_zip(zip_file)
                            if not extracted:
                                st.warning("No image files found inside the ZIP.")
                            else:
                                for fname, b in extracted:
                                    dataset_items.append({"filename": fname, "bytes": b})

                        if not dataset_items:
                            st.warning("Please upload multiple images or a ZIP file first.")
                        else:
                            if run_all:
                                st.info("Running all 10 models… this can take a moment depending on dataset size.")

                                summary_df, all_preds_df = run_all_models_on_dataset(
                                    device=device,
                                    hidden_dim=hidden_dim,
                                    dropout_p=dropout_p,
                                    rows_source=dataset_items
                                )

                                st.subheader("Per-model summary")
                                st.dataframe(summary_df, use_container_width=True)

                                # Accuracy plot (only if labels exist)
                                labeled_any = (summary_df["n_labeled"] > 0).any()
                                if labeled_any:
                                    # Keep only models with numeric accuracies
                                    plot_df = summary_df.copy()
                                    plot_df = plot_df[plot_df["accuracy"] != ""]
                                    
                                    if len(plot_df) > 0:
                                        plot_df["accuracy"] = pd.to_numeric(plot_df["accuracy"], errors='coerce')
                                        plot_df = plot_df.dropna(subset=["accuracy"])
                                        
                                        if len(plot_df) > 0:
                                            st.subheader("Accuracy vs Model (Hidden Layers)")
                                            st.line_chart(plot_df.set_index("model")["accuracy"])
                                            st.caption("Model number = number of FC hidden layers.")
                                        else:
                                            st.info("No valid accuracy data to plot.")
                                    else:
                                        st.info("No labeled images found for accuracy calculation.")
                                else:
                                    st.warning("No labels found in filenames (must start with 0–9). Accuracy cannot be computed.")

                                # Download full predictions CSV
                                csv_bytes = all_preds_df.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    "Download ALL predictions (all models) as CSV",
                                    data=csv_bytes,
                                    file_name="dataset_predictions_all_models.csv",
                                    mime="text/csv"
                                )

                            else:
                                # Single-model mode (your existing behavior)
                                model = load_model(model_number, device, hidden_dim=hidden_dim, dropout_p=dropout_p)

                                rows = []
                                for item in dataset_items:
                                    rows.append(predict_from_bytes(model, device, item["filename"], item["bytes"]))

                                df = pd.DataFrame(rows)

                                labeled = df[df["true"] != ""]
                                if len(labeled) > 0:
                                    acc = (labeled["correct"] == True).mean()
                                    st.success(f"Dataset accuracy (based on filename labels): **{acc*100:.2f}%**  ({len(labeled)} labeled images)")
                                else:
                                    st.info("No filename labels detected (filenames must start with 0–9 to compute accuracy).")

                                st.dataframe(df, use_container_width=True)

                                csv_bytes = df.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    "Download results as CSV",
                                    data=csv_bytes,
                                    file_name=f"dataset_predictions_model{model_number}.csv",
                                    mime="text/csv"
                                )


    with col_right:
        st.subheader("2) Training plots")
        plot1 = Path("test_loss_all_models.png")

        if plot1.exists():
            st.image(str(plot1), caption="Test loss vs epoch (Models 1–10)", use_container_width=True)
        else:
            st.warning("Missing test_loss_all_models.png (run training first).")

        with st.expander("Show available weight files (.pth)"):
            weights = sorted(Path(".").glob("model_hidden_layers_*.pth"))
            if weights:
                st.write([w.name for w in weights])
            else:
                st.write("No .pth files found. Train first with: python main.py")


if __name__ == "__main__":
    main()
