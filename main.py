import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import random
import numpy as np
import csv
import matplotlib.pyplot as plt
import argparse


from pathlib import Path
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CUDA deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess_image_for_mnist(image_path: str = None, image_pil: Image.Image = None) -> torch.Tensor:
    """
    - If image is already 28x28: do NOT crop/resize
    - Otherwise:
        * grayscale
        * invert if background is bright (keeps MNIST convention: bright digit on dark bg)
        * gentle autocontrast (does NOT binarize, keeps noise)
        * find digit bbox using a *soft mask* (only for locating content)
        * crop using that bbox but keep original grayscale values
        * pad to square (no stretching)
        * resize to 28x28 (LANCZOS)
    - Output: tensor [1,1,28,28] in [0,1]
    
    Args:
        image_path: path to image file (for file-based usage)
        image_pil: PIL Image object (for in-memory usage)
    """
    if image_pil is not None:
        img = image_pil.convert("L")
    elif image_path is not None:
        img = Image.open(image_path).convert("L")
    else:
        raise ValueError("Must provide either image_path or image_pil")

    #If already MNIST size, keep geometry unchanged
    if img.size == (28, 28):
        mean_val = sum(img.get_flattened_data()) / (28 * 28)
        if mean_val > 127:
            img = ImageOps.invert(img)
        x = transforms.ToTensor()(img).unsqueeze(0)
        return x

    #Invert if background is bright
    mean_val = sum(img.get_flattened_data()) / (img.size[0] * img.size[1])
    if mean_val > 127:
        img = ImageOps.invert(img)

    #Gentle contrast normalization (preserves noise; no binarization)
    img = ImageOps.autocontrast(img)

    #Compute a SOFT mask just to find bbox (doesn't change the image)
    #We treat pixels > t as "ink" for bbox detection only.
    # Use a low-ish threshold so faint strokes still count.
    t = 20
    mask = img.point(lambda p: 255 if p > t else 0)
    bbox = mask.getbbox()
    if bbox is not None:
        img = img.crop(bbox)
    else:
        pass  # keep as-is (rare), continue

    #Pad to square to avoid stretching
    w, h = img.size
    side = max(w, h)
    pad_left = (side - w) // 2
    pad_top = (side - h) // 2
    pad_right = side - w - pad_left
    pad_bottom = side - h - pad_top
    img = ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)

    # 5) Resize to 28x28 using high-quality downsampling
    img = img.resize((28, 28), resample=Image.Resampling.LANCZOS)

    # 6) To tensor + batch dim
    x = transforms.ToTensor()(img).unsqueeze(0)
    return x

@torch.no_grad()
def predict_digit(model: nn.Module, image_path: str, device: torch.device):
    model.eval()

    x = preprocess_image_for_mnist(image_path).to(device)  # [1,1,28,28]
    logits = model(x)                                      # [1,10]

    probs = torch.softmax(logits, dim=1)                   # probabilities
    pred = probs.argmax(dim=1).item()
    confidence = probs[0, pred].item()

    return pred, confidence


#Base model
class BaseCNN(nn.Module):
    """
    CNN feature extractor (same for all models) + classifier (varies by hidden layers).

    Input:  [B, 1, 28, 28]
    Output: [B, 10] logits
    """

    def __init__(self, num_hidden_layers: int, hidden_dim: int = 64, dropout_p: float = 0.25):
        super().__init__()

        #Feature extractor
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout2d = nn.Dropout2d(p=dropout_p)

        # After conv/pool, MNIST becomes 20*4*4 = 320 features
        feature_dim = 320

        #Classifier (FC part): depth varies
        layers = []

        #First hidden layer: 320 -> hidden_dim
        layers.append(nn.Linear(feature_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_p))

        #Extra hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_p))

        #Output layer: hidden_dim -> 10
        layers.append(nn.Linear(hidden_dim, 10))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        #Conv block 1
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #Conv block 2
        x = self.dropout2d(self.conv2(x))
        x = F.relu(F.max_pool2d(x, 2))

        #Flatten [B,20,4,4] -> [B,320]
        x = x.view(x.size(0), -1)

        #Classifier returns logits
        logits = self.classifier(x)
        return logits



#10 model classes 
class Model1(BaseCNN):
    def __init__(self, hidden_dim=64, dropout_p=0.25):
        super().__init__(num_hidden_layers=1, hidden_dim=hidden_dim, dropout_p=dropout_p)

class Model2(BaseCNN):
    def __init__(self, hidden_dim=64, dropout_p=0.25):
        super().__init__(num_hidden_layers=2, hidden_dim=hidden_dim, dropout_p=dropout_p)

class Model3(BaseCNN):
    def __init__(self, hidden_dim=64, dropout_p=0.25):
        super().__init__(num_hidden_layers=3, hidden_dim=hidden_dim, dropout_p=dropout_p)

class Model4(BaseCNN):
    def __init__(self, hidden_dim=64, dropout_p=0.25):
        super().__init__(num_hidden_layers=4, hidden_dim=hidden_dim, dropout_p=dropout_p)

class Model5(BaseCNN):
    def __init__(self, hidden_dim=64, dropout_p=0.25):
        super().__init__(num_hidden_layers=5, hidden_dim=hidden_dim, dropout_p=dropout_p)

class Model6(BaseCNN):
    def __init__(self, hidden_dim=64, dropout_p=0.25):
        super().__init__(num_hidden_layers=6, hidden_dim=hidden_dim, dropout_p=dropout_p)

class Model7(BaseCNN):
    def __init__(self, hidden_dim=64, dropout_p=0.25):
        super().__init__(num_hidden_layers=7, hidden_dim=hidden_dim, dropout_p=dropout_p)

class Model8(BaseCNN):
    def __init__(self, hidden_dim=64, dropout_p=0.25):
        super().__init__(num_hidden_layers=8, hidden_dim=hidden_dim, dropout_p=dropout_p)

class Model9(BaseCNN):
    def __init__(self, hidden_dim=64, dropout_p=0.25):
        super().__init__(num_hidden_layers=9, hidden_dim=hidden_dim, dropout_p=dropout_p)

class Model10(BaseCNN):
    def __init__(self, hidden_dim=64, dropout_p=0.25):
        super().__init__(num_hidden_layers=10, hidden_dim=hidden_dim, dropout_p=dropout_p)



#Train + test functions
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

    return running_loss / total, correct / total


@torch.no_grad()
def test(model, loader, loss_fn, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

    return running_loss / total, correct / total



#Main: train all 10 models
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Hyperparameters
    batch_size = 128
    epochs = 10          
    lr = 0.001
    hidden_dim = 64
    dropout_p = 0.25

    seed = 42
    set_seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    loss_fn = nn.CrossEntropyLoss()

    model_classes = [Model1, Model2, Model3, Model4, Model5, Model6, Model7, Model8, Model9, Model10]
    results = []

    # Store per-epoch history for plotting + saving
    history = {}

    for i, ModelClass in enumerate(model_classes, start=1):
        print("\n" + "=" * 60)
        print(f"Training Model{i} (hidden FC layers = {i})")
        print("=" * 60)

        model = ModelClass(hidden_dim=hidden_dim, dropout_p=dropout_p).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        history[i] = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

        start = time.time()

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            test_loss, test_acc = test(model, test_loader, loss_fn, device)

            # Save epoch metrics
            history[i]["train_loss"].append(train_loss)
            history[i]["test_loss"].append(test_loss)
            history[i]["train_acc"].append(train_acc)
            history[i]["test_acc"].append(test_acc)

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train loss {train_loss:.4f}, acc {train_acc*100:.2f}% | "
                f"Test loss {test_loss:.4f}, acc {test_acc*100:.2f}%"
            )

        elapsed = time.time() - start
        results.append((i, history[i]["test_acc"][-1], history[i]["test_loss"][-1], elapsed))

        #Save each trained model weights
        torch.save(model.state_dict(), f"model_hidden_layers_{i}.pth")
        print(f"Saved model_hidden_layers_{i}.pth")

    
    #Save history to CSV
    with open("loss_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "epoch", "train_loss", "test_loss", "train_acc", "test_acc"])

        for model_num, h in history.items():
            for epoch_idx in range(epochs):
                writer.writerow([
                    model_num,
                    epoch_idx + 1,
                    h["train_loss"][epoch_idx],
                    h["test_loss"][epoch_idx],
                    h["train_acc"][epoch_idx],
                    h["test_acc"][epoch_idx],
                ])

    print("Saved loss history to loss_history.csv")

    
    #Plot: Test loss for all models
    plt.figure()
    for model_num, h in history.items():
        x = range(1, epochs + 1)
        plt.plot(x, h["test_loss"], label=f"Model {model_num}")
    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")
    plt.title("Test Loss vs Epoch (Models 1–10)")
    plt.legend()
    plt.grid(True)
    plt.savefig("test_loss_all_models.png", dpi=200)
    plt.show()
    print("Saved plot to test_loss_all_models.png")
    
    #Final summary
    print("\n" + "#" * 60)
    print("FINAL SUMMARY")
    print("#" * 60)
    print(f"{'Model':>5} | {'HiddenLayers':>12} | {'TestAcc%':>8} | {'TestLoss':>8} | {'Time(s)':>8}")
    print("-" * 60)
    for model_num, acc, loss, t in results:
        print(f"{model_num:>5} | {model_num:>12} | {acc*100:>8.2f} | {loss:>8.4f} | {t:>8.1f}")



def list_images_in_folder(folder: str, recursive: bool = False):
    """
    Returns a list of image file paths in a folder.
    Supports: .png, .jpg, .jpeg, .bmp
    """
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    folder_path = Path(folder)

    if recursive:
        files = [p for p in folder_path.rglob("*") if p.suffix.lower() in exts]
    else:
        files = [p for p in folder_path.glob("*") if p.suffix.lower() in exts]

    return sorted(files)


def parse_true_label_from_filename(path: Path):
    """
    If file is named like '8_something.png' or '8.png',
    returns 8. Otherwise returns None.
    """
    name = path.stem  # filename without extension
    if len(name) > 0 and name[0].isdigit():
        return int(name[0])
    return None


@torch.no_grad()
def predict_one(model: nn.Module, image_path: str, device: torch.device):
    """
    Runs prediction and returns:
    pred (int), confidence (float 0..1), top3 list [(digit, prob), ...]
    """
    model.eval()
    x = preprocess_image_for_mnist(image_path).to(device)  # [1,1,28,28]
    logits = model(x)

    probs = torch.softmax(logits, dim=1)[0]  # [10]
    top_probs, top_idx = torch.topk(probs, k=3)
    top3 = [(top_idx[i].item(), top_probs[i].item()) for i in range(3)]

    pred = top3[0][0]
    conf = top3[0][1]
    return pred, conf, top3


def load_model_by_number(model_number: int, device: torch.device, hidden_dim=64, dropout_p=0.25):
    """
    Builds Model{n}, loads weights from model_hidden_layers_{n}.pth, returns model.
    """
    model_classes = [Model1, Model2, Model3, Model4, Model5, Model6, Model7, Model8, Model9, Model10]
    ModelClass = model_classes[model_number - 1]

    model = ModelClass(hidden_dim=hidden_dim, dropout_p=dropout_p).to(device)
    weight_path = f"model_hidden_layers_{model_number}.pth"
    model.load_state_dict(torch.load(weight_path, map_location=device))
    return model, weight_path


def run_folder_predictions(model_numbers, folder, device, out_csv=None, recursive=False):
    """
    Runs predictions for one or multiple models over all images in a folder.
    Saves CSV if out_csv is provided.
    Prints a quick summary + optional accuracy (if filename labels exist).
    """
    images = list_images_in_folder(folder, recursive=recursive)
    if not images:
        print(f"No images found in: {folder}")
        return

    # Prepare CSV output
    writer = None
    f = None
    if out_csv is not None:
        f = open(out_csv, "w", newline="", encoding="utf-8")
        writer = csv.writer(f)
        writer.writerow([
            "model", "image_path", "true_label",
            "pred", "confidence",
            "top2_digit", "top2_prob",
            "top3_digit", "top3_prob"
        ])

    # For each model, load once and run through all images
    for m in model_numbers:
        model, weight_path = load_model_by_number(m, device)
        print(f"\nModel {m} loaded from {weight_path}")

        correct = 0
        total_labeled = 0

        for img_path in images:
            true_label = parse_true_label_from_filename(img_path)
            pred, conf, top3 = predict_one(model, str(img_path), device)

            if true_label is not None:
                total_labeled += 1
                if pred == true_label:
                    correct += 1

            # Print one-line result
            print(f"{img_path.name:30s} -> pred {pred} ({conf*100:5.1f}%)", end="")
            if true_label is not None:
                print(f" | true {true_label}")
            else:
                print()

            # Save to CSV
            if writer is not None:
                writer.writerow([
                    m,
                    str(img_path),
                    true_label if true_label is not None else "",
                    pred,
                    conf,
                    top3[1][0], top3[1][1],
                    top3[2][0], top3[2][1],
                ])

        if total_labeled > 0:
            acc = correct / total_labeled
            print(f"Model {m} accuracy on labeled filenames: {acc*100:.2f}% ({correct}/{total_labeled})")

    if f is not None:
        f.close()
        print(f"\nSaved results to: {out_csv}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST 10-model experiment")
    subparsers = parser.add_subparsers(dest="command")

    # ---- Predict subcommand ----
    p_predict = subparsers.add_parser("predict", help="Run inference on an image or folder")
    p_predict.add_argument("--model", type=int, choices=range(1, 11), help="Model number 1-10")
    p_predict.add_argument("--models", type=str, default=None, help='Use "all" to run models 1-10')
    p_predict.add_argument("--image", type=str, default=None, help="Path to a single image")
    p_predict.add_argument("--folder", type=str, default=None, help="Path to a folder of images")
    p_predict.add_argument("--recursive", action="store_true", help="Search folder recursively")
    p_predict.add_argument("--out", type=str, default=None, help="Optional CSV output path")

    args = parser.parse_args()

    if args.command == "predict":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

        #Decide which models to run
        if args.models is not None and args.models.lower() == "all":
            model_numbers = list(range(1, 11))
        elif args.model is not None:
            model_numbers = [args.model]
        else:
            raise SystemExit("You must provide --model N or --models all")

        #Decide what input to run on
        if args.image is not None:
            # Single image: run each model on the image
            for m in model_numbers:
                model, weight_path = load_model_by_number(m, device)
                pred, conf, top3 = predict_one(model, args.image, device)
                print(f"\nModel {m} ({weight_path})")
                print(f"Prediction: {pred} | Confidence: {conf*100:.2f}%")
                print("Top-3:", ", ".join([f"{d} ({p*100:.1f}%)" for d, p in top3]))

        elif args.folder is not None:
            run_folder_predictions(
                model_numbers=model_numbers,
                folder=args.folder,
                device=device,
                out_csv=args.out,
                recursive=args.recursive
            )
        else:
            raise SystemExit("You must provide --image PATH or --folder PATH")

    else:
        #Default: train 
        main()



