import os
import io
import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict

from flask import Flask, request, jsonify, render_template_string, send_file
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# ---------------------------
# Configuration
# ---------------------------
# Default locations to try for model file (will pick first existing)
MODEL_CANDIDATES = [
    os.environ.get("MODEL_PATH", ""),  # allow override
    "/content/drive/MyDrive/disaster_classification_project/saved_models/model_best.pth",
    "saved_models/model_best.pth",
    "saved_models/best_model.pth",
    "/content/drive/MyDrive/disaster_classification/saved_models/model_best.pth",
    "/content/drive/MyDrive/disaster_classification/model_best.pth",
    "/content/drive/MyDrive/disaster_classification_project/saved_models/model_best.pth",
    "/content/drive/MyDrive/disaster_classification/reorganized/saved_models/model_best.pth",
]
MODEL_CANDIDATES = [p for p in MODEL_CANDIDATES if p]

# runtime config
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB upload limit
TOP_K_DEFAULT = 3
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# default class names fallback if checkpoint doesn't contain them
FALLBACK_CLASS_NAMES = ['earthquake','flood','hurricane','landslide','mild','none','severe']

# set up logging
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logger.addHandler(ch)

# ---------------------------
# Utilities
# ---------------------------
def find_model_path() -> str:
    for p in MODEL_CANDIDATES:
        if p and os.path.exists(p):
            logger.info(f"Using model file: {p}")
            return p
    raise FileNotFoundError(f"No model found in candidates: {MODEL_CANDIDATES}")

def load_checkpoint(path: str, map_location="cpu"):
    """Load checkpoint, return raw object from torch.load"""
    logger.info(f"Loading checkpoint from {path} ...")
    ckpt = torch.load(path, map_location=map_location)
    logger.info("Checkpoint loaded.")
    return ckpt

def extract_state_and_labels(ckpt) -> Tuple[dict, List[str]]:
    """
    Accepts many checkpoint shapes:
      - {'model_state': state_dict, ...}
      - {'model_state_dict': state_dict, ...}
      - raw state_dict itself
      - may contain 'idx_to_label' or 'label_to_idx'
    Returns (state_dict, class_names_list_or_None)
    """
    state = None
    class_names = None

    if isinstance(ckpt, dict):
        # look for common keys
        for key in ["model_state", "state_dict", "model_state_dict", "model"]:
            if key in ckpt:
                state = ckpt[key]
                break
        # Sometimes state dict is top-level (keys like 'fc.weight'), handle below
        # label mappings
        if "idx_to_label" in ckpt:
            idx_map = ckpt["idx_to_label"]
            # idx_to_label might be dict or list
            if isinstance(idx_map, dict):
                # convert dict to list ordered by index
                try:
                    class_names = [idx_map[str(i)] if str(i) in idx_map else idx_map[i] for i in range(len(idx_map))]
                except Exception:
                    class_names = [idx_map[k] for k in sorted(idx_map.keys())]
            elif isinstance(idx_map, list):
                class_names = idx_map
        elif "label_to_idx" in ckpt:
            l2i = ckpt["label_to_idx"]
            try:
                # invert
                inv = {v:k for k,v in l2i.items()}
                class_names = [inv[i] for i in range(len(inv))]
            except Exception:
                # best effort
                class_names = list(l2i.keys())
    # if state still None and ckpt looks like raw state_dict (keys include 'fc.weight')
    if state is None:
        # assume ckpt is raw state_dict (mapping of param names -> tensors)
        if isinstance(ckpt, dict) and any(k.endswith("fc.weight") or "fc.weight" in k for k in ckpt.keys()):
            state = ckpt
    return state, class_names

def infer_num_classes_from_state(state: dict) -> int:
    """Try to infer number of classes from state dict by looking at fc.weight shape"""
    for candidate in ["fc.weight", "classifier.weight", "head.weight"]:
        if candidate in state:
            return int(state[candidate].shape[0])
    # look for any weight that endswith 'fc.weight'
    for k in state.keys():
        if k.endswith("fc.weight"):
            return int(state[k].shape[0])
    # fallback None -> caller must handle
    return None

def build_model(num_classes: int):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ---------------------------
# Load model once at startup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

try:
    MODEL_PATH = find_model_path()
except FileNotFoundError as e:
    logger.error(str(e))
    MODEL_PATH = None

model = None
class_names = None

if MODEL_PATH:
    try:
        raw_ckpt = load_checkpoint(MODEL_PATH, map_location=device)
        state_dict, ckpt_class_names = extract_state_and_labels(raw_ckpt)
        inferred_num = None
        if state_dict is not None:
            inferred_num = infer_num_classes_from_state(state_dict)
        if ckpt_class_names:
            class_names = list(ckpt_class_names)
            num_classes = len(class_names)
        elif inferred_num:
            num_classes = int(inferred_num)
        else:
            # fallback
            class_names = FALLBACK_CLASS_NAMES
            num_classes = len(class_names)
        # build model and load weights (partial load if necessary)
        logger.info(f"Building ResNet18 with num_classes={num_classes}")
        model = build_model(num_classes)
        model_state = state_dict if state_dict is not None else raw_ckpt
        # filter state to matching shapes where possible to avoid fc mismatch crashes
        model_sd = model.state_dict()
        filtered_state = {}
        for k, v in model_state.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                filtered_state[k] = v
            else:
                # skip mismatched param (e.g., fc.weight shape mismatch)
                logger.debug(f"Skipping param (missing/mismatch): {k}")
        model.load_state_dict(filtered_state, strict=False)
        logger.info("Model weights loaded (partial=True where needed).")
        if class_names is None:
            # if checkpoint had no class names, but it contained label_to_idx we tried earlier.
            class_names = FALLBACK_CLASS_NAMES[:num_classes]  # trim to num_classes
    except Exception as ex:
        logger.exception("Error loading model; continuing with randomly initialized model.")
        # fallback: create random model with fallback num classes
        if 'num_classes' not in locals():
            num_classes = len(FALLBACK_CLASS_NAMES)
        model = build_model(num_classes)
        class_names = FALLBACK_CLASS_NAMES[:num_classes]

# move model to device and eval
if model is None:
    num_classes = len(FALLBACK_CLASS_NAMES)
    model = build_model(num_classes)
    class_names = FALLBACK_CLASS_NAMES[:num_classes]

model.to(device)
model.eval()
logger.info(f"Model ready. Class names: {class_names}")

# Preprocessing transform
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ---------------------------
# Inference helpers
# ---------------------------
def read_image(stream) -> Image.Image:
    try:
        img = Image.open(stream).convert("RGB")
        return img
    except Exception:
        raise

def predict_image(img: Image.Image, top_k: int = 3) -> List[Dict]:
    model_input = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(model_input)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    # get topk
    topk = min(top_k, len(probs))
    indices = probs.argsort()[::-1][:topk]
    results = []
    for idx in indices:
        results.append({"label": class_names[int(idx)], "index": int(idx), "prob": float(probs[int(idx)])})
    return results

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# simple HTML UI for quick test
HTML_TEMPLATE = """
<!doctype html>
<title>Disaster Classification</title>
<h1>Upload an image to classify</h1>
<form action="/predict" method="post" enctype="multipart/form-data">
  <input type=file name=file>
  <input type=submit value="Upload">
</form>
<p>Use POST /predict (form-data key 'file') for JSON response.</p>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(device), "model_loaded": bool(model is not None), "num_classes": len(class_names)})

@app.route("/predict", methods=["POST"])
def predict():
    # Accept single file upload
    if "file" not in request.files:
        return jsonify({"error": "No file part. Use form key 'file'."}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400
    try:
        img = read_image(file.stream)
    except Exception:
        return jsonify({"error": "Unable to read image."}), 400
    top_k = int(request.form.get("top_k", TOP_K_DEFAULT))
    try:
        results = predict_image(img, top_k=top_k)
    except Exception as e:
        logger.exception("Inference error")
        return jsonify({"error": "Inference failed", "details": str(e)}), 500

    # Save a small output record
    try:
        now = int(time.time())
        out_json = {"time": now, "results": results}
        out_path = os.path.join(OUTPUTS_DIR, f"pred_{now}.json")
        with open(out_path, "w") as f:
            json.dump(out_json, f)
    except Exception:
        logger.exception("Failed to save output")

    return jsonify({"predictions": results})

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    # Accept multiple files under 'files'
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded. Use form key 'files' multiple times."}), 400
    top_k = int(request.form.get("top_k", TOP_K_DEFAULT))
    responses = []
    for file in files:
        try:
            img = read_image(file.stream)
            res = predict_image(img, top_k=top_k)
            responses.append({"filename": file.filename, "predictions": res})
        except Exception as e:
            responses.append({"filename": file.filename, "error": str(e)})
    return jsonify({"results": responses})

@app.route("/classes", methods=["GET"])
def classes():
    return jsonify({"classes": class_names})

@app.route("/download/<path:fname>", methods=["GET"])
def download(fname):
    # download outputs saved in outputs folder
    safe = os.path.join(OUTPUTS_DIR, os.path.basename(fname))
    if os.path.exists(safe):
        return send_file(safe, as_attachment=True)
    return jsonify({"error": "file not found"}), 404

# ---------------------------
# CLI / Run
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0"
    logger.info(f"Starting Flask app on {host}:{port} ...")
    # Run with threaded server for simplicity; production use gunicorn or uvicorn if using ASGI
    #app.run(host=host, port=port, threaded=True)
    app.run()
