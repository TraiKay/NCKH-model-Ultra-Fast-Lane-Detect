"""
Streamlit Web UI for Ultra-Fast Lane Detection V2.
Provides an interactive interface for image/video upload and real-time inference.

Usage:
    streamlit run app_streamlit.py
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
from datetime import datetime
import time
from PIL import Image
import torch
import torchvision.transforms as transforms

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure output directories exist
Path("./output").mkdir(parents=True, exist_ok=True)

# --- Model Configurations ---
# Each model has different anchors and parameters based on dataset
import numpy as np

# Weights directory
WEIGHTS_DIR = PROJECT_ROOT / "weights" / "pth"

# Model configurations for each dataset/backbone combination
MODEL_CONFIGS = {
    "culane_res18": {
        "name": "CULane ResNet18",
        "backbone": "18",
        "dataset": "CULane",
        "weights": WEIGHTS_DIR / "culane_res18.pth",
        "num_lanes": 4,
        "train_width": 1600,
        "train_height": 320,
        "crop_ratio": 0.6,
        "num_cell_row": 200,
        "num_cell_col": 100,
        "num_row": 72,
        "num_col": 81,
        "fc_norm": True,
        "use_aux": False,
        "row_anchor": np.linspace(0.42, 1, 72).tolist(),
        "col_anchor": np.linspace(0, 1, 81).tolist(),
        "f1_score": 75.0,
        "description": "Nhanh, nhẹ. Tốt cho real-time",
        "pros": ["Nhanh nhất", "Nhẹ ~42MB", "Real-time 30+ FPS"],
        "cons": ["Accuracy thấp nhất 75%", "Khó với đường cong"],
    },
    "culane_res34": {
        "name": "CULane ResNet34",
        "backbone": "34",
        "dataset": "CULane",
        "weights": WEIGHTS_DIR / "culane_res34.pth",
        "num_lanes": 4,
        "train_width": 1600,
        "train_height": 320,
        "crop_ratio": 0.6,
        "num_cell_row": 200,
        "num_cell_col": 100,
        "num_row": 72,
        "num_col": 81,
        "fc_norm": True,
        "use_aux": False,
        "row_anchor": np.linspace(0.42, 1, 72).tolist(),
        "col_anchor": np.linspace(0, 1, 81).tolist(),
        "f1_score": 76.0,
        "description": "Chính xác hơn ResNet18",
        "pros": ["Chính xác hơn 1%", "Đa dạng scene", "Backbone mạnh"],
        "cons": ["Chậm hơn ~20%", "Nặng hơn ~83MB"],
    },
    "tusimple_res18": {
        "name": "TuSimple ResNet18",
        "backbone": "18",
        "dataset": "Tusimple",
        "weights": WEIGHTS_DIR / "tusimple_res18.pth",
        "num_lanes": 4,
        "train_width": 800,
        "train_height": 320,
        "crop_ratio": 0.8,
        "num_cell_row": 100,
        "num_cell_col": 100,
        "num_row": 56,
        "num_col": 41,
        "fc_norm": False,  # TuSimple uses fc_norm=False
        "use_aux": False,
        "row_anchor": np.linspace(160/720, 710/720, 56).tolist(),
        "col_anchor": np.linspace(0, 1, 41).tolist(),
        "f1_score": 96.11,
        "description": "Rất chính xác trên highway",
        "pros": ["Accuracy cao nhất 96%", "Tốt trên highway", "Nhanh"],
        "cons": ["Chỉ tốt trên highway", "Yếu với đường đô thị"],
    },
    "tusimple_res34": {
        "name": "TuSimple ResNet34",
        "backbone": "34",
        "dataset": "Tusimple",
        "weights": WEIGHTS_DIR / "tusimple_res34.pth",
        "num_lanes": 4,
        "train_width": 800,
        "train_height": 320,
        "crop_ratio": 0.8,
        "num_cell_row": 100,
        "num_cell_col": 100,
        "num_row": 56,
        "num_col": 41,
        "fc_norm": False,  # TuSimple uses fc_norm=False
        "use_aux": False,
        "row_anchor": np.linspace(160/720, 710/720, 56).tolist(),
        "col_anchor": np.linspace(0, 1, 41).tolist(),
        "f1_score": 96.24,
        "description": "Tốt nhất cho highway",
        "pros": ["Accuracy cao nhất TuSimple", "Highway tốt nhất", "Stable"],
        "cons": ["Chỉ highway", "Chậm hơn ResNet18"],
    },
    "curvelanes_res18": {
        "name": "CurveLanes ResNet18",
        "backbone": "18",
        "dataset": "CurveLanes",
        "weights": WEIGHTS_DIR / "curvelanes_res18.pth",
        "num_lanes": 10,
        "train_width": 1600,
        "train_height": 800,
        "crop_ratio": 0.8,
        "num_cell_row": 200,  # Verified: 200*72 + 2*72 = 14544 = cls_row weight shape
        "num_cell_col": 100,  # Verified: 100*41 + 2*41 = 4182 = cls_col weight shape
        "num_row": 72,
        "num_col": 41,  # Verified: ResNet18 checkpoint has cls_col [4182,2048] = (100+1)*41+2*41
        "fc_norm": False,
        "use_aux": False,
        "row_anchor": np.linspace(0.4, 1, 72).tolist(),
        "col_anchor": np.linspace(0, 1, 41).tolist(),
        "f1_score": 80.42,
        "description": "Tốt cho đường cong",
        "pros": ["Hỗ trợ đến 10 làn", "Tốt với đường cong phức tạp", "Nhanh"],
        "cons": ["Cần fine-tune cho đường Việt Nam", "Accuracy thấp hơn ResNet34"],
    },
    "curvelanes_res34": {
        "name": "CurveLanes ResNet34",
        "backbone": "34",
        "dataset": "CurveLanes",
        "weights": WEIGHTS_DIR / "curvelanes_res34.pth",
        "num_lanes": 10,
        "train_width": 1600,
        "train_height": 800,
        "crop_ratio": 0.8,
        "num_cell_row": 200,  # Verified: 200*72 + 2*72 = 14544 = cls_row weight shape
        "num_cell_col": 100,  # Verified: 100*81 + 2*81 = 8262 = cls_col weight shape
        "num_row": 72,
        "num_col": 81,  # Fixed: was 41, should be 81 (from official config)
        "fc_norm": False,
        "use_aux": False,
        "row_anchor": np.linspace(0.4, 1, 72).tolist(),
        "col_anchor": np.linspace(0, 1, 81).tolist(),  # Fixed: 81 points instead of 41
        "f1_score": 81.34,
        "description": "Chính xác nhất cho đường cong",
        "pros": ["Chính xác nhất CurveLanes", "Hỗ trợ 10 làn", "Tốt đường cong"],
        "cons": ["Chậm hơn ResNet18 ~20%", "Cần fine-tune cho đường VN"],
    },
}

# ONNX weights directory
ONNX_DIR = PROJECT_ROOT / "weights" / "onnx"

# Add ONNX weights paths to configs
for key in MODEL_CONFIGS:
    MODEL_CONFIGS[key]["onnx_weights"] = ONNX_DIR / f"{key}.onnx"

# Default config (for backward compatibility)
DEFAULT_MODEL = "culane_res18"
LANE_COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), 
               (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
               (0, 128, 255), (128, 255, 0)]  # 10 colors for CurveLanes

# Page configuration
st.set_page_config(
    page_title="Ultra-Fast Lane Detection V2",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark gradient theme
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Headers */
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(to right, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
        margin-bottom: 0.5rem;
        letter-spacing: 2px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #b8c6db;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 1px;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 700;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 114, 255, 0.4);
    }
    
    /* Stats Cards */
    .stats-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Lane colors legend */
    .lane-legend {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .lane-color {
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }
    .lane-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
    }
</style>
""", unsafe_allow_html=True)


# --- Model Loading ---
@st.cache_resource
def load_model(device: str, model_key: str = DEFAULT_MODEL):
    """Load and cache the lane detection model based on selected model key."""
    from addict import Dict
    
    # Get config for selected model
    if model_key not in MODEL_CONFIGS:
        model_key = DEFAULT_MODEL
    
    config = MODEL_CONFIGS[model_key]
    cfg = Dict(config)
    
    # Select model based on dataset - CurveLanes uses different architecture
    dataset = config.get("dataset", "CULane")
    if dataset == "CurveLanes":
        from model.model_curvelanes import get_model as get_lane_model
    else:
        # CULane and TuSimple use same architecture
        from model.model_culane import get_model as get_lane_model
    
    # Check if weights exist
    weights_path = config["weights"]
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    
    # Create model
    net = get_lane_model(cfg)
    
    # Load weights
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v
    
    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()
    
    # Move to device
    dev = torch.device(device)
    net = net.to(dev)
    
    return net, cfg, dev


@st.cache_resource
def load_model_onnx(model_key: str = DEFAULT_MODEL, use_gpu: bool = False):
    """Load ONNX model with ONNX Runtime for faster inference."""
    import onnxruntime as ort
    from addict import Dict
    
    # Get config for selected model
    if model_key not in MODEL_CONFIGS:
        model_key = DEFAULT_MODEL
    
    config = MODEL_CONFIGS[model_key]
    cfg = Dict(config)
    
    # Check if ONNX weights exist
    onnx_path = config["onnx_weights"]
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX weights not found: {onnx_path}\nRun: python deploy/batch_convert_onnx.py")
    
    # Configure session options for performance
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4
    
    # Select execution provider
    if use_gpu and 'CUDAExecutionProvider' in ort.get_available_providers():
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    
    # Create inference session
    session = ort.InferenceSession(str(onnx_path), sess_options, providers=providers)
    
    # Get actual provider being used
    actual_provider = session.get_providers()[0]
    
    return session, cfg, actual_provider


def check_cuda_compatibility():
    """Check if CUDA is available AND compatible with this GPU."""
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    try:
        # Try a simple CUDA operation to verify compatibility
        test_tensor = torch.zeros(1).cuda()
        _ = test_tensor + 1  # This will fail if kernel not available
        return True, torch.cuda.get_device_name(0)
    except RuntimeError as e:
        if "no kernel image" in str(e) or "not compatible" in str(e):
            return False, f"GPU architecture not supported by this PyTorch version"
        return False, str(e)


def get_transforms(cfg):
    """Get image transforms based on model config.
    
    Each dataset has different input requirements:
    - CULane: (1600, 320) → resize to (1600, 533) → crop bottom 320
    - TuSimple: (800, 320) → resize to (800, 400) → crop bottom 320
    - CurveLanes: (1600, 800) → resize to (1600, 1000) → crop bottom 800
    
    Formula: resize_height = train_height / crop_ratio
    """
    # Calculate resize dimensions based on config
    resize_width = cfg.train_width
    resize_height = int(cfg.train_height / cfg.crop_ratio)
    crop_height = cfg.train_height
    
    return transforms.Compose([
        # Step 1: Resize to (resize_height, resize_width)
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # Step 2: Crop bottom crop_height rows (matching dataset.py: img[:,-crop_size:,:])
        transforms.Lambda(lambda x: x[:, -crop_height:, :]),
    ])


def pred2coords(pred, row_anchor, col_anchor, local_width=1, original_image_width=1640, original_image_height=590, dataset="CULane", num_lanes=4):
    """Convert model prediction to lane coordinates - matching official demo.py.
    
    Note: This exactly mirrors the logic in demo.py for compatibility.
    Row lanes use row_anchor for Y position, col lanes use col_anchor for X position.
    
    Returns:
        coords: List of lane coordinates
        confidences: List of confidence scores for each lane
        metrics: Dict with detection metrics
    """
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape
    
    max_indices_row = pred['loc_row'].argmax(1).cpu()
    valid_row = pred['exist_row'].argmax(1).cpu()
    max_indices_col = pred['loc_col'].argmax(1).cpu()
    valid_col = pred['exist_col'].argmax(1).cpu()
    
    # Get existence probabilities for confidence calculation
    exist_row_prob = torch.softmax(pred['exist_row'], dim=1)[:, 1, :, :].cpu()
    exist_col_prob = torch.softmax(pred['exist_col'], dim=1)[:, 1, :, :].cpu()
    
    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()
    
    coords = []
    confidences = []
    
    # Lane indices - dataset-specific assignment
    # TuSimple: highway straight lanes only, skip col lanes (garbage output)
    # CULane: mixed roads, use 2 row + 2 col lanes
    # CurveLanes: complex curved roads, use all lanes
    if dataset == "Tusimple":
        # TuSimple model trained on straight highway lanes only
        # Col lanes output is unreliable, skip them
        row_lane_idx = [1, 2]
        col_lane_idx = []  # Skip col lanes for TuSimple
    elif dataset == "CurveLanes":
        row_lane_idx = list(range(num_lane_row))
        col_lane_idx = list(range(num_lane_col))
    else:
        # CULane: use 2row2col mode
        row_lane_idx = [1, 2]
        col_lane_idx = [0, 3]
    
    # Track metrics
    total_points = 0
    valid_points = 0
    
    # Process row lanes - exactly like demo.py
    for i in row_lane_idx:
        if i >= num_lane_row:
            continue
        tmp = []
        lane_conf = 0.0
        point_count = 0
        if valid_row[0, :, i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if k >= len(row_anchor):  # Bounds check
                    continue
                if valid_row[0, k, i]:
                    all_ind = torch.tensor(list(range(max(0, max_indices_row[0, k, i] - local_width), 
                                                       min(num_grid_row - 1, max_indices_row[0, k, i] + local_width) + 1)))
                    out_tmp = (pred['loc_row'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    # Exactly like demo.py: out_tmp / (num_grid_row-1) * original_image_width
                    out_tmp = out_tmp / (num_grid_row - 1) * original_image_width
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
                    lane_conf += float(exist_row_prob[0, k, i])
                    point_count += 1
            if point_count > 0:
                lane_conf = lane_conf / point_count
            coords.append(tmp)
            confidences.append(lane_conf)
            valid_points += len(tmp)
        total_points += num_cls_row
    
    # Process col lanes - exactly like demo.py
    for i in col_lane_idx:
        if i >= num_lane_col:
            continue
        tmp = []
        lane_conf = 0.0
        point_count = 0
        if valid_col[0, :, i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if k >= len(col_anchor):  # Bounds check
                    continue
                if valid_col[0, k, i]:
                    all_ind = torch.tensor(list(range(max(0, max_indices_col[0, k, i] - local_width), 
                                                       min(num_grid_col - 1, max_indices_col[0, k, i] + local_width) + 1)))
                    out_tmp = (pred['loc_col'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    # Exactly like demo.py: out_tmp / (num_grid_col-1) * original_image_height
                    out_tmp = out_tmp / (num_grid_col - 1) * original_image_height
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
                    lane_conf += float(exist_col_prob[0, k, i])
                    point_count += 1
            if point_count > 0:
                lane_conf = lane_conf / point_count
            coords.append(tmp)
            confidences.append(lane_conf)
            valid_points += len(tmp)
        total_points += num_cls_col
    
    # Calculate metrics
    metrics = {
        'num_lanes': len(coords),
        'avg_confidence': sum(confidences) / len(confidences) if confidences else 0.0,
        'point_coverage': valid_points / total_points if total_points > 0 else 0.0,
        'total_points': valid_points,
    }
    
    return coords, confidences, metrics


def draw_lanes(image, coords):
    """Draw detected lanes on image."""
    vis = image.copy()
    for lane_idx, lane in enumerate(coords):
        color = LANE_COLORS[lane_idx % len(LANE_COLORS)]
        for coord in lane:
            cv2.circle(vis, coord, 5, color, -1)
        if len(lane) > 1:
            for i in range(len(lane) - 1):
                cv2.line(vis, lane[i], lane[i+1], color, 3)
    return vis


def process_single_frame(frame, net, img_transform, cfg, device):
    """Process a single frame and return the annotated frame with metrics.
    
    Returns:
        annotated: Frame with lanes drawn
        metrics: Dict with detection metrics (num_lanes, avg_confidence, point_coverage, total_points)
    """
    h, w = frame.shape[:2]
    
    # Dataset-specific reference dimensions (matching demo.py)
    # These are the dimensions the model was trained on and anchors are calculated for
    if cfg.dataset == "Tusimple":
        ref_w, ref_h = 1280, 720
    elif cfg.dataset == "CurveLanes":
        ref_w, ref_h = 2560, 1440
    else:  # CULane
        ref_w, ref_h = 1640, 590
    
    # Convert BGR to RGB for PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Transform and run inference
    input_tensor = img_transform(pil_image).unsqueeze(0).to(device)
    
    # Measure inference time
    start_inference = time.time()
    with torch.no_grad():
        pred = net(input_tensor)
    inference_time = time.time() - start_inference
    
    # Get coordinates in reference dimensions (matching demo.py behavior)
    coords, confidences, metrics = pred2coords(
        pred, cfg.row_anchor, cfg.col_anchor, 
        original_image_width=ref_w, original_image_height=ref_h, 
        dataset=cfg.dataset, num_lanes=cfg.num_lanes
    )
    
    # Scale coordinates from reference dimensions to actual video dimensions
    scale_x = w / ref_w
    scale_y = h / ref_h
    scaled_coords = []
    for lane in coords:
        scaled_lane = [(int(x * scale_x), int(y * scale_y)) for x, y in lane]
        scaled_coords.append(scaled_lane)
    
    annotated = draw_lanes(frame, scaled_coords)
    
    # Add inference time to metrics
    metrics['inference_time_ms'] = inference_time * 1000
    metrics['confidences'] = confidences
    
    return annotated, metrics


def process_single_frame_onnx(frame, session, img_transform, cfg):
    """Process a single frame using ONNX Runtime for faster inference.
    
    Returns:
        annotated: Frame with lanes drawn
        metrics: Dict with detection metrics
    """
    h, w = frame.shape[:2]
    
    # Dataset-specific reference dimensions (matching demo.py)
    if cfg.dataset == "Tusimple":
        ref_w, ref_h = 1280, 720
    elif cfg.dataset == "CurveLanes":
        ref_w, ref_h = 2560, 1440
    else:  # CULane
        ref_w, ref_h = 1640, 590
    
    # Convert BGR to RGB for PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Transform (returns torch tensor, convert to numpy)
    input_tensor = img_transform(pil_image).unsqueeze(0).numpy()
    
    # Get input name dynamically
    input_name = session.get_inputs()[0].name
    
    # Measure inference time
    start_inference = time.time()
    
    # ONNX Runtime inference
    outputs = session.run(None, {input_name: input_tensor})
    
    inference_time = time.time() - start_inference
    
    # Get output names dynamically and map correctly
    output_names = [o.name for o in session.get_outputs()]
    output_dict = {name: outputs[i] for i, name in enumerate(output_names)}
    
    # Map outputs to expected keys (ONNX export names may vary)
    def find_output(patterns):
        for pattern in patterns:
            for name in output_names:
                if pattern in name.lower():
                    return output_dict[name]
        return None
    
    # Try to match by name patterns, fallback to index order
    loc_row = find_output(['loc_row', 'row_loc', '201'])
    loc_col = find_output(['loc_col', 'col_loc', '202'])  
    exist_row = find_output(['exist_row', 'row_exist', '203'])
    exist_col = find_output(['exist_col', 'col_exist', '204'])
    
    # Fallback: use index order if patterns don't match
    if loc_row is None or loc_col is None or exist_row is None or exist_col is None:
        loc_row = outputs[0]
        loc_col = outputs[1]
        exist_row = outputs[2]
        exist_col = outputs[3]
    
    # Convert outputs to torch tensors for pred2coords compatibility
    pred = {
        'loc_row': torch.from_numpy(loc_row),
        'loc_col': torch.from_numpy(loc_col),
        'exist_row': torch.from_numpy(exist_row),
        'exist_col': torch.from_numpy(exist_col),
    }
    
    # Get coordinates in reference dimensions (matching demo.py behavior)
    coords, confidences, metrics = pred2coords(
        pred, cfg.row_anchor, cfg.col_anchor, 
        original_image_width=ref_w, original_image_height=ref_h, 
        dataset=cfg.dataset, num_lanes=cfg.num_lanes
    )
    
    # Scale coordinates from reference dimensions to actual video dimensions
    scale_x = w / ref_w
    scale_y = h / ref_h
    scaled_coords = []
    for lane in coords:
        scaled_lane = [(int(x * scale_x), int(y * scale_y)) for x, y in lane]
        scaled_coords.append(scaled_lane)
    
    annotated = draw_lanes(frame, scaled_coords)
    
    # Add inference time to metrics
    metrics['inference_time_ms'] = inference_time * 1000
    metrics['confidences'] = confidences
    
    return annotated, metrics


def init_session_state():
    """Initialize session state variables."""
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'results' not in st.session_state:
        st.session_state.results = None


def main():
    """Main Streamlit app."""
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🚗 Ultra-Fast Lane Detection V2</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Phát hiện làn đường thời gian thực với ResNet18 | GPU Accelerated</p>', unsafe_allow_html=True)
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        
        # Device selection with auto-detection
        with st.expander("🖥️ System Settings", expanded=True):
            # Check CUDA compatibility (not just availability)
            cuda_compatible, gpu_info = check_cuda_compatibility()
            
            # Always show both options, but prioritize CPU if GPU not compatible
            if cuda_compatible:
                device_options = ["cuda", "cpu"]
                default_device_idx = 0  # GPU first if compatible
                st.success(f"✅ GPU: {gpu_info}")
            elif torch.cuda.is_available():
                # CUDA available but not compatible (e.g., RTX 5060 sm_120)
                device_options = ["cpu", "cuda"]  # CPU first
                default_device_idx = 0  # Default to CPU
                st.warning(f"⚠️ GPU: {torch.cuda.get_device_name(0)} (chưa tương thích)")
                st.info("💡 GPU sm_120 quá mới → Ưu tiên **CPU** để tránh lỗi")
            else:
                device_options = ["cpu"]
                default_device_idx = 0
                st.warning("⚠️ CUDA không khả dụng. Sử dụng CPU.")
            
            device = st.selectbox(
                "Device",
                options=device_options,
                index=default_device_idx,
                help=f"GPU Compatible: {'✅ Có' if cuda_compatible else '❌ Không - Ưu tiên CPU'}"
            )
            
            # Preview settings
            preview_mode = st.selectbox(
                "Preview Mode",
                options=["Real-time (Every frame)", "Fast (Every 2 frames)", "Normal (Every 5 frames)", "Disabled"],
                index=1,
                help="Tốc độ cập nhật preview"
            )
            
            preview_interval_map = {
                "Real-time (Every frame)": 1,
                "Fast (Every 2 frames)": 2,
                "Normal (Every 5 frames)": 5,
                "Disabled": 0
            }
            preview_interval = preview_interval_map[preview_mode]
            
            st.divider()
            
            # Backend selection (PyTorch vs ONNX)
            backend = st.radio(
                "🚀 Inference Backend",
                options=["ONNX Runtime (Nhanh)", "PyTorch (Đầy đủ)"],
                index=0,
                help="ONNX Runtime nhanh hơn ~2-3x so với PyTorch"
            )
            use_onnx = "ONNX" in backend
        
        # Model Selection
        with st.expander("🤖 Chọn Model", expanded=True):
            # Create model options with F1 scores
            model_options = {
                key: f"{cfg['name']} (F1: {cfg['f1_score']}%)"
                for key, cfg in MODEL_CONFIGS.items()
            }
            
            selected_model = st.selectbox(
                "Model",
                options=list(model_options.keys()),
                format_func=lambda x: model_options[x],
                index=0,
                help="Chọn model phù hợp với loại đường"
            )
            
            # Show selected model info
            selected_cfg = MODEL_CONFIGS[selected_model]
            st.markdown(f"""
            **📊 Thông tin Model:**
            - **Dataset:** {selected_cfg['dataset']}
            - **Backbone:** ResNet{selected_cfg['backbone']}
            - **F1 Score:** {selected_cfg['f1_score']}%
            - **Input:** {selected_cfg['train_width']}×{selected_cfg['train_height']}
            - **Lanes:** {selected_cfg['num_lanes']}
            
            💡 _{selected_cfg['description']}_
            """)
            
            # Check if weights exist (PTH for PyTorch, ONNX for ONNX Runtime)
            if use_onnx:
                weights_exist = selected_cfg['onnx_weights'].exists()
                weights_type = "ONNX"
                weights_name = selected_cfg['onnx_weights'].name
            else:
                weights_exist = selected_cfg['weights'].exists()
                weights_type = "PTH"
                weights_name = selected_cfg['weights'].name
            
            if not weights_exist:
                st.error(f"❌ {weights_type} weights không tồn tại: {weights_name}")
                if use_onnx:
                    st.info("💡 Chạy: `python deploy/batch_convert_onnx.py` để tạo ONNX files")
        
        st.divider()
        
        # Load model button
        if st.button("🔄 Load/Reload Model", type="primary", use_container_width=True):
            if not weights_exist:
                st.error("❌ Vui lòng kiểm tra file weights!")
            else:
                backend_name = "ONNX Runtime" if use_onnx else "PyTorch"
                with st.spinner(f"Đang load {selected_cfg['name']} ({backend_name})..."):
                    try:
                        if use_onnx:
                            # Clear ONNX cache and reload
                            load_model_onnx.clear()
                            session, cfg, provider = load_model_onnx(selected_model, device == "cuda")
                            st.session_state.model_loaded = True
                            st.session_state.use_onnx = True
                            st.session_state.onnx_session = session
                            st.session_state.cfg = cfg
                            st.session_state.current_model = selected_model
                            st.success(f"✅ {selected_cfg['name']} (ONNX - {provider}) đã sẵn sàng!")
                        else:
                            # Clear PyTorch cache and reload
                            load_model.clear()
                            net, cfg, dev = load_model(device, selected_model)
                            st.session_state.model_loaded = True
                            st.session_state.use_onnx = False
                            st.session_state.net = net
                            st.session_state.cfg = cfg
                            st.session_state.device = device
                            st.session_state.current_model = selected_model
                            st.success(f"✅ {selected_cfg['name']} (PyTorch - {device.upper()}) đã sẵn sàng!")
                    except Exception as e:
                        st.error(f"❌ Lỗi load model: {e}")
                        st.session_state.model_loaded = False
        
        # Status indicator - Clear display of backend and model
        st.divider()
        if st.session_state.model_loaded:
            use_onnx = st.session_state.get('use_onnx', False)
            model_name = st.session_state.get('current_model', 'Unknown')
            
            if use_onnx:
                backend_icon = "⚡"
                backend_name = "ONNX Runtime"
                file_ext = ".onnx"
                badge_color = "#00D4AA"  # Teal
            else:
                backend_icon = "🔥"
                backend_name = f"PyTorch ({st.session_state.get('device', 'cpu').upper()})"
                file_ext = ".pth"
                badge_color = "#FF6B6B"  # Red
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {badge_color}22, {badge_color}44); 
                        border: 2px solid {badge_color}; 
                        border-radius: 10px; 
                        padding: 15px; 
                        margin: 10px 0;">
                <div style="font-size: 1.1em; font-weight: bold; color: {badge_color};">
                    {backend_icon} {backend_name}
                </div>
                <div style="font-size: 0.9em; margin-top: 5px;">
                    📁 <code style="background: {badge_color}44; padding: 2px 6px; border-radius: 4px;">{model_name}{file_ext}</code>
                </div>
                <div style="font-size: 0.85em; color: #888; margin-top: 5px;">
                    ✅ Sẵn sàng xử lý
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("🟡 Chưa load model - Chọn backend và bấm **Load/Reload Model**")
    
    # Main content area with tabs
    tab_image, tab_video, tab_train, tab_onnx_test = st.tabs(["🖼️ Image Processing", "🎬 Video Processing", "🎯 Training", "🧪 ONNX Test"])
    
    # ========== TAB 1: IMAGE PROCESSING ==========
    with tab_image:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📤 Upload Image")
            uploaded_image = st.file_uploader(
                "Upload an image",
                type=['jpg', 'jpeg', 'png'],
                help="Hỗ trợ JPG, PNG"
            )
            
            if uploaded_image is not None:
                # Display original
                image = Image.open(uploaded_image)
                st.image(image, caption="Original Image", use_container_width=True)
        
        with col2:
            st.subheader("🎯 Detection Result")
            
            if uploaded_image is not None and st.session_state.model_loaded:
                if st.button("🔍 Detect Lanes", key="detect_image", use_container_width=True):
                    with st.spinner("Detecting lanes..."):
                        # Get config and transform from session state
                        cfg = st.session_state.cfg
                        img_transform = get_transforms(cfg)
                        use_onnx = st.session_state.get('use_onnx', False)
                        
                        # Convert to CV2 format
                        image_np = np.array(image)
                        if len(image_np.shape) == 2:
                            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
                        elif image_np.shape[2] == 4:
                            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
                        else:
                            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                        
                        # Process with appropriate backend
                        if use_onnx:
                            session = st.session_state.onnx_session
                            result, metrics = process_single_frame_onnx(image_np, session, img_transform, cfg)
                            backend_info = "ONNX"
                        else:
                            net = st.session_state.net
                            dev = torch.device(st.session_state.device)
                            result, metrics = process_single_frame(image_np, net, img_transform, cfg, dev)
                            backend_info = f"PyTorch ({st.session_state.device.upper()})"
                        
                        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                        num_lanes = metrics.get('num_lanes', 0)
                        
                        st.image(result_rgb, caption=f"Detected {num_lanes} lanes ({backend_info})", use_container_width=True)
                        
                        # Show metrics
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("🛣️ Lanes", num_lanes)
                        with col_m2:
                            st.metric("⚡ Inference", f"{metrics.get('inference_time_ms', 0):.1f} ms")
                        with col_m3:
                            st.metric("📊 Confidence", f"{metrics.get('avg_confidence', 0)*100:.1f}%")
                        
                        st.success(f"✅ Detected **{num_lanes}** lanes using **{backend_info}**!")
            elif not st.session_state.model_loaded:
                st.info("👈 Vui lòng load model trước (sidebar)")
    
    # ========== TAB 2: VIDEO PROCESSING ==========
    with tab_video:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📹 Input Video")
            
            # Video upload
            uploaded_video = st.file_uploader(
                "Upload video file",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Hỗ trợ MP4, AVI, MOV, MKV"
            )
            
            # Or select from folder
            st.markdown("**Hoặc chọn từ thư mục:**")
            video_folder = PROJECT_ROOT
            video_files = list(video_folder.glob("*.mp4")) + list(video_folder.glob("*.avi"))
            video_files = [f for f in video_files if f.is_file() and not f.name.startswith('.')]
            
            if video_files:
                video_options = {f.name: f for f in video_files}
                selected_video = st.selectbox(
                    "Chọn video có sẵn",
                    options=["-- Chọn --"] + list(video_options.keys())
                )
            else:
                st.info("📁 Không có video trong thư mục project")
                selected_video = "-- Chọn --"
                video_options = {}
            
            # Preview video
            if uploaded_video is not None:
                st.video(uploaded_video)
                video_source = "upload"
            elif selected_video != "-- Chọn --" and selected_video in video_options:
                video_path = video_options[selected_video]
                st.video(str(video_path))
                video_source = "file"
            else:
                video_source = None
        
        with col2:
            st.subheader("🎬 Output Preview")
            
            # Placeholder for processed video
            frame_placeholder = st.empty()
            
            if st.session_state.results:
                results = st.session_state.results
                
                # Display results
                st.success(f"✅ Xử lý hoàn tất trong {results['processing_time']:.1f}s")
                
                # Basic Stats Row
                c1, c2, c3, c4 = st.columns(4)
                fps = results['total_frames'] / results['processing_time'] if results['processing_time'] > 0 else 0
                with c1:
                    st.metric("📊 Total Frames", results['total_frames'])
                with c2:
                    st.metric("🛣️ Avg Lanes", f"{results['avg_lanes']:.1f}")
                with c3:
                    st.metric("⚡ Avg FPS", f"{fps:.1f}")
                with c4:
                    st.metric("🎯 Detection Rate", f"{results.get('detection_rate', 0):.1%}")
                
                # Detailed Metrics Row
                st.markdown("#### 📈 Chỉ Số Đánh Giá Chi Tiết")
                m1, m2, m3 = st.columns(3)
                with m1:
                    conf = results.get('avg_confidence', 0)
                    conf_color = "🟢" if conf > 0.7 else "🟡" if conf > 0.5 else "🔴"
                    st.metric(f"{conf_color} Avg Confidence", f"{conf:.1%}")
                with m2:
                    inf_ms = results.get('avg_inference_ms', 0)
                    inf_color = "🟢" if inf_ms < 20 else "🟡" if inf_ms < 50 else "🔴"
                    st.metric(f"{inf_color} Inference Time", f"{inf_ms:.1f}ms")
                with m3:
                    st.metric("📍 Total Points", results.get('total_points', 0))
                
                # Model Assessment with Explanations
                with st.expander("🔍 Phân Tích Model & Giải Thích Chỉ Số", expanded=True):
                    st.markdown("""
                    #### 📖 Giải Thích Các Chỉ Số
                    
                    | Chỉ Số | Ý Nghĩa | Tốt | Trung Bình | Kém |
                    |--------|---------|-----|------------|-----|
                    | **Avg Confidence** | Độ tin cậy trung bình của model khi phát hiện làn | >70% | 50-70% | <50% |
                    | **Detection Rate** | Tỷ lệ frame phát hiện được ít nhất 1 làn | >90% | 70-90% | <70% |
                    | **Avg FPS** | Tốc độ xử lý (khung hình/giây) | >30 | 15-30 | <15 |
                    | **Inference Time** | Thời gian model xử lý 1 frame | <20ms | 20-50ms | >50ms |
                    | **Avg Lanes** | Số làn trung bình phát hiện/frame | 2-4 | 1-2 | 0-1 |
                    """)
                    
                    st.markdown("---")
                    
                    # Automatic Assessment
                    st.markdown("#### ⚖️ Đánh Giá Tự Động")
                    
                    pros = []
                    cons = []
                    
                    # Assess confidence
                    if conf > 0.7:
                        pros.append("✅ **Độ tin cậy cao** - Model nhận diện làn chính xác")
                    elif conf < 0.5:
                        cons.append("⚠️ **Độ tin cậy thấp** - Cần cải thiện chất lượng video hoặc fine-tune model")
                    
                    # Assess detection rate
                    det_rate = results.get('detection_rate', 0)
                    if det_rate > 0.9:
                        pros.append("✅ **Tỷ lệ phát hiện cao** - Gần như mọi frame đều có làn đường")
                    elif det_rate < 0.7:
                        cons.append("⚠️ **Bỏ sót nhiều frame** - Video có thể có điều kiện khó (ánh sáng, thời tiết)")
                    
                    # Assess FPS
                    if fps > 30:
                        pros.append("✅ **Xử lý real-time** - Đủ nhanh cho ứng dụng thực tế")
                    elif fps < 15:
                        cons.append("⚠️ **Tốc độ chậm** - Cần GPU mạnh hơn hoặc giảm resolution")
                    
                    # Assess inference time
                    if inf_ms < 20:
                        pros.append("✅ **Inference nhanh** - Model tối ưu tốt")
                    elif inf_ms > 50:
                        cons.append("⚠️ **Inference chậm** - Xem xét dùng TensorRT hoặc ONNX")
                    
                    # Assess lane count
                    avg_lanes = results.get('avg_lanes', 0)
                    if 2 <= avg_lanes <= 4:
                        pros.append("✅ **Số làn hợp lý** - Phù hợp đường 2-4 làn xe")
                    elif avg_lanes < 1:
                        cons.append("⚠️ **Ít làn đường** - Kiểm tra lại video hoặc config model")
                    
                    # Display assessment
                    if pros:
                        st.markdown("**🌟 Ưu điểm:**")
                        for p in pros:
                            st.markdown(f"- {p}")
                    
                    if cons:
                        st.markdown("**⚡ Cần cải thiện:**")
                        for c in cons:
                            st.markdown(f"- {c}")
                    
                    if not pros and not cons:
                        st.info("📊 Kết quả ở mức trung bình")
                    
                    # Model comparison info
                    st.markdown("---")
                    st.markdown("""
                    #### 🔄 So Sánh Với Các Model Khác
                    
                    | Model | Dataset | F1 Score | Ưu điểm | Nhược điểm |
                    |-------|---------|----------|---------|------------|
                    | **ResNet18** (đang dùng) | CULane | 75.0% | Nhanh, nhẹ | Accuracy thấp hơn |
                    | ResNet34 | CULane | 76.0% | Chính xác hơn | Chậm hơn ~20% |
                    | ResNet18 | TuSimple | 96.1% | Rất chính xác | Chỉ tốt trên highway |
                    | ResNet18 | CurveLanes | 80.4% | Tốt cho đường cong | Cần fine-tune |
                    
                    💡 **Khuyến nghị:** Nếu cần accuracy cao hơn, hãy thử ResNet34 weights.
                    """)
                
                # Download button
                if os.path.exists(results['output_path']):
                    with open(results['output_path'], 'rb') as f:
                        st.download_button(
                            label="📥 Download Video đã xử lý",
                            data=f.read(),
                            file_name=f"lane_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
        
        # Process button
        st.divider()
        
        process_col1, process_col2, process_col3 = st.columns([1, 2, 1])
        
        with process_col2:
            if st.button("🚀 BẮT ĐẦU XỬ LÝ", type="primary", use_container_width=True, disabled=not st.session_state.model_loaded):
                if not st.session_state.model_loaded:
                    st.error("❌ Vui lòng load model trước!")
                elif video_source is None:
                    st.error("❌ Vui lòng chọn hoặc upload video!")
                else:
                    st.session_state.processing = True
                    
                    # Prepare video path
                    if video_source == "upload":
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                            tmp.write(uploaded_video.read())
                            input_path = tmp.name
                        clean_name = Path(uploaded_video.name).stem
                    else:
                        input_path = str(video_options[selected_video])
                        clean_name = Path(input_path).stem
                    
                    # Output path
                    output_path = str(PROJECT_ROOT / "output" / f"{clean_name}_lanes.mp4")
                    
                    # Load model from session state
                    use_onnx = st.session_state.get('use_onnx', False)
                    cfg = st.session_state.cfg
                    img_transform = get_transforms(cfg)
                    
                    if use_onnx:
                        session = st.session_state.onnx_session
                    else:
                        net = st.session_state.net
                        dev = torch.device(st.session_state.device)
                    
                    # Open video
                    cap = cv2.VideoCapture(input_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    # Progress elements
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Use imageio for H264 output
                    import imageio
                    writer = imageio.get_writer(
                        output_path, 
                        fps=fps,
                        codec='libx264',
                        pixelformat='yuv420p',
                        output_params=['-preset', 'fast', '-crf', '23']
                    )
                    
                    # Process - Track metrics
                    frame_count = 0
                    total_lanes = 0
                    total_confidence = 0.0
                    total_points = 0
                    total_inference_time = 0.0
                    frames_with_lanes = 0
                    lane_counts = []  # For distribution analysis
                    start_time = time.time()
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_count += 1
                        
                        # Process frame - use ONNX or PyTorch based on backend
                        if use_onnx:
                            annotated, frame_metrics = process_single_frame_onnx(frame, session, img_transform, cfg)
                        else:
                            annotated, frame_metrics = process_single_frame(frame, net, img_transform, cfg, dev)
                        
                        # Accumulate metrics
                        num_lanes = frame_metrics['num_lanes']
                        total_lanes += num_lanes
                        total_confidence += frame_metrics['avg_confidence']
                        total_points += frame_metrics['total_points']
                        total_inference_time += frame_metrics['inference_time_ms']
                        lane_counts.append(num_lanes)
                        if num_lanes > 0:
                            frames_with_lanes += 1
                        
                        # Write frame (BGR to RGB for imageio)
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        writer.append_data(annotated_rgb)
                        
                        # Update progress
                        progress = frame_count / total_frames
                        progress_bar.progress(progress)
                        
                        elapsed = time.time() - start_time
                        fps_actual = frame_count / elapsed if elapsed > 0 else 0
                        eta = (total_frames - frame_count) / fps_actual if fps_actual > 0 else 0
                        
                        status_text.markdown(f"""
                        **Frame:** {frame_count}/{total_frames} | 
                        **FPS:** {fps_actual:.1f} | 
                        **ETA:** {eta:.0f}s | 
                        **Lanes:** {num_lanes} |
                        **Conf:** {frame_metrics['avg_confidence']:.1%}
                        """)
                        
                        # Display preview
                        if preview_interval > 0 and frame_count % preview_interval == 0:
                            display_frame = cv2.resize(annotated_rgb, (640, 360))
                            frame_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                    
                    cap.release()
                    writer.close()
                    
                    # Store enhanced results
                    st.session_state.results = {
                        'total_frames': frame_count,
                        'avg_lanes': total_lanes / frame_count if frame_count > 0 else 0,
                        'processing_time': time.time() - start_time,
                        'output_path': output_path,
                        # New metrics
                        'avg_confidence': total_confidence / frame_count if frame_count > 0 else 0,
                        'avg_inference_ms': total_inference_time / frame_count if frame_count > 0 else 0,
                        'detection_rate': frames_with_lanes / frame_count if frame_count > 0 else 0,
                        'total_points': total_points,
                        'lane_distribution': lane_counts,
                    }
                    
                    st.session_state.processing = False
                    st.balloons()
                    st.rerun()
    
    # Footer
    # ========== TAB 3: TRAINING ==========
    with tab_train:
        st.subheader("🎯 Training Model")
        st.info("💡 Huấn luyện model mới với dataset của bạn hoặc fine-tune từ pretrained weights.")
        
        col_config, col_status = st.columns([1, 1])
        
        with col_config:
            st.markdown("### ⚙️ Training Configuration")
            
            # Dataset selection
            train_dataset = st.selectbox(
                "📁 Dataset",
                options=["CULane", "TuSimple", "CurveLanes"],
                index=0,
                help="Chọn dataset để train"
            )
            
            # Data path
            data_root = st.text_input(
                "📂 Data Root Path",
                value="",
                placeholder="D:/datasets/culane/",
                help="Đường dẫn tuyệt đối đến thư mục dataset"
            )
            
            st.divider()
            
            # Backbone selection
            train_backbone = st.selectbox(
                "🧠 Backbone",
                options=["18", "34"],
                format_func=lambda x: f"ResNet{x}",
                index=0,
                help="ResNet18 nhanh hơn, ResNet34 chính xác hơn"
            )
            
            # Hyperparameters
            st.markdown("### 📊 Hyperparameters")
            
            col_hyper1, col_hyper2 = st.columns(2)
            with col_hyper1:
                train_epochs = st.number_input(
                    "Epochs",
                    min_value=1,
                    max_value=200,
                    value=50,
                    help="Số epoch training"
                )
                
                train_batch_size = st.selectbox(
                    "Batch Size",
                    options=[4, 8, 16, 32],
                    index=2,
                    help="Batch size (tùy thuộc GPU VRAM)"
                )
            
            with col_hyper2:
                train_lr = st.number_input(
                    "Learning Rate",
                    min_value=0.0001,
                    max_value=0.1,
                    value=0.025,
                    step=0.001,
                    format="%.4f",
                    help="Learning rate ban đầu"
                )
                
                train_optimizer = st.selectbox(
                    "Optimizer",
                    options=["SGD", "Adam", "AdamW"],
                    index=0,
                    help="Optimizer cho training"
                )
            
            st.divider()
            
            # Advanced options
            with st.expander("🔧 Advanced Options"):
                use_aux = st.checkbox("Use Auxiliary Loss", value=False, help="Thêm auxiliary loss để cải thiện accuracy")
                use_pretrained = st.checkbox("Use Pretrained Backbone", value=True, help="Sử dụng backbone pretrained từ ImageNet")
                num_workers = st.slider("Num Workers", 0, 8, 4, help="Số workers cho dataloader")
                
                # Finetune from existing weights
                finetune_weights = st.selectbox(
                    "🎯 Finetune From",
                    options=["None (Train from scratch)"] + [f.name for f in WEIGHTS_DIR.glob("*.pth")],
                    index=0,
                    help="Fine-tune từ weights có sẵn"
                )
        
        with col_status:
            st.markdown("### 📋 Training Command")
            
            # Generate training command
            config_name = f"{train_dataset.lower()}_res{train_backbone}"
            
            # Build command
            train_cmd_parts = [
                "uv run python train.py",
                f"--config_path configs/{config_name}.py",
            ]
            
            if data_root:
                train_cmd_parts.append(f"--data_root \"{data_root}\"")
            
            train_cmd_parts.extend([
                f"--epoch {train_epochs}",
                f"--batch_size {train_batch_size}",
                f"--learning_rate {train_lr}",
                f"--optimizer {train_optimizer}",
            ])
            
            if finetune_weights != "None (Train from scratch)" and finetune_weights != "None":
                train_cmd_parts.append(f"--finetune weights/pth/{finetune_weights}")
            
            train_cmd = " \\\n    ".join(train_cmd_parts)
            
            st.code(train_cmd, language="bash")
            
            # Copy button
            st.download_button(
                "📋 Download Command Script",
                data=f"#!/bin/bash\n{train_cmd}",
                file_name="train_command.sh",
                mime="text/plain"
            )
            
            st.divider()
            
            st.markdown("### 🚀 Start Training")
            
            # Validation
            can_train = True
            issues = []
            
            if not data_root:
                issues.append("❌ Chưa nhập Data Root Path")
                can_train = False
            elif not Path(data_root).exists():
                issues.append(f"⚠️ Đường dẫn không tồn tại: {data_root}")
                can_train = False
            
            config_path = PROJECT_ROOT / "configs" / f"{config_name}.py"
            if not config_path.exists():
                issues.append(f"❌ Config không tồn tại: {config_name}.py")
                can_train = False
            
            train_script = PROJECT_ROOT / "train.py"
            if not train_script.exists():
                issues.append("❌ train.py không tồn tại")
                can_train = False
            
            # Show issues
            if issues:
                for issue in issues:
                    st.warning(issue)
            else:
                st.success("✅ Cấu hình hợp lệ, sẵn sàng training!")
            
            # Initialize training state
            if 'training_active' not in st.session_state:
                st.session_state.training_active = False
            
            # Start/Stop buttons
            col_start, col_stop = st.columns(2)
            
            with col_start:
                if st.button("🚀 Start Training", type="primary", use_container_width=True, disabled=not can_train or st.session_state.training_active):
                    st.session_state.training_active = True
                    st.info("🔄 Training sẽ chạy trong terminal riêng...")
                    st.code(train_cmd, language="bash")
                    st.warning("⚠️ Copy command trên và chạy trong terminal để theo dõi log training!")
            
            with col_stop:
                if st.button("⏹️ Cancel", use_container_width=True, disabled=not st.session_state.training_active):
                    st.session_state.training_active = False
                    st.info("Training cancelled")
            
            st.divider()
            
            # Training tips
            with st.expander("💡 Training Tips"):
                st.markdown("""
                **Khuyến nghị:**
                - **GPU VRAM 4GB:** Batch size 4-8
                - **GPU VRAM 8GB:** Batch size 16
                - **GPU VRAM 12GB+:** Batch size 32
                
                **Dataset chuẩn:**
                - **CULane:** ~100GB, 100k images, 4 lanes
                - **TuSimple:** ~10GB, highway only
                - **CurveLanes:** ~30GB, curved roads
                
                **Training time:**
                - RTX 3060: ~12h (CULane 50 epochs)
                - RTX 4090: ~3h (CULane 50 epochs)
                """)
    
    # ========== TAB 4: ONNX TEST ==========
    with tab_onnx_test:
        st.subheader("🧪 ONNX Model Testing")
        st.info("💡 Kiểm tra và validate các file ONNX model trước khi deploy.")
        
        col_select, col_info = st.columns([1, 1])
        
        with col_select:
            st.markdown("### 📁 Select ONNX Model")
            
            # List available ONNX files
            onnx_files = list(ONNX_DIR.glob("*.onnx")) if ONNX_DIR.exists() else []
            
            if not onnx_files:
                st.warning("⚠️ Không tìm thấy file ONNX nào trong weights/onnx/")
                st.info("💡 Chạy: `python deploy/batch_convert_onnx.py` để tạo ONNX files")
            else:
                # Model selector
                onnx_options = {f.name: f for f in onnx_files}
                selected_onnx = st.selectbox(
                    "Chọn file ONNX",
                    options=list(onnx_options.keys()),
                    help="Chọn model ONNX để test"
                )
                
                selected_onnx_path = onnx_options[selected_onnx]
                
                # File info
                st.markdown("### 📊 File Info")
                file_size = selected_onnx_path.stat().st_size / (1024 * 1024)
                
                # Check for .data file
                data_file = selected_onnx_path.with_suffix(".onnx.data")
                if data_file.exists():
                    data_size = data_file.stat().st_size / (1024 * 1024)
                    total_size = file_size + data_size
                    st.markdown(f"""
                    | Property | Value |
                    |----------|-------|
                    | **File** | `{selected_onnx}` |
                    | **Graph Size** | {file_size:.2f} MB |
                    | **Weights** | {data_size:.1f} MB |
                    | **Total Size** | {total_size:.1f} MB |
                    """)
                else:
                    st.markdown(f"""
                    | Property | Value |
                    |----------|-------|
                    | **File** | `{selected_onnx}` |
                    | **Size** | {file_size:.2f} MB |
                    """)
                
                st.divider()
                
                # Provider selection for test
                st.markdown("### ⚙️ Test Settings")
                import onnxruntime as ort
                available_providers = ort.get_available_providers()
                
                test_provider = st.selectbox(
                    "Execution Provider",
                    options=available_providers,
                    help="Provider để test inference"
                )
                
                num_warmup = st.slider("Warmup Iterations", 1, 10, 3, help="Số lần chạy warmup")
                num_iterations = st.slider("Benchmark Iterations", 10, 100, 30, help="Số lần chạy benchmark")
        
        with col_info:
            if onnx_files:
                st.markdown("### 📋 Model Details")
                
                # Run test button
                if st.button("🚀 Run ONNX Test", type="primary", use_container_width=True):
                    with st.spinner("Đang phân tích model..."):
                        try:
                            import onnxruntime as ort
                            import onnx
                            
                            # Load ONNX model for inspection
                            onnx_model = onnx.load(str(selected_onnx_path))
                            
                            # Model metadata
                            st.markdown("#### 📌 Model Metadata")
                            st.markdown(f"- **IR Version:** {onnx_model.ir_version}")
                            st.markdown(f"- **Opset Version:** {onnx_model.opset_import[0].version}")
                            st.markdown(f"- **Producer:** {onnx_model.producer_name or 'N/A'}")
                            
                            st.divider()
                            
                            # Input/Output shapes
                            st.markdown("#### 📥 Inputs")
                            for inp in onnx_model.graph.input:
                                shape = [d.dim_value if d.dim_value > 0 else "dynamic" for d in inp.type.tensor_type.shape.dim]
                                dtype = onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
                                st.code(f"{inp.name}: {shape} ({dtype})")
                            
                            st.markdown("#### 📤 Outputs")
                            for out in onnx_model.graph.output:
                                shape = [d.dim_value if d.dim_value > 0 else "dynamic" for d in out.type.tensor_type.shape.dim]
                                dtype = onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type)
                                st.code(f"{out.name}: {shape} ({dtype})")
                            
                            st.divider()
                            
                            # Create session and run benchmark
                            st.markdown("#### ⚡ Benchmark")
                            
                            sess_options = ort.SessionOptions()
                            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                            
                            session = ort.InferenceSession(
                                str(selected_onnx_path), 
                                sess_options, 
                                providers=[test_provider]
                            )
                            
                            actual_provider = session.get_providers()[0]
                            st.success(f"✅ Provider: **{actual_provider}**")
                            
                            # Get input shape
                            input_info = session.get_inputs()[0]
                            input_shape = input_info.shape
                            # Handle dynamic dims
                            input_shape = [s if isinstance(s, int) and s > 0 else 1 for s in input_shape]
                            
                            # Create dummy input
                            dummy_input = np.random.randn(*input_shape).astype(np.float32)
                            
                            # Warmup
                            progress = st.progress(0)
                            status = st.empty()
                            
                            status.text(f"Warmup ({num_warmup} iterations)...")
                            for i in range(num_warmup):
                                _ = session.run(None, {input_info.name: dummy_input})
                                progress.progress((i + 1) / (num_warmup + num_iterations))
                            
                            # Benchmark
                            status.text(f"Benchmarking ({num_iterations} iterations)...")
                            times = []
                            for i in range(num_iterations):
                                start = time.time()
                                _ = session.run(None, {input_info.name: dummy_input})
                                times.append((time.time() - start) * 1000)
                                progress.progress((num_warmup + i + 1) / (num_warmup + num_iterations))
                            
                            progress.empty()
                            status.empty()
                            
                            # Results
                            avg_time = np.mean(times)
                            min_time = np.min(times)
                            max_time = np.max(times)
                            std_time = np.std(times)
                            fps = 1000 / avg_time
                            
                            st.markdown(f"""
                            | Metric | Value |
                            |--------|-------|
                            | **Average** | {avg_time:.2f} ms |
                            | **Min** | {min_time:.2f} ms |
                            | **Max** | {max_time:.2f} ms |
                            | **Std Dev** | {std_time:.2f} ms |
                            | **FPS** | {fps:.1f} |
                            """)
                            
                            # Performance assessment
                            if avg_time < 10:
                                st.success("🚀 Excellent! Real-time capable (100+ FPS)")
                            elif avg_time < 33:
                                st.success("✅ Good! 30+ FPS achievable")
                            elif avg_time < 100:
                                st.warning("⚠️ Moderate speed (~10-30 FPS)")
                            else:
                                st.error("❌ Slow inference (< 10 FPS)")
                            
                        except Exception as e:
                            st.error(f"❌ Error testing model: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                
                # Quick validation
                st.divider()
                with st.expander("🔍 Quick Validation Checks"):
                    st.markdown("""
                    **Checklist trước khi deploy:**
                    - [ ] Inference chạy không lỗi
                    - [ ] Input shape đúng với expected
                    - [ ] Output shape có 4 tensors (loc_row, loc_col, exist_row, exist_col)
                    - [ ] FPS đạt yêu cầu (>30 cho real-time)
                    - [ ] Memory usage acceptable
                    """)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        🚗 Ultra-Fast Lane Detection V2 | ResNet18 + CULane | GPU Accelerated
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    # Check weights directory exists
    if not WEIGHTS_DIR.exists():
        st.error(f"❌ Weights directory not found at {WEIGHTS_DIR}")
        st.info("Please create weights/pth/ directory and add model files.")
        st.stop()
    
    # Check at least one weight file exists
    weight_files = list(WEIGHTS_DIR.glob("*.pth"))
    if not weight_files:
        st.error("❌ No .pth files found in weights/pth/")
        st.info("Please download model weights and place them in weights/pth/")
        st.stop()
    
    main()
