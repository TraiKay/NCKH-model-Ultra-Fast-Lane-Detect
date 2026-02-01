"""
Batch convert all PTH models to ONNX format.
Supports CPU execution and custom output directory.

Usage:
    # Convert all models to weights/onnx/
    python deploy/batch_convert_onnx.py

    # Convert specific model
    python deploy/batch_convert_onnx.py --model culane_res18

    # Use FP16
    python deploy/batch_convert_onnx.py --accuracy fp16
"""

import torch
import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.common import get_model
from utils.config import Config


# Model configurations
MODELS = {
    "culane_res18": {
        "config": "configs/culane_res18.py",
        "weights": "weights/pth/culane_res18.pth",
        "size": (1600, 320),
    },
    "culane_res34": {
        "config": "configs/culane_res34.py",
        "weights": "weights/pth/culane_res34.pth",
        "size": (1600, 320),
    },
    "tusimple_res18": {
        "config": "configs/tusimple_res18.py",
        "weights": "weights/pth/tusimple_res18.pth",
        "size": (800, 320),
    },
    "tusimple_res34": {
        "config": "configs/tusimple_res34.py",
        "weights": "weights/pth/tusimple_res34.pth",
        "size": (800, 320),
    },
    "curvelanes_res18": {
        "config": "configs/curvelanes_res18.py",
        "weights": "weights/pth/curvelanes_res18.pth",
        "size": (1600, 800),
    },
    "curvelanes_res34": {
        "config": "configs/curvelanes_res34.py",
        "weights": "weights/pth/curvelanes_res34.pth",
        "size": (1600, 800),
    },
}


def convert_to_onnx(model_name: str, output_dir: Path, accuracy: str = "fp32", device: str = "cpu"):
    """Convert a single model to ONNX format."""
    if model_name not in MODELS:
        print(f"❌ Unknown model: {model_name}")
        print(f"   Available: {list(MODELS.keys())}")
        return False
    
    model_info = MODELS[model_name]
    config_path = PROJECT_ROOT / model_info["config"]
    weights_path = PROJECT_ROOT / model_info["weights"]
    width, height = model_info["size"]
    
    # Check files exist
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return False
    
    if not weights_path.exists():
        print(f"❌ Weights not found: {weights_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"🔄 Converting: {model_name}")
    print(f"   Config: {config_path.name}")
    print(f"   Weights: {weights_path.name}")
    print(f"   Size: {width}x{height}")
    print(f"   Device: {device.upper()}")
    print(f"   Accuracy: {accuracy.upper()}")
    print(f"{'='*60}")
    
    try:
        # Load config
        cfg = Config.fromfile(str(config_path))
        cfg.batch_size = 1
        
        # Create model
        net = get_model(cfg)
        
        # Load weights
        state_dict = torch.load(str(weights_path), map_location='cpu', weights_only=False)['model']
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
        
        # Create dummy input
        dummy_input = torch.ones((1, 3, height, width)).to(dev)
        
        # Output path
        output_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = output_dir / f"{model_name}.onnx"
        
        # Export to ONNX
        print(f"📦 Exporting to ONNX...")
        with torch.no_grad():
            torch.onnx.export(
                net,
                dummy_input,
                str(onnx_path),
                verbose=False,
                input_names=['input'],
                output_names=["loc_row", "loc_col", "exist_row", "exist_col"],
                opset_version=11,
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'loc_row': {0: 'batch_size'},
                    'loc_col': {0: 'batch_size'},
                    'exist_row': {0: 'batch_size'},
                    'exist_col': {0: 'batch_size'},
                }
            )
        
        file_size = onnx_path.stat().st_size / (1024 * 1024)
        print(f"✅ Saved: {onnx_path.name} ({file_size:.1f} MB)")
        
        # FP16 conversion (optional)
        if accuracy == "fp16":
            try:
                import onnxmltools
                from onnxmltools.utils.float16_converter import convert_float_to_float16
                
                print(f"🔄 Converting to FP16...")
                onnx_model = onnxmltools.utils.load_model(str(onnx_path))
                onnx_model_fp16 = convert_float_to_float16(onnx_model)
                
                fp16_path = output_dir / f"{model_name}_fp16.onnx"
                onnxmltools.utils.save_model(onnx_model_fp16, str(fp16_path))
                
                fp16_size = fp16_path.stat().st_size / (1024 * 1024)
                print(f"✅ Saved: {fp16_path.name} ({fp16_size:.1f} MB)")
            except ImportError:
                print(f"⚠️ onnxmltools not installed. Skipping FP16 conversion.")
                print(f"   Install with: pip install onnxmltools")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch convert PTH to ONNX")
    parser.add_argument('--model', type=str, default=None,
                        help='Specific model to convert (default: all)')
    parser.add_argument('--output', type=str, default='weights/onnx',
                        help='Output directory (default: weights/onnx)')
    parser.add_argument('--accuracy', type=str, default='fp32', choices=['fp32', 'fp16'],
                        help='Precision (default: fp32)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to use (default: cpu)')
    args = parser.parse_args()
    
    output_dir = PROJECT_ROOT / args.output
    
    print("\n" + "="*60)
    print("🚀 ONNX Batch Converter for Ultra-Fast-Lane-Detection-v2")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device.upper()}")
    print(f"Accuracy: {args.accuracy.upper()}")
    
    # Determine which models to convert
    if args.model:
        models_to_convert = [args.model]
    else:
        models_to_convert = list(MODELS.keys())
    
    print(f"Models to convert: {len(models_to_convert)}")
    
    # Convert models
    success = 0
    failed = 0
    
    for model_name in models_to_convert:
        if convert_to_onnx(model_name, output_dir, args.accuracy, args.device):
            success += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"✅ Success: {success}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output: {output_dir}")
    
    if success > 0:
        print("\n📝 ONNX files created:")
        for f in sorted(output_dir.glob("*.onnx")):
            size = f.stat().st_size / (1024 * 1024)
            print(f"   - {f.name} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
