# Production Watermark System

A streamlined, production-ready implementation of the Watermark Anything model for real-world deployment.

## Features

- **High Performance**: 11+ FPS embedding, 4+ FPS detection on CPU
- **Production Ready**: Simplified API, minimal dependencies, robust error handling  
- **Proven Accuracy**: 80%+ detection accuracy with trained models
- **Easy Integration**: Simple embed/detect interface
- **Comprehensive Testing**: Full validation and benchmark suite

## Prerequisites & Installation

**System Requirements:**
- Python 3.10.14 (required for modern type annotations)
- PyTorch 2.0+ with CUDA 11.8+ (or CPU-only)
- 4GB+ RAM, GPU recommended

## Conda Environment

**Step 1: Create Python 3.10.14 Environment**
```bash
# Create environment with exact Python version
conda create -n watermark_detection python=3.10.14 -y
conda activate watermark_detection
```

**Step 2: Install PyTorch**
```bash
# For GPU (CUDA 11.8) - if this fails, use CPU fallback below
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# For CPU-only (fallback if GPU installation fails)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Step 3: Install Required Packages**
```bash
pip install -r requirements.txt
```


**Step 4: Downloading the modle weights**
```bash
wget https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth -P checkpoints/
```

## Validation & Testing

**Quick System Test:**
```bash
# Test core functionality with trained model
python validate_system_original.py
```

**Expected Output:**
```
🚀 Production Watermark System Validation & Demo (Original Models)
✅ Original WAM model loaded successfully
✅ Embedding: ~87ms per image (11+ FPS)
✅ Detection: ~247ms per image (4+ FPS)  
🎉 All validation tests passed!
📋 System Status: READY FOR PRODUCTION
```

**Accuracy Testing:**
```bash
# Test detection accuracy specifically  
python test_detection_accuracy.py

# Expected: 80%+ accuracy distinguishing watermarked vs clean images
```

**Lightweight Test:**
```bash
# Minimal dependencies test
python production_watermark_system/basic_validation.py
```

## Troubleshooting

**Common Issues:**

1. **Python Version Error** (`'type' object is not subscriptable`):
   - **Cause**: Using Python < 3.10
   - **Solution**: Use Python 3.10.14+ (required for modern type annotations)

2. **Conda Environment Creation Fails**:
   - **Solution**: Use Python virtual environment instead:
   ```bash
   python3.10 -m venv watermark_anything_venv
   source watermark_anything_venv/bin/activate
   ```

3. **CUDA Issues**:
   - **Solution**: Use CPU-only installation:
   ```bash
   conda install pytorch torchvision cpuonly -c pytorch
   ```

4. **Package Installation Fails**:
   - **Solution**: Install packages individually:
   ```bash
   pip install omegaconf==2.3.0 einops==0.8.0 timm==1.0.11 lpips==0.1.4
   ```

## Quick Demo

**Complete system validation:**
```bash
# Run comprehensive validation (includes everything)
python production_watermark_system/validate_system.py

# Expected: 600+ FPS embedding, 700+ FPS detection
```

**Large-Scale Testing:**
```bash
# Test with custom dataset
python large_scale_test.py --num_images 1000 --test_dir results/

# Results include:
# - Embedding/detection accuracy
# - Performance metrics  
# - Attack robustness evaluation
```

## Advanced Features

**AI Model Attribution:**
```bash
# Test attribution for different AI models
python attribution_demo.py --model_ids 0,1,5,9
```

**Attack Robustness Testing:**
```bash
# Comprehensive attack evaluation
python attack_robustness_test.py --attack_suite comprehensive
```
