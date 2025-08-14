# ProvenAIre

A streamlined and more robust implementation of the Watermark Anything model for real-world deployment. Used to discriminate generated images with high precision even under sever agumentations like jpeg compression, blur, rotation and many more

## Installation

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
cd provenAIre
pip install -r requirements.txt
```


**Step 4: Downloading the modle weights**
```bash
wget https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth -P checkpoints/
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

## Demo

To validate the model just run the scrypt `final_ultimate_attribution_benchmark.py`

**Quick System Test:**
```bash
python final_ultimate_attribution_benchmark.py --total_images 40 --output_file test_with_cropping.json
```

This will test on only 40 images, for larger tests just change the `total_images` parameter.

## Advanced Features

**AI Model Attribution:**


**Attack Robustness Testing:**
