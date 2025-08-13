#!/usr/bin/env python3
"""
Watermark Attack Augmentations
Various image processing techniques aimed at removing or degrading watermarks
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
import io
import random
import math

class WatermarkAttacks:
    """Collection of image processing attacks to remove watermarks"""
    
    def __init__(self, device='cpu'):
        self.device = device
    
    def gaussian_blur(self, image_tensor, kernel_size=5, sigma=1.0):
        """Apply Gaussian blur to degrade watermark"""
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Create Gaussian kernel
        channels = image_tensor.shape[1]
        kernel = self._gaussian_kernel_2d(kernel_size, sigma).to(self.device)
        kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
        
        # Apply convolution with padding
        padding = kernel_size // 2
        blurred = F.conv2d(image_tensor, kernel, padding=padding, groups=channels)
        
        return torch.clamp(blurred, 0, 1)
    
    def jpeg_compression(self, image_tensor, quality=50):
        """Simulate JPEG compression artifacts"""
        if image_tensor.dim() == 4:
            batch_size = image_tensor.shape[0]
            results = []
            
            for i in range(batch_size):
                img = image_tensor[i]
                compressed = self._apply_jpeg_single(img, quality)
                results.append(compressed)
            
            return torch.stack(results)
        else:
            return self._apply_jpeg_single(image_tensor, quality)
    
    def _apply_jpeg_single(self, image_tensor, quality):
        """Apply JPEG compression to single image"""
        # Convert tensor to PIL Image
        if image_tensor.dim() == 3:
            img_array = image_tensor.permute(1, 2, 0).cpu().numpy()
        else:
            img_array = image_tensor.cpu().numpy()
        
        img_array = (img_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_array)
        
        # Apply JPEG compression
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_img = Image.open(buffer)
        
        # Convert back to tensor
        compressed_array = np.array(compressed_img).astype(np.float32) / 255.0
        if len(compressed_array.shape) == 3:
            compressed_tensor = torch.from_numpy(compressed_array).permute(2, 0, 1)
        else:
            compressed_tensor = torch.from_numpy(compressed_array)
        
        return compressed_tensor.to(self.device)
    
    def rotation(self, image_tensor, angle_degrees=15):
        """Apply rotation transformation"""
        original_shape = image_tensor.shape
        
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        batch_size = image_tensor.size(0)
        angle_rad = math.radians(angle_degrees)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        rotation_matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Create grid and apply rotation
        grid = F.affine_grid(
            rotation_matrix, 
            image_tensor.size(), 
            align_corners=False
        )
        
        rotated = F.grid_sample(
            image_tensor, 
            grid, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=False
        )
        
        # Restore original shape if needed
        if len(original_shape) == 3:
            rotated = rotated.squeeze(0)
        
        return torch.clamp(rotated, 0, 1)
    
    def additive_white_gaussian_noise(self, image_tensor, noise_std=0.05):
        """Add white Gaussian noise to mask watermark"""
        noise = torch.randn_like(image_tensor) * noise_std
        noisy = image_tensor + noise
        return torch.clamp(noisy, 0, 1)
    
    def sharpening(self, image_tensor, strength=1.0):
        """Apply sharpening filter that can distort watermarks"""
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Sharpening kernel
        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=torch.float32, device=self.device) * strength
        
        # Add center bias
        kernel[1, 1] = 1 + 4 * strength
        
        # Normalize kernel
        kernel = kernel / kernel.sum()
        
        # Expand for all channels
        channels = image_tensor.shape[1]
        kernel = kernel.unsqueeze(0).unsqueeze(0).expand(channels, 1, 3, 3)
        
        # Apply convolution
        sharpened = F.conv2d(image_tensor, kernel, padding=1, groups=channels)
        
        return torch.clamp(sharpened, 0, 1)
    
    def median_filter(self, image_tensor, kernel_size=5):
        """Apply median filter to remove noise-like watermarks"""
        if image_tensor.dim() == 4:
            batch_size = image_tensor.shape[0]
            results = []
            
            for i in range(batch_size):
                img = image_tensor[i]
                filtered = self._apply_median_single(img, kernel_size)
                results.append(filtered)
            
            return torch.stack(results)
        else:
            return self._apply_median_single(image_tensor, kernel_size)
    
    def _apply_median_single(self, image_tensor, kernel_size):
        """Apply median filter to single image"""
        # Convert to PIL and apply median filter
        img_array = image_tensor.permute(1, 2, 0).cpu().numpy()
        img_array = (img_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_array)
        
        # Apply median filter
        filtered_img = pil_img.filter(ImageFilter.MedianFilter(size=kernel_size))
        
        # Convert back to tensor
        filtered_array = np.array(filtered_img).astype(np.float32) / 255.0
        filtered_tensor = torch.from_numpy(filtered_array).permute(2, 0, 1)
        
        return filtered_tensor.to(self.device)
    
    def scaling(self, image_tensor, scale_factor=0.8):
        """Scale image up/down to disrupt watermark"""
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Scale down then back up
        _, _, h, w = image_tensor.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        # Downscale
        scaled_down = F.interpolate(
            image_tensor, 
            size=(new_h, new_w), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Scale back up
        scaled_back = F.interpolate(
            scaled_down, 
            size=(h, w), 
            mode='bilinear', 
            align_corners=False
        )
        
        return torch.clamp(scaled_back, 0, 1)
    
    def _gaussian_kernel_2d(self, kernel_size, sigma):
        """Generate 2D Gaussian kernel"""
        kernel_1d = self._gaussian_kernel_1d(kernel_size, sigma)
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        return kernel_2d.unsqueeze(0).unsqueeze(0)
    
    def _gaussian_kernel_1d(self, kernel_size, sigma):
        """Generate 1D Gaussian kernel"""
        x = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        gaussian = torch.exp(-0.5 * (x / sigma) ** 2)
        return gaussian / gaussian.sum()
    
    def apply_attack_combination(self, image_tensor, attack_types=None, attack_params=None):
        """Apply combination of attacks"""
        if attack_types is None:
            attack_types = ['blur', 'jpeg', 'rotation', 'awgn', 'sharpening']
        
        if attack_params is None:
            attack_params = {
                'blur': {'kernel_size': 5, 'sigma': 1.0},
                'jpeg': {'quality': 50},
                'rotation': {'angle_degrees': 10},
                'awgn': {'noise_std': 0.03},
                'sharpening': {'strength': 0.5},
                'median': {'kernel_size': 3},
                'scaling': {'scale_factor': 0.9}
            }
        
        attacked_image = image_tensor.clone()
        applied_attacks = []
        
        for attack in attack_types:
            if attack == 'blur':
                attacked_image = self.gaussian_blur(
                    attacked_image, 
                    **attack_params.get('blur', {})
                )
                applied_attacks.append('blur')
            
            elif attack == 'jpeg':
                attacked_image = self.jpeg_compression(
                    attacked_image, 
                    **attack_params.get('jpeg', {})
                )
                applied_attacks.append('jpeg')
            
            elif attack == 'rotation':
                attacked_image = self.rotation(
                    attacked_image, 
                    **attack_params.get('rotation', {})
                )
                applied_attacks.append('rotation')
            
            elif attack == 'awgn':
                attacked_image = self.additive_white_gaussian_noise(
                    attacked_image, 
                    **attack_params.get('awgn', {})
                )
                applied_attacks.append('awgn')
            
            elif attack == 'sharpening':
                attacked_image = self.sharpening(
                    attacked_image, 
                    **attack_params.get('sharpening', {})
                )
                applied_attacks.append('sharpening')
            
            elif attack == 'median':
                attacked_image = self.median_filter(
                    attacked_image, 
                    **attack_params.get('median', {})
                )
                applied_attacks.append('median')
            
            elif attack == 'scaling':
                attacked_image = self.scaling(
                    attacked_image, 
                    **attack_params.get('scaling', {})
                )
                applied_attacks.append('scaling')
        
        return attacked_image, applied_attacks

# Predefined attack configurations
ATTACK_PRESETS = {
    'mild': {
        'attack_types': ['blur', 'jpeg', 'awgn'],
        'attack_params': {
            'blur': {'kernel_size': 3, 'sigma': 0.5},
            'jpeg': {'quality': 80},
            'awgn': {'noise_std': 0.01}
        }
    },
    'moderate': {
        'attack_types': ['blur', 'jpeg', 'rotation', 'awgn'],
        'attack_params': {
            'blur': {'kernel_size': 5, 'sigma': 1.0},
            'jpeg': {'quality': 60},
            'rotation': {'angle_degrees': 5},
            'awgn': {'noise_std': 0.03}
        }
    },
    'strong': {
        'attack_types': ['blur', 'jpeg', 'rotation', 'awgn', 'sharpening'],
        'attack_params': {
            'blur': {'kernel_size': 7, 'sigma': 1.5},
            'jpeg': {'quality': 40},
            'rotation': {'angle_degrees': 15},
            'awgn': {'noise_std': 0.05},
            'sharpening': {'strength': 1.0}
        }
    },
    'extreme': {
        'attack_types': ['blur', 'jpeg', 'rotation', 'awgn', 'sharpening', 'median', 'scaling'],
        'attack_params': {
            'blur': {'kernel_size': 9, 'sigma': 2.0},
            'jpeg': {'quality': 20},
            'rotation': {'angle_degrees': 30},
            'awgn': {'noise_std': 0.08},
            'sharpening': {'strength': 1.5},
            'median': {'kernel_size': 5},
            'scaling': {'scale_factor': 0.7}
        }
    }
}

def apply_preset_attack(image_tensor, preset='moderate', device='cpu'):
    """Apply predefined attack preset"""
    attacks = WatermarkAttacks(device)
    
    if preset not in ATTACK_PRESETS:
        print(f"Unknown preset '{preset}', using 'moderate'")
        preset = 'moderate'
    
    config = ATTACK_PRESETS[preset]
    attacked_image, applied_attacks = attacks.apply_attack_combination(
        image_tensor,
        attack_types=config['attack_types'],
        attack_params=config['attack_params']
    )
    
    return attacked_image, applied_attacks

if __name__ == "__main__":
    # Test the attacks
    print("🔧 Testing Watermark Attacks...")
    
    # Create test tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_image = torch.rand(1, 3, 256, 256).to(device)
    
    attacks = WatermarkAttacks(device)
    
    # Test individual attacks
    print("Testing individual attacks:")
    blurred = attacks.gaussian_blur(test_image)
    print(f"✓ Blur: {blurred.shape}")
    
    compressed = attacks.jpeg_compression(test_image, quality=50)
    print(f"✓ JPEG: {compressed.shape}")
    
    rotated = attacks.rotation(test_image, angle_degrees=15)
    print(f"✓ Rotation: {rotated.shape}")
    
    noisy = attacks.additive_white_gaussian_noise(test_image)
    print(f"✓ AWGN: {noisy.shape}")
    
    sharpened = attacks.sharpening(test_image)
    print(f"✓ Sharpening: {sharpened.shape}")
    
    # Test preset attacks
    print("\nTesting preset attacks:")
    for preset in ATTACK_PRESETS.keys():
        attacked, applied = apply_preset_attack(test_image, preset, device)
        print(f"✓ {preset.upper()}: {attacked.shape}, attacks: {applied}")
    
    print("🎉 All attack tests completed successfully!")
