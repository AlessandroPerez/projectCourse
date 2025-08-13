#!/usr/bin/env python3
"""
FINAL COMPREHENSIVE ATTRIBUTION BENCHMARK
=========================================

Ultimate benchmark testing the complete attribution system with:
- Balanced dataset: 50% watermarked, 50% clean
- Equal distribution across 10 AI models (10% each of watermarked)
- Attacks from watermark_attacks.py at different intensities
- F1 scores for detection and attribution accuracy
- Arbitrary number of test images
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import json
import time
from datetime import datetime
import torchvision.transforms as transforms
from tqdm import tqdm
import random
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# Import ultimate system and attacks
from ultimate_attribution_system import UltimateRobustAttributionSystem
from watermark_attacks import WatermarkAttacks, ATTACK_PRESETS

class FinalComprehensiveAttributionBenchmark:
    """Final benchmark for the ultimate attribution system"""
    
    def __init__(self):
        print("🏆 FINAL COMPREHENSIVE ATTRIBUTION BENCHMARK")
        print("=" * 50)
        print("Ultimate Watermark Detection + Attribution System")
        print("Testing: Balanced Dataset + Attack Robustness + F1 Metrics")
        
        # Initialize ultimate system
        self.attribution_system = UltimateRobustAttributionSystem()
        
        # Initialize attack system
        self.attack_system = WatermarkAttacks(device=self.attribution_system.watermark_tester.device)
        
        # Attack configurations from watermark_attacks.py
        self.attack_configs = {
            # No attack (baseline)
            "clean": {"type": "none", "intensity": "none"},
            
            # Individual attacks at different intensities
            "blur_mild": {"type": "blur", "intensity": "mild", "params": {"kernel_size": 3, "sigma": 0.5}},
            "blur_moderate": {"type": "blur", "intensity": "moderate", "params": {"kernel_size": 5, "sigma": 1.0}},
            "blur_strong": {"type": "blur", "intensity": "strong", "params": {"kernel_size": 7, "sigma": 1.5}},
            
            "jpeg_mild": {"type": "jpeg", "intensity": "mild", "params": {"quality": 80}},
            "jpeg_moderate": {"type": "jpeg", "intensity": "moderate", "params": {"quality": 60}},
            "jpeg_strong": {"type": "jpeg", "intensity": "strong", "params": {"quality": 40}},
            "jpeg_extreme": {"type": "jpeg", "intensity": "extreme", "params": {"quality": 20}},
            
            "rotation_mild": {"type": "rotation", "intensity": "mild", "params": {"angle_degrees": 5}},
            "rotation_moderate": {"type": "rotation", "intensity": "moderate", "params": {"angle_degrees": 10}},
            "rotation_strong": {"type": "rotation", "intensity": "strong", "params": {"angle_degrees": 15}},
            "rotation_extreme": {"type": "rotation", "intensity": "extreme", "params": {"angle_degrees": 30}},
            
            "noise_mild": {"type": "awgn", "intensity": "mild", "params": {"noise_std": 0.02}},
            "noise_moderate": {"type": "awgn", "intensity": "moderate", "params": {"noise_std": 0.03}},
            "noise_strong": {"type": "awgn", "intensity": "strong", "params": {"noise_std": 0.05}},
            "noise_extreme": {"type": "awgn", "intensity": "extreme", "params": {"noise_std": 0.08}},
            
            "scaling_mild": {"type": "scaling", "intensity": "mild", "params": {"scale_factor": 0.9}},
            "scaling_moderate": {"type": "scaling", "intensity": "moderate", "params": {"scale_factor": 0.8}},
            "scaling_strong": {"type": "scaling", "intensity": "strong", "params": {"scale_factor": 0.7}},
            
            "cropping_mild": {"type": "cropping", "intensity": "mild", "params": {"crop_ratio": 0.9}},
            "cropping_moderate": {"type": "cropping", "intensity": "moderate", "params": {"crop_ratio": 0.8}},
            "cropping_strong": {"type": "cropping", "intensity": "strong", "params": {"crop_ratio": 0.7}},
            "cropping_extreme": {"type": "cropping", "intensity": "extreme", "params": {"crop_ratio": 0.6}},
            
            "sharpening_mild": {"type": "sharpening", "intensity": "mild", "params": {"strength": 0.5}},
            "sharpening_moderate": {"type": "sharpening", "intensity": "moderate", "params": {"strength": 1.0}},
            "sharpening_strong": {"type": "sharpening", "intensity": "strong", "params": {"strength": 1.5}},
            
            # Combination attacks (presets from watermark_attacks.py)
            "combo_mild": {"type": "preset", "intensity": "mild", "params": {"preset": "mild"}},
            "combo_moderate": {"type": "preset", "intensity": "moderate", "params": {"preset": "moderate"}},
            "combo_strong": {"type": "preset", "intensity": "strong", "params": {"preset": "strong"}},
            "combo_extreme": {"type": "preset", "intensity": "extreme", "params": {"preset": "extreme"}},
        }
        
        print(f"   ✅ Ultimate attribution system loaded")
        print(f"   ⚔️  Attack system loaded")
        print(f"   📊 Testing {len(self.attack_configs)} attack variants")
        print(f"   🎯 Focus: Balanced dataset + F1 metrics")
        print(f"   📈 Models: {len(self.attribution_system.ai_models)} AI models")

    def create_balanced_dataset(self, total_images: int) -> list:
        """
        Create balanced dataset:
        - 50% watermarked (distributed equally across 10 AI models)
        - 50% clean (no watermark)
        """
        print(f"\n🎨 CREATING BALANCED DATASET")
        print("=" * 30)
        
        # Calculate split
        watermarked_count = total_images // 2
        clean_count = total_images - watermarked_count
        images_per_model = watermarked_count // len(self.attribution_system.ai_models)
        
        print(f"   📊 Total images: {total_images}")
        print(f"   💧 Watermarked: {watermarked_count} ({watermarked_count/total_images:.1%})")
        print(f"   🧹 Clean: {clean_count} ({clean_count/total_images:.1%})")
        print(f"   🤖 Per AI model: {images_per_model}")
        
        dataset = []
        device = self.attribution_system.watermark_tester.device
        
        # Create watermarked images (distributed across AI models)
        with tqdm(total=watermarked_count, desc="Creating watermarked images", 
                 bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}') as pbar:
            
            for model_id in self.attribution_system.ai_models.keys():
                for i in range(images_per_model):
                    try:
                        # Create test image
                        test_image = torch.rand(1, 3, 256, 256, device=device)
                        
                        # Generate message for this AI model
                        message = self.attribution_system.encode_ai_message(model_id)
                        
                        # Watermark it
                        with torch.no_grad():
                            watermarked = self.attribution_system.watermark_tester.embedder(test_image, message)
                        
                        dataset.append({
                            "image": watermarked,
                            "message": message,
                            "true_model_id": model_id,
                            "model_name": self.attribution_system.ai_models[model_id],
                            "is_watermarked": True,
                            "index": len(dataset)
                        })
                        
                    except Exception as e:
                        print(f"Error creating watermarked image: {e}")
                    
                    pbar.update(1)
        
        # Create clean images (no watermark)
        with tqdm(total=clean_count, desc="Creating clean images", 
                 bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}') as pbar:
            
            for i in range(clean_count):
                try:
                    # Create clean test image
                    clean_image = torch.rand(1, 3, 256, 256, device=device)
                    
                    dataset.append({
                        "image": clean_image,
                        "message": None,
                        "true_model_id": -1,  # No model (clean)
                        "model_name": "Clean",
                        "is_watermarked": False,
                        "index": len(dataset)
                    })
                    
                except Exception as e:
                    print(f"Error creating clean image: {e}")
                
                pbar.update(1)
        
        # Shuffle dataset
        random.shuffle(dataset)
        
        print(f"   ✅ Created {len(dataset)} total images")
        return dataset

    def apply_attack(self, image: torch.Tensor, attack_config: dict) -> torch.Tensor:
        """Apply attack based on configuration"""
        if attack_config["type"] == "none":
            return image
        
        try:
            if attack_config["type"] == "blur":
                return self.attack_system.gaussian_blur(image, **attack_config["params"])
            elif attack_config["type"] == "jpeg":
                return self.attack_system.jpeg_compression(image, **attack_config["params"])
            elif attack_config["type"] == "rotation":
                return self.attack_system.rotation(image, **attack_config["params"])
            elif attack_config["type"] == "awgn":
                return self.attack_system.additive_white_gaussian_noise(image, **attack_config["params"])
            elif attack_config["type"] == "scaling":
                return self.attack_system.scaling(image, **attack_config["params"])
            elif attack_config["type"] == "cropping":
                return self.apply_cropping_attack(image, **attack_config["params"])
            elif attack_config["type"] == "sharpening":
                return self.attack_system.sharpening(image, **attack_config["params"])
            elif attack_config["type"] == "preset":
                # Use preset attacks from watermark_attacks.py
                preset = attack_config["params"]["preset"]
                if preset in ATTACK_PRESETS:
                    config = ATTACK_PRESETS[preset]
                    attacked_image, _ = self.attack_system.apply_attack_combination(
                        image,
                        attack_types=config['attack_types'],
                        attack_params=config['attack_params']
                    )
                    return attacked_image
                else:
                    return image
            else:
                return image
                
        except Exception as e:
            print(f"Error applying attack {attack_config['type']}: {e}")
            return image

    def apply_cropping_attack(self, image: torch.Tensor, crop_ratio: float) -> torch.Tensor:
        """
        Apply cropping attack by cropping to crop_ratio of original size and then resizing back
        
        Args:
            image: Input image tensor [B, C, H, W]
            crop_ratio: Ratio of original size to keep (0.9 = keep 90% of image)
        """
        try:
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
            batch_size, channels, height, width = image.shape
            
            # Calculate cropped dimensions
            crop_height = int(height * crop_ratio)
            crop_width = int(width * crop_ratio)
            
            # Calculate starting position for center crop
            start_h = (height - crop_height) // 2
            start_w = (width - crop_width) // 2
            
            # Crop the image
            cropped = image[:, :, start_h:start_h + crop_height, start_w:start_w + crop_width]
            
            # Resize back to original dimensions
            resized = F.interpolate(
                cropped, 
                size=(height, width), 
                mode='bilinear', 
                align_corners=False
            )
            
            return torch.clamp(resized, 0, 1)
            
        except Exception as e:
            print(f"Error in cropping attack: {e}")
            return image

    def calculate_metrics(self, y_true: list, y_pred: list, labels: list = None) -> dict:
        """Calculate F1, precision, recall metrics"""
        try:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            if labels is None:
                # Binary classification
                f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
                precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
                recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
                
                return {
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall
                }
            else:
                # Multi-class classification
                f1_macro = f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
                f1_micro = f1_score(y_true, y_pred, average='micro', labels=labels, zero_division=0)
                precision_macro = precision_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
                recall_macro = recall_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
                
                return {
                    "f1_score_macro": f1_macro,
                    "f1_score_micro": f1_micro,
                    "precision_macro": precision_macro,
                    "recall_macro": recall_macro
                }
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {"f1_score": 0.0, "precision": 0.0, "recall": 0.0}
    def run_final_comprehensive_benchmark(self, total_images: int = 1000) -> dict:
        """Run the ultimate comprehensive benchmark with balanced dataset"""
        
        print(f"\n🚀 STARTING FINAL COMPREHENSIVE BENCHMARK")
        print("=" * 45)
        
        # Create balanced dataset
        dataset = self.create_balanced_dataset(total_images)
        
        # Test all attacks
        print(f"\n⚔️  TESTING ATTACK ROBUSTNESS")
        print("=" * 30)
        
        results = {}
        
        for attack_name, attack_config in self.attack_configs.items():
            print(f"\n   ⚔️  Testing {attack_name} ({attack_config['intensity']}):")
            
            # Lists for metric calculation
            detection_true = []  # Ground truth: 1 if watermarked, 0 if clean
            detection_pred = []  # Prediction: 1 if detected, 0 if not
            attribution_true = []  # Ground truth model IDs (only for watermarked)
            attribution_pred = []  # Predicted model IDs (only for watermarked)
            
            attack_times = []
            confidences = []
            
            # Sample subset of dataset for each attack (for efficiency)
            sample_size = min(200, len(dataset))
            attack_sample = random.sample(dataset, sample_size)
            
            with tqdm(total=len(attack_sample), desc=f"  {attack_name}",
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}') as pbar:
                
                for test_item in attack_sample:
                    try:
                        # Apply attack
                        start_time = time.time()
                        attacked_image = self.apply_attack(test_item["image"], attack_config)
                        
                        # Test detection and attribution
                        result = self.attribution_system.ultimate_detection_with_attribution(
                            attacked_image, 
                            attack_name,
                            test_item["true_model_id"] if test_item["is_watermarked"] else None
                        )
                        
                        detection_time = time.time() - start_time
                        
                        # Record ground truth and predictions for detection
                        detection_true.append(1 if test_item["is_watermarked"] else 0)
                        detection_pred.append(1 if result["detected"] else 0)
                        
                        # Record attribution only for watermarked images
                        if test_item["is_watermarked"]:
                            attribution_true.append(test_item["true_model_id"])
                            if result["detected"] and result["attributed"]:
                                # Extract model_id from attribution_info
                                predicted_model_id = result.get("attribution_info", {}).get("model_id", -1)
                                attribution_pred.append(predicted_model_id)
                            else:
                                attribution_pred.append(-1)  # Failed attribution
                        
                        attack_times.append(detection_time)
                        confidences.append(result["confidence"])
                        
                    except Exception as e:
                        # Handle errors gracefully
                        detection_true.append(1 if test_item["is_watermarked"] else 0)
                        detection_pred.append(0)  # Failed detection
                        if test_item["is_watermarked"]:
                            attribution_true.append(test_item["true_model_id"])
                            attribution_pred.append(-1)  # Failed attribution
                        attack_times.append(0.0)
                        confidences.append(0.0)
                    
                    pbar.update(1)
            
            # Calculate detection metrics (binary classification)
            detection_metrics = self.calculate_metrics(detection_true, detection_pred)
            
            # Calculate attribution metrics (multi-class classification)
            attribution_metrics = {}
            if attribution_true and attribution_pred:
                valid_labels = list(self.attribution_system.ai_models.keys())
                attribution_metrics = self.calculate_metrics(
                    attribution_true, 
                    attribution_pred, 
                    labels=valid_labels
                )
            
            # Calculate additional statistics
            total_watermarked = sum(detection_true)
            total_clean = len(detection_true) - total_watermarked
            true_positives = sum(1 for t, p in zip(detection_true, detection_pred) if t == 1 and p == 1)
            false_positives = sum(1 for t, p in zip(detection_true, detection_pred) if t == 0 and p == 1)
            true_negatives = sum(1 for t, p in zip(detection_true, detection_pred) if t == 0 and p == 0)
            false_negatives = sum(1 for t, p in zip(detection_true, detection_pred) if t == 1 and p == 0)
            
            # Attribution accuracy among detected watermarks
            correct_attributions = sum(1 for t, p in zip(attribution_true, attribution_pred) if t == p and p != -1)
            total_attributed = len([p for p in attribution_pred if p != -1])
            attribution_accuracy = correct_attributions / len(attribution_true) if attribution_true else 0
            
            avg_confidence = np.mean(confidences) if confidences else 0
            avg_time = np.mean(attack_times) if attack_times else 0
            
            results[attack_name] = {
                "attack_config": attack_config,
                "sample_size": len(attack_sample),
                "total_watermarked": total_watermarked,
                "total_clean": total_clean,
                
                # Detection metrics
                "detection_metrics": detection_metrics,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "true_negatives": true_negatives,
                "false_negatives": false_negatives,
                
                # Attribution metrics
                "attribution_metrics": attribution_metrics,
                "attribution_accuracy": attribution_accuracy,
                "correct_attributions": correct_attributions,
                "total_attributed": total_attributed,
                
                # Performance metrics
                "avg_confidence": avg_confidence,
                "avg_time": avg_time,
                "intensty": attack_config["intensity"]
            }
            
            # Print immediate results
            print(f"      🎯 Detection F1: {detection_metrics.get('f1_score', 0):.3f}")
            print(f"      🎯 Attribution F1: {attribution_metrics.get('f1_score_macro', 0):.3f}")
            print(f"      📊 TP: {true_positives}, FP: {false_positives}, TN: {true_negatives}, FN: {false_negatives}")
            print(f"      🤖 Correct Attribution: {correct_attributions}/{len(attribution_true)} ({attribution_accuracy:.1%})")
            print(f"      ⏱️  Avg Time: {avg_time:.3f}s")
        
        return results

    def print_final_benchmark_results(self, results: dict):
        """Print comprehensive benchmark results with F1 scores"""
        
        print(f"\n🏆 FINAL COMPREHENSIVE ATTRIBUTION BENCHMARK RESULTS")
        print("=" * 60)
        
        # Summary statistics
        all_attacks = list(results.keys())
        print(f"\n📊 BENCHMARK SUMMARY:")
        print(f"   • Total attacks tested: {len(all_attacks)}")
        print(f"   • Attack intensities: {len(set(results[attack]['intensty'] for attack in all_attacks))}")
        print(f"   • Balanced dataset: 50% watermarked, 50% clean")
        print(f"   • AI models tested: {len(self.attribution_system.ai_models)}")
        
        # Group results by intensity
        by_intensity = {}
        for attack_name, attack_data in results.items():
            intensity = attack_data.get('intensty', 'unknown')
            if intensity not in by_intensity:
                by_intensity[intensity] = []
            by_intensity[intensity].append((attack_name, attack_data))
        
        print(f"\n⚔️  ATTACK RESISTANCE BY INTENSITY:")
        print(f"   {'Intensity':<12} {'Attacks':<8} {'Detection F1':<12} {'Attribution F1':<14} {'Avg Time':<10}")
        print(f"   {'-'*12} {'-'*8} {'-'*12} {'-'*14} {'-'*10}")
        
        for intensity in ['none', 'mild', 'moderate', 'strong', 'extreme']:
            if intensity in by_intensity:
                attacks_data = by_intensity[intensity]
                det_f1_scores = [data['detection_metrics'].get('f1_score', 0) for _, data in attacks_data]
                attr_f1_scores = [data['attribution_metrics'].get('f1_score_macro', 0) for _, data in attacks_data]
                avg_times = [data['avg_time'] for _, data in attacks_data]
                
                avg_det_f1 = np.mean(det_f1_scores) if det_f1_scores else 0
                avg_attr_f1 = np.mean(attr_f1_scores) if attr_f1_scores else 0
                avg_time = np.mean(avg_times) if avg_times else 0
                
                print(f"   {intensity.upper():<12} {len(attacks_data):<8} {avg_det_f1:<11.3f} {avg_attr_f1:<13.3f} {avg_time:<9.3f}s")
        
        # Detailed results for each attack
        print(f"\n📈 DETAILED ATTACK RESULTS:")
        print(f"   {'Attack':<20} {'Type':<12} {'Det F1':<8} {'Attr F1':<8} {'TP':<4} {'FP':<4} {'TN':<4} {'FN':<4}")
        print(f"   {'-'*20} {'-'*12} {'-'*8} {'-'*8} {'-'*4} {'-'*4} {'-'*4} {'-'*4}")
        
        for attack_name, attack_data in results.items():
            attack_type = attack_data['attack_config']['type']
            det_f1 = attack_data['detection_metrics'].get('f1_score', 0)
            attr_f1 = attack_data['attribution_metrics'].get('f1_score_macro', 0)
            tp = attack_data['true_positives']
            fp = attack_data['false_positives']
            tn = attack_data['true_negatives']
            fn = attack_data['false_negatives']
            
            print(f"   {attack_name:<20} {attack_type:<12} {det_f1:<7.3f} {attr_f1:<7.3f} {tp:<4} {fp:<4} {tn:<4} {fn:<4}")
        
        # Overall performance assessment
        all_det_f1 = [data['detection_metrics'].get('f1_score', 0) for data in results.values()]
        all_attr_f1 = [data['attribution_metrics'].get('f1_score_macro', 0) for data in results.values()]
        
        overall_det_f1 = np.mean(all_det_f1)
        overall_attr_f1 = np.mean(all_attr_f1)
        min_det_f1 = np.min(all_det_f1)
        min_attr_f1 = np.min(all_attr_f1)
        
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"   • Average Detection F1: {overall_det_f1:.3f}")
        print(f"   • Average Attribution F1: {overall_attr_f1:.3f}")
        print(f"   • Minimum Detection F1: {min_det_f1:.3f}")
        print(f"   • Minimum Attribution F1: {min_attr_f1:.3f}")
        
        # Performance verdict
        if min_det_f1 >= 0.90 and min_attr_f1 >= 0.80:
            verdict = "🏆 EXCEPTIONAL - Production Ready"
        elif min_det_f1 >= 0.80 and min_attr_f1 >= 0.70:
            verdict = "✅ EXCELLENT - Near Production"
        elif min_det_f1 >= 0.70 and min_attr_f1 >= 0.60:
            verdict = "✅ GOOD - Requires Minor Tuning"
        else:
            verdict = "⚠️  NEEDS IMPROVEMENT"
        
        print(f"   • Final Verdict: {verdict}")
        
        # Specific improvements highlighted
        print(f"\n🚀 SYSTEM CAPABILITIES:")
        print("=" * 25)
        print(f"   ✅ Balanced testing (50%/50% split)")
        print(f"   ✅ F1 score evaluation (detection + attribution)")
        print(f"   ✅ Multiple attack intensities")
        print(f"   ✅ Comprehensive attack suite")
        print(f"   ✅ Real-world applicable metrics")


def main():
    """Run final comprehensive benchmark with command line arguments"""
    parser = argparse.ArgumentParser(description='Final Comprehensive Attribution Benchmark')
    parser.add_argument('--total_images', type=int, default=1000,
                       help='Total number of test images (default: 1000)')
    parser.add_argument('--output_file', type=str, default='final_benchmark_results.json',
                       help='Output file for results (default: final_benchmark_results.json)')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    try:
        print(f"🔧 Configuration:")
        print(f"   • Total images: {args.total_images}")
        print(f"   • Output file: {args.output_file}")
        print(f"   • Balanced dataset: 50% watermarked, 50% clean")
        print(f"   • Per AI model: {args.total_images // 20} watermarked images")
        
        # Initialize benchmark
        benchmark = FinalComprehensiveAttributionBenchmark()
        
        # Run comprehensive benchmark
        results = benchmark.run_final_comprehensive_benchmark(total_images=args.total_images)
        
        # Print results
        benchmark.print_final_benchmark_results(results)
        
        # Save results
        timestamp = datetime.now().isoformat()
        results["benchmark_info"] = {
            "timestamp": timestamp,
            "duration_seconds": time.time() - start_time,
            "total_images": args.total_images,
            "balanced_dataset": True,
            "watermarked_images": args.total_images // 2,
            "clean_images": args.total_images // 2,
            "ai_models": len(benchmark.attribution_system.ai_models),
            "attacks_tested": len(benchmark.attack_configs),
            "model_type": "Ultimate Robust Attribution System",
            "test_scope": "Balanced Dataset + F1 Metrics + Attack Robustness",
            "improvements": [
                "Balanced 50/50 dataset",
                "F1 score evaluation",
                "Comprehensive attack suite",
                "Real-world applicable metrics"
            ]
        }
        
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  Benchmark completed in {elapsed_time/60:.1f} minutes")
        print(f"   💾 Results saved: {args.output_file}")
        
        print(f"\n💡 FINAL CONCLUSION:")
        print(f"   The Final Comprehensive Attribution Benchmark provides")
        print(f"   balanced evaluation with F1 scores, comprehensive attack")
        print(f"   testing, and real-world applicable metrics for watermark")
        print(f"   detection and AI model attribution systems.")
        
    except Exception as e:
        print(f"\n❌ Error in benchmark: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
