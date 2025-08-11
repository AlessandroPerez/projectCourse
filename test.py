import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

from notebooks.inference_utils import multiwm_dbscan

from watermark_anything.data.metrics import msg_predict_inference
from notebooks.inference_utils import (
    load_model_from_checkpoint, default_transform, unnormalize_img,
    create_random_mask, msg2str
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_img(path):
    img = Image.open(path).convert("RGB")
    img = default_transform(img).unsqueeze(0).to(device)
    return img

# Load the model from the specified checkpoint
exp_dir = "checkpoints"
json_path = os.path.join(exp_dir, "params.json")
ckpt_path = os.path.join(exp_dir, 'checkpoint.pth')
wam = load_model_from_checkpoint(json_path, ckpt_path).to(device).eval()

# Define the directory containing the images to watermark
img_dir = "assets/images"  # Directory containing the original images
num_imgs = 5  # Number of images to watermark from the folder
output_dir = "outputs"  # Directory to save the watermarked images
os.makedirs(output_dir, exist_ok=True)


# Proportion of the image to be watermarked (0.5 means 50% of the image).
# This is used here to show the watermark localization property. In practice, you may want to use a predifined mask or the entire image.
proportion_masked = 1

# Seed 
seed = 42
torch.manual_seed(seed)

# DBSCAN parameters for detection
epsilon = 1 # min distance between decoded messages in a cluster
min_samples = 128 # min number of pixels in a 256x256 image to form a cluster

# multiple 32 bit message to hide
wm_msgs = wam.get_random_msg(3)
print("Original messages: ", [msg2str(msg) for msg in wm_msgs])
proportion_masked = 1 # max proportion per watermark, randomly placed

for img_ in tqdm(os.listdir(img_dir)[:num_imgs]):
    # Load and preprocess the image
    img_pt = load_img(os.path.join(img_dir, img_))  # [1, 3, H, W]
    
    # Mask to use. 1 values correspond to pixels where the watermark will be placed.
    masks = create_random_mask(img_pt, num_masks=len(wm_msgs), mask_percentage=proportion_masked)  # create one random mask per message
    multi_wm_img = img_pt.clone()
    for ii in range(len(wm_msgs)):
        wm_msg, mask = wm_msgs[ii].unsqueeze(0), masks[ii]
        outputs = wam.embed(img_pt, wm_msg) 
        multi_wm_img = outputs['imgs_w'] * mask + multi_wm_img * (1 - mask)  # [1, 3, H, W]

    # Detect the watermark in the multi-watermarked image
    preds = wam.detect(multi_wm_img)["preds"]  # [1, 33, 256, 256]
    mask_preds = F.sigmoid(preds[:, 0, :, :])  # [1, 256, 256], predicted mask
    mask_preds_res = F.interpolate(mask_preds.unsqueeze(1), size=(img_pt.shape[-2], img_pt.shape[-1]), mode="bilinear", align_corners=False)  # [1, 1, H, W]
    bit_preds = preds[:, 1:, :, :]  # [1, 32, 256, 256], predicted bits

    centroids, positions = multiwm_dbscan(bit_preds, mask_preds, epsilon = epsilon, min_samples = min_samples)
    centroids_pt = torch.stack(list(centroids.values()))
    # positions has the cluster number at each pixel. can be upsaled back to the original size.

    accuracies = []
    hamming_distances = []
    for centroid in centroids_pt:
        bit_acc = (centroid == wm_msgs).float().mean(dim=1)
        # get message with maximum bit accuracy
        bit_acc, idx = bit_acc.max(dim=0)
        hamming = int(torch.sum(centroid != wm_msgs[idx]).item())
        accuracies.append(bit_acc.item())
        hamming_distances.append(hamming)

# Print average results after processing all images
avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
avg_hamming = sum(hamming_distances) / len(hamming_distances) if hamming_distances else 0
print(f"\nResults Summary:")
print(f"Average bit accuracy: {avg_accuracy:.4f}")
print(f"Average Hamming distance: {avg_hamming:.2f}/{len(wm_msgs[0])}")