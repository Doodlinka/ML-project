import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# load
import os
os.chdir(os.path.dirname(__file__))
model = YOLO('./runs/segment/ForestSeg/fixed_bush_nano_15k/weights/best.pt', 'segment')

IMGS_TO_CHECK = [
    './images/000000004186.png',
    './images/000000005444.png',
    './images/000000000795.png',
    './images/000000006477.png',
    './images/000000022161.png',
    './images/000000019080.png',
    './images/000000018382.png',
    './images/000000017619.png',
]

# predict
results = model.predict(source=IMGS_TO_CHECK, save=False)

for result in results:
    # happens to backgrounds
    
    if not result.masks: 
        print("empty image, skipping")
        continue
    # convert tensor to mask
    masks_tensor = result.masks.data
    masks_np = masks_tensor.cpu().numpy()
    binary_mask = (masks_np[0] * 255).astype(np.uint8)


    kernel = np.ones((5, 5), np.uint8)
    # contracts, then expands objects, smoothing edges and removing strays
    mask_opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    # expands, then contracts objects, closing tiny gaps
    mask_cleaned = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    axes[0].imshow(binary_mask, cmap='gray')
    axes[0].set_title("original")
    axes[0].axis('off')

    axes[1].imshow(mask_cleaned, cmap='gray')
    axes[1].set_title("fixed")
    axes[1].axis('off')
    
    axes[2].imshow(result.orig_img)
    axes[2].set_title("image")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()