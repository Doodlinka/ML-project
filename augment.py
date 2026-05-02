import os
import cv2
import random
import shutil
import albumentations as A
from pathlib import Path

# no geometric - can't be assed to edit labels
transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.GaussNoise(std_range=(0.02, 0.05), mean_range=(0.0, 0.0), p=0.3),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.4),
    A.CLAHE(clip_limit=2.0, p=0.2)
])

def create_augmented_dataset(targetimgs, aug_count, start_count):
    augmented_image_paths = []
    print(f"{start_count}/{start_count + aug_count}")

    for i in range(start_count, start_count + aug_count):
        img_path = random.choice(targetimgs)
        img_path_obj = Path(img_path)

        new_img_name = f"{img_path_obj.stem}_aug_{i}{img_path_obj.suffix}"
        new_img_path = img_path_obj.parent / new_img_name

        lbl_path = Path(str(img_path).replace('images', 'labels')).with_suffix('.txt')
        new_lbl_path = Path(str(new_img_path).replace('images', 'labels')).with_suffix('.txt')

        image = cv2.imread(str(img_path))
        if image is None: continue # i'm gonna hope borked images are fine
            
        # albumentations expects RGB, OpenCV uses BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        augmented = transform(image=image)
        aug_img_bgr = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(new_img_path), aug_img_bgr)

        # if label doesn't exist it's intended (empty)
        if lbl_path.exists():
            shutil.copy(str(lbl_path), str(new_lbl_path))

        augmented_image_paths.append(str(new_img_path))

        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{start_count + aug_count}")

    return augmented_image_paths


def create_several_augs(base_split_txt, aug_count=5000, iter_count=3):
    with open(base_split_txt, 'r') as f:
        baseimgs = [line.strip() for line in f if line.strip()]
    if not baseimgs:
        print("err: split file not found")
        return
    targetimgs = []
    for img_path in baseimgs:
        lbl_path = Path(str(img_path).replace('images', 'labels')).with_suffix('.txt')
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                if any(line.startswith('1') for line in f):
                    targetimgs.append(img_path)
    currentimgs = baseimgs[:]
    for i in range(iter_count):
        print(f"\niteration {i+1}")
        newimgs = create_augmented_dataset(targetimgs, aug_count, aug_count * i)
        currentimgs.extend(newimgs)
        with open(f"train_{10000 + aug_count*(i+1)}_split.txt", "w") as f:
            for p in currentimgs: f.write(p + "\n")
    print("\ndone!")
            


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    create_several_augs('train_split.txt')