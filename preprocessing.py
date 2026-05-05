import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_SIZE = (64, 64)
DATA_DIRS = {
    "fruits": "data/fruits",
    "vegetables": "data/vegetables",
}

def load_images():
    images = []
    labels = []
    class_names = []

    for folder in DATA_DIRS.values():
        for class_name in sorted(os.listdir(folder)):
            class_path = os.path.join(folder, class_name)
            if not os.path.isdir(class_path):
                continue
            label = len(class_names)
            class_names.append(class_name)
            for fname in sorted(os.listdir(class_path)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img_path = os.path.join(class_path, fname)
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(IMG_SIZE)
                    images.append(np.array(img))
                    labels.append(label)
                except Exception:
                    continue

    return np.array(images), np.array(labels), class_names

