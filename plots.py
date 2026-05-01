import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

FRUITS_ROOT = "data/fruits"
VEGS_ROOT = "data/vegetables"

FRESH_FRUIT_CLASSES = [
    "FreshApple",
    "FreshBanana",
    "FreshMango",
    "FreshOrange",
    "FreshStrawberry",
]
FRESH_VEG_CLASSES = [
    "FreshBellpepper",
    "FreshCarrot",
    "FreshCucumber",
    "FreshPotato",
    "FreshTomato",
]

CLASSES = FRESH_FRUIT_CLASSES + FRESH_VEG_CLASSES

CLASS_ROOTS = {cls: FRUITS_ROOT for cls in FRESH_FRUIT_CLASSES}
CLASS_ROOTS.update({cls: VEGS_ROOT for cls in FRESH_VEG_CLASSES})

LABELS = [
    "Apple",
    "Banana",
    "Mango",
    "Orange",
    "Strawberry",
    "Bell pepper",
    "Carrot",
    "Cucumber",
    "Potato",
    "Tomato",
]

CLASS_COLORS = [
    "#A52525",  # apple       - red
    "#f0c040",  # banana      - yellow
    "#833900",  # mango       - orange
    "#ffac82",  # orange      - orange
    "#ffb4bf",  # strawberry  - dark red
    "#ff0000",  # bell pepper - purple/pink
    "#ff6200",  # carrot      - orange-red
    "#4caf50",  # cucumber    - green
    "#443b29",  # potato      - tan
    "#640000",  # tomato      - red
]

IMG_SIZE = (64, 64)  # resize target for feature extraction
MAX_IMAGES = 100  # images per class used for PCA (keeps runtime fast)
SAVE_DIR = "."  # where to save the output figures
DPI = 150

def load_images_from_class(class_path, max_n=None):
    files = [
        f
        for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if max_n is not None:
        files = files[:max_n]
    imgs = []
    for f in files:
        try:
            img = Image.open(os.path.join(class_path, f)).convert("RGB")
            img = img.resize(IMG_SIZE)
            imgs.append(np.array(img))
        except Exception:
            pass
    return imgs


def flatten_to_feature_vector(imgs):
    return np.stack([img.astype(np.float32).flatten() / 255.0 for img in imgs])



all_images = {}
for cls in CLASSES:
    cls_path = os.path.join(CLASS_ROOTS[cls], cls)
    if not os.path.isdir(cls_path):
        raise FileNotFoundError(
            f"Folder not found: {cls_path}\n"
            f"Make sure you're running this script from the AppliedML_project root."
        )
    all_images[cls] = load_images_from_class(cls_path)

counts = [len(all_images[c]) for c in CLASSES]


# PLOT 1: CLASS DISTRIBUTION 

fig, ax = plt.subplots(figsize=(10, 4.5))
bars = ax.bar(LABELS, counts, color=CLASS_COLORS, edgecolor="white", linewidth=0.8)
ax.set_title(
    "Class distribution – fresh fruits and vegetables", fontsize=13, fontweight="bold"
)
ax.set_xlabel("Class")
ax.set_ylabel("Number of images")
ax.set_ylim(0, max(counts) * 1.18)
for bar, count in zip(bars, counts):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 5,
        str(count),
        ha="center",
        va="bottom",
        fontsize=10,
    )
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "plot1_class_distribution.png"), dpi=DPI)
plt.close()


# PLOT 2: MEAN RGB CHANNEL INTENSITY PER CLASS 

channel_names = ["Red", "Green", "Blue"]
channel_colors = ["#e05252", "#4caf50", "#4a90d9"]

mean_channels = []
for cls in CLASSES:
    stack = np.stack(all_images[cls])  # (N, H, W, 3)
    means = stack.mean(axis=(0, 1, 2))  # mean over N, H, W → shape (3,)
    mean_channels.append(means)
mean_channels = np.array(mean_channels)  # (10, 3)

x = np.arange(len(CLASSES))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 5))
for i, (ch_name, ch_color) in enumerate(zip(channel_names, channel_colors)):
    ax.bar(
        x + (i - 1) * width,
        mean_channels[:, i],
        width,
        label=ch_name,
        color=ch_color,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.88,
    )

ax.set_title("Mean RGB channel intensity per class", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(LABELS, rotation=15, ha="right")
ax.set_ylabel("Mean pixel intensity (0–255)")
ax.set_ylim(0, 300)
ax.legend(title="Channel")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "plot2_mean_rgb_per_class.png"), dpi=DPI)
plt.close()


# PLOT 3: PCA 

pca_features = []
pca_labels = []

for cls in CLASSES:
    imgs_subset = load_images_from_class(
        os.path.join(CLASS_ROOTS[cls], cls), max_n=MAX_IMAGES
    )
    X = flatten_to_feature_vector(imgs_subset)
    pca_features.append(X)
    pca_labels.extend([cls] * len(imgs_subset))

X_all = np.vstack(pca_features)
y_all = np.array(pca_labels)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_all)
var_exp = pca.explained_variance_ratio_ * 100

fig, ax = plt.subplots(figsize=(8, 6))
patches = []
for cls, label, color in zip(CLASSES, LABELS, CLASS_COLORS):
    mask = y_all == cls
    ax.scatter(
        X_pca[mask, 0], X_pca[mask, 1], s=14, alpha=0.55, color=color, edgecolors="none"
    )
    patches.append(mpatches.Patch(color=color, label=label))

ax.set_title(
    "PCA projection of fresh fruit and vegetable images\n(RGB pixels, 64×64)",
    fontsize=12,
    fontweight="bold",
)
ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance explained)")
ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance explained)")
ax.legend(
    handles=patches,
    title="Class",
    framealpha=0.7,
    fontsize=9,
    bbox_to_anchor=(1.01, 1),
    loc="upper left",
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "plot3_pca.png"), dpi=DPI, bbox_inches="tight")
plt.close()