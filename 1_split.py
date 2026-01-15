import os, shutil, random
from pathlib import Path

SRC = "raw_dataset"
DST = "dataset"
CLASSES = ["metal", "plastik", "muka"]

SPLIT = {
    "train": 70,
    "val": 10,
    "test": 20
}

random.seed(42)

for cls in CLASSES:
    imgs = list(Path(SRC, cls).glob("*"))
    random.shuffle(imgs)

    idx = 0
    for split, n in SPLIT.items():
        out = Path(DST, split, cls)
        out.mkdir(parents=True, exist_ok=True)

        for img in imgs[idx:idx+n]:
            shutil.copy(img, out / img.name)

        idx += n
