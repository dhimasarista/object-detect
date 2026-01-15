from pathlib import Path

BASE = Path("dataset")
SPLITS = ["train", "val", "test"]
CLASSES = ["metal", "plastik", "muka"]

for split in SPLITS:
    print(f"\n[{split.upper()}]")
    for cls in CLASSES:
        count = len(list((BASE / split / cls).glob("*")))
        print(f"{cls:8s}: {count}")
