import cv2
from utils import get_train_transform, get_eval_transform
import os

img_path = os.path.join("dataset/train/metal",
                        os.listdir("dataset/train/metal")[0])

img = cv2.imread(img_path)
assert img is not None, "Gagal load gambar"

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

train_tf = get_train_transform()
eval_tf  = get_eval_transform()

x_train = train_tf(img)
x_eval  = eval_tf(img)

print("TRAIN:", x_train.shape, x_train.min().item(), x_train.max().item())
print("EVAL :", x_eval.shape, x_eval.min().item(), x_eval.max().item())
