# %%
from skimage import io
from ultralytics.data.utils import exif_size, IMG_FORMATS, FORMATS_HELP_MSG
from pathlib import Path
from tqdm import tqdm
import cv2


# %%
def detect_image_corrupt(img_file):
    # try:
    #     _ = io.imread(img_file)
    # except Exception as e:
    #     print(f"{img_file} is corrupt, {e}")
    cv_img = cv2.imread(img_file)
    cv2.imwrite(img_file, cv_img)
    if cv_img is None:
        print(f"{img_file} is corrupt")


# %%
image_files = []
dir_path = "/home/yuzhong/data1/image_classifier_data/train/human"
for ext in IMG_FORMATS:
    image_files.extend(Path(dir_path).rglob(f'*.{ext}'))
print(f"total images: {len(image_files)}")
total_deleted = 0
total_images = len(image_files)

# %%
for im_file in tqdm(image_files, total=total_images, desc='Checking images', unit=' image'):
    detect_image_corrupt(im_file)
