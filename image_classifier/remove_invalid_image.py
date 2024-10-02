import sys
sys.path.append("..")
from process_data.verify_image_and_remove_image_and_label import remove_corrupt_images_multiprocess

remove_corrupt_images_multiprocess("/home/yuzhong/data1/image_classifier_data", batch_size=1000, max_workers=32)
