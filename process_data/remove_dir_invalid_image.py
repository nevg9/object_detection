import argparse
from verify_image_and_remove_image_and_label import remove_corrupt_images_multiprocess

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', type=str,
                    required=True, help='下载的图片的json文件路径')
args = parser.parse_args()

remove_corrupt_images_multiprocess(args.dir, batch_size=1000, max_workers=32)
