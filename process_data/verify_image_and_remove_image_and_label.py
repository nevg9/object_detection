from PIL import Image, ImageOps, ImageFile
from ultralytics.data.utils import exif_size, IMG_FORMATS, FORMATS_HELP_MSG
import os
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None  # 清空缓存，防止旧图像缓存
# 获取当前文件名（不带扩展名）
file_name = os.path.splitext(os.path.basename(__file__))[0]
# 设置日志文件名
log_file_name = f'{file_name}.log'

# 配置日志记录
logging.basicConfig(
    filename=log_file_name,   # 指定日志文件名
    filemode='a',             # 文件模式，'a'为追加模式
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    level=logging.INFO       # 设置日志级别
)
logger = logging.getLogger(__name__)


def fix_corrupt_jpeg(im_file):
    """尝试修复损坏的JPEG文件"""
    try:
        with open(im_file, "rb") as f:
            f.seek(-2, 2)
            if f.read() != b"\xff\xd9":  # corrupt JPEG
                logger.warning(f"发现损坏的JPEG: {im_file}，尝试修复")
                ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                logger.info(f"警告 ⚠️ {im_file}: 已修复并保存损坏的JPEG")
        return True
    except Exception as e:
        logger.error(f"修复JPEG时出错: {im_file}, 错误: {e}")
        return False


def verify_image_and_remove_image(im_file):
    """
    检查图片是否损坏，如果损坏就删除图片
    返回删除的图片数量(0或1)
    """
    if not os.path.exists(im_file):
        logger.warning(f"文件不存在: {im_file}")
        return 0
    try:
        im = Image.open(im_file)
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"Invalid image format {im.format}. {FORMATS_HELP_MSG}"
        if im.format.lower() in {"jpg", "jpeg"}:
            fix_res = fix_corrupt_jpeg(im_file)
            if not fix_res:
                os.remove(im_file)
                return 1
            else:
                im.close()
        im = Image.open(im_file)
        # 使用PIL验证
        im.verify()
        im.close()
        cv_img = cv2.imread(im_file)
        cv2.imwrite(im_file, cv_img)
        if cv_img is None:
            raise ValueError(f"opencv无法读取图像文件: {im_file}")
    except Exception as e:
        os.remove(im_file)
        msg = f"ERROR {im_file}: ignoring corrupt image, removed!!!, exception: {e}"
        logger.info(msg)
        return 1
    return 0


def remove_corrupt_images(dir_path, batch_size=100):
    # 获取所有图片文件的路径列表
    image_files = []
    for ext in IMG_FORMATS:
        image_files.extend(Path(dir_path).rglob(f'*.{ext}'))
    print(f"total images: {len(image_files)}")
    total_deleted = 0
    total_images = len(image_files)
    for i in range(0, total_images, batch_size):
        batch = image_files[i:i + batch_size]
        deleted_counts = [verify_image_and_remove_image(img) for img in batch]
        total_deleted += sum(deleted_counts)
    print(f"Total deleted images: {total_deleted}")  # 打印删除的图片数量


def remove_corrupt_images_multiprocess(dir_path, batch_size=100, max_workers=4):
    # 获取所有图片文件的路径列表
    image_files = []
    for ext in IMG_FORMATS:
        image_files.extend(Path(dir_path).rglob(f'*.{ext}'))
    print(f"total images: {len(image_files)}")
    total_deleted = 0
    total_images = len(image_files)
    with tqdm(total=total_images, desc=f"Processing {dir_path} images") as pbar:
        # 创建进程池
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 按批次提交图片文件
            for i in range(0, total_images, batch_size):
                batch = image_files[i:i + batch_size]
                futures = [executor.submit(verify_image_and_remove_image, img) for img in batch]
                # 逐个等待批次任务完成，并更新进度条
                for future in as_completed(futures):
                    deleted_count = future.result()
                    total_deleted += deleted_count  # 累加删除数量
                    pbar.update(1)
    logger.info(f"Total deleted images: {total_deleted}")  # 打印删除的图片数量
