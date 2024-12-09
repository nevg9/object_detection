# 请求server的客户端代码，并把本地的图片文件以二进制的数据传输个python server
import requests
from PIL import Image, ImageOps, ImageFile
from ultralytics.data.utils import exif_size, IMG_FORMATS, FORMATS_HELP_MSG
import os
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True


def verify_image(im_file):
    """Verify one image."""
    # Number (found, corrupt), message
    im = Image.open(im_file)
    im.verify()  # PIL verify
    shape = exif_size(im)  # image size
    shape = (shape[1], shape[0])  # hw
    assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
    assert im.format.lower() in IMG_FORMATS, f"Invalid image format {im.format}. {FORMATS_HELP_MSG}"
    if im.format.lower() in {"jpg", "jpeg"}:
        with open(im_file, "rb") as f:
            f.seek(-2, 2)
            if f.read() != b"\xff\xd9":  # corrupt JPEG
                ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
    return im_file


def process_directory(directory):
    files_path = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            # 检查文件扩展名是否为支持的图片格式
            if file.split('.')[-1].lower() in IMG_FORMATS:
                files_path.append(file_path)
    return files_path


def send_image_to_server(image_path, url, resource_id="12341234", media_type="photo"):
    """
    将本地图片文件以二进制数据的形式传输到Python服务器。

    参数：
    image_path (str): 图片文件的本地路径。

    返回：
    requests.Response: 服务器的响应对象。
    """
    # 读取图片文件内容
    # image_path = verify_image(image_path)
    with open(image_path, 'rb') as file:
        image_content = file.read()
    # 定义请求头，包含resourceId和mediaType字段
    headers = {
        'Content-Type': 'application/octet-stream',
        'resourceId': resource_id,
        'mediaType': media_type
    }
    # 发送POST请求，将图片数据作为二进制数据传输
    response = requests.post(url, data=image_content, headers=headers)
    if response.status_code == 200:
        # 请求成功，尝试解析JSON数据
        try:
            response_data = response.json()
            # 访问解析后的数据字典中的键
            print(image_path, response_data)
        except ValueError:
            # 如果响应不是JSON格式，将引发ValueError
            print("Response content is not in JSON format.")
    else:
        # 请求失败，打印状态码和原因
        print(image_path, f"Request failed with status code {response.status_code}: {response.reason}")


if __name__ == '__main__':
    # image_path = '/home/yuzhong/data1/yuzhong/image_detection/images/SCNR001X-HX1703-20210814-5240.jpg'
    # image_path = '/home/yuzhong/data1/yuzhong/image_detection/images/HBNR009X-HP0006-20190703-00308.jpg'
    # image_path = '/home/yuzhong/data1/yuzhong/image_detection/images/SNNR005X-SNH036-20161016-00392.jpg'
    # image_path = '/home/yuzhong/nndata/yuzhong/photos/JXNR013X-DZS019-20230707-00664.jpg'
    # url = 'http://localhost:5002/predict_detect'
    # send_image_to_server(image_path, url)
    # images_path = process_directory('/home/yuzhong/nndata/yuzhong/photos')
    # 空白照片识别的url
    url = 'http://localhost:5002/predict_classifier'
    # 目标识别的url
    # url = 'http://localhost:5000/predict_detect'
    # send_image_to_server(image_path, url)
    # url = 'http://localhost:5002/predict_classifier'
    # 目标识别的url
    images_path = process_directory('/home/yuzhong/nndata/yuzhong/photos')
    url = 'http://localhost:5002/predict_detect'
    for image_path in tqdm(images_path, desc="预测图片"):
        send_image_to_server(image_path, url)
