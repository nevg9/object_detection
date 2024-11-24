from flask import Flask, jsonify, request
import torch
from PIL import Image
from torchvision import models, transforms
import torch.nn.functional as F
import io
import cv2
import os
import sys
from pathlib import Path
import numpy as np
import torch
import json
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode
import time
import logging
from logging.handlers import TimedRotatingFileHandler


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


app = Flask(__name__)
# 配置日志记录
log_directory = 'logs'
os.makedirs(log_directory, exist_ok=True)  # 创建日志目录（如果不存在）
log_handler = TimedRotatingFileHandler(
    os.path.join(log_directory, 'app.log'),  # 日志文件名
    when='D',  # 'D'表示每天滚动日志文件
    interval=1,  # 滚动周期（天）
    backupCount=7  # 保留的旧日志文件数量
)
log_handler.setLevel(logging.INFO)
log_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
app.logger.addHandler(log_handler)  # 添加日志处理程序
app.logger.setLevel(logging.INFO)
app.logger.info('Flask application started')  # 启动时记录一条日志


def load_detect_mode(device: str, weights: str, data: str, imgsz=(640, 640)):
    device = select_device(device)
    model = DetectMultiBackend(
        weights, device=device, dnn=False, data=data, fp16=False)
    stride = model.stride
    imgsz = check_img_size(imgsz, s=stride)
    return model, imgsz, device


def read_name_to_detail(file_name):
    data_dict = {}
    for line in open(file_name, 'r'):
        line = line.strip()
        if line == "":
            continue
        data = json.loads(line)
        for k, v in data.items():
            data_dict[float(k)] = v
    return data_dict


weights_detection = "/mnt/data1/model_0925/image_detection/species_43_model/best.onnx"
data_file = "/mnt/data1/model_0925/image_detection/species_43_model/species_43.yaml"
name_detail_file = "/mnt/data1/model_0925/image_detection/species_43_model/species_43_class_gender_age.json"
imgsz = [640, 640]
app.config['detect_model'], app.config['detect_imgsz'], app.config['detect_device'] = load_detect_mode(device=1,
                                                                                                       weights=weights_detection,
                                                                                                       data=data_file,
                                                                                                       imgsz=imgsz)
app.config['detect_class_detail_dict'] = read_name_to_detail(name_detail_file)


def load_classifier_model():
    """加载模型并返回"""
    return torch.jit.load("/mnt/data1/model_0925/image_classifier/image_classifier_model.pt")


app.config['classifier_model'] = load_classifier_model()


def process_image_classifier_image(binary_data):
    # 定义均值和标准差，用于归一化
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # 创建转换器
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=mean, std=std)
    # 将二进制数据转换为PIL图像
    image = Image.open(io.BytesIO(binary_data))
    # 对图像进行预处理
    image = image.resize((256, 256))
    tensor_image = to_tensor(image)
    normalized_image = normalize(tensor_image)
    # 返回处理后的图像张量
    return normalized_image.unsqueeze(0).to(0)  # 添加batch维度


def binary_to_jpg(binary_data, output_filename='output.jpg'):
    """
    将二进制图片数据转换为 JPG 图片并保存到文件。
    参数:
    - binary_data: 包含图片数据的二进制字符串。
    - output_filename: 输出的 JPG 文件名，默认为 'output.jpg'。
    """
    # 创建一个 BytesIO 对象，它可以像文件对象一样使用
    image_stream = io.BytesIO(binary_data)
    # 使用 PIL 的 Image.open() 方法打开字节流
    image = Image.open(image_stream)
    # 将图片保存为 JPG 格式
    image.save(output_filename, 'JPEG')


@app.route('/predict_classifier', methods=['POST'])
def predict_classifier():
    start_time = time.time()  # 开始时间
    # 获取二进制图片数据
    binary_data = request.data
    resource_id = request.headers.get('resourceId')
    media_type = request.headers.get('mediaType')
    # binary_to_jpg(binary_data)
    # 获取已加载的模型
    model = app.config['classifier_model']
    processed_image = process_image_classifier_image(binary_data)
    process_image_end_time = time.time()
    image_time = (process_image_end_time - start_time) * 1000
    # 使用模型进行预测
    predict_value = model(processed_image)

    max_prob, predicted_class = torch.max(
        F.softmax(predict_value, dim=1), dim=1)
    end_time = time.time()  # 开始时间
    processing_time = (end_time - start_time) * 1000  # 计算耗时（单位：毫秒）
    app.logger.info(f"classifier processing time: {processing_time} ms, process image time: {image_time} ms")
    # 构建返回数据
    response_data = {
        'prediction': max_prob.item(),
        'label': predicted_class.item()
    }
    if resource_id:
        response_data['resourceId'] = resource_id
    if media_type:
        response_data['mediaType'] = media_type
    return jsonify(response_data)


def transform_images(im0, img_size, stride, auto):
    assert im0 is not None, 'Image is None'
    im = letterbox(im0, img_size, stride=stride, auto=auto)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    return im


def process_predict(pred, im0, im, names, class_detail_dict):
    # line_thickness = 5
    results = []
    integer_classs_num = {}
    for i, det in enumerate(pred):  # per image
        # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            # annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            for *xyxy, conf, cls in reversed(det):
                c = cls.item()  # integer class
                if c not in integer_classs_num:
                    integer_classs_num[c] = 0
                integer_classs_num[c] += 1
                label = names[c]
                x1, y1, x2, y2 = xyxy
                results.append((label, (x1.item(), y1.item(), x2.item(), y2.item()), conf.item(), class_detail_dict[c]))

    agg_results = []
    for k, v in integer_classs_num.items():
        agg_results.append((class_detail_dict[k], v))

    results_dict = {'results': results, 'agg_results': agg_results}
    return results_dict


def process_image_to_detect(binary_data):
    # 将二进制数据转换为 numpy 数组
    nparr = np.frombuffer(binary_data, np.uint8)

    # 使用 cv2.imdecode 将其解码为图像
    im0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    model = app.config['detect_model']
    imgsz = app.config['detect_imgsz']
    im = transform_images(im0, imgsz, model.stride, model.pt)
    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    pred = model(im, augment=False, visualize=False)
    conf_thres = 0.25
    iou_thres = 0.45
    max_det = 1000
    pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)
    results_dict = process_predict(pred, im0, im, model.names, app.config['detect_class_detail_dict'])
    return results_dict


@app.route('/predict_detect', methods=['POST'])
def predict_detect():
    start_time = time.time()  # 开始时间
    # 获取二进制图片数据
    binary_data = request.data
    resource_id = request.headers.get('resourceId')
    media_type = request.headers.get('mediaType')
    # 获取已加载的模型
    results_dict = process_image_to_detect(binary_data)
    end_time = time.time()  # 结束时间
    processing_time = (end_time - start_time) * 1000  # 计算耗时（单位：毫秒）
    app.logger.info(f"detect processing time: {processing_time} ms")
    if resource_id:
        results_dict['resourceId'] = resource_id
    if media_type:
        results_dict['mediaType'] = media_type
    return jsonify(results_dict)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
