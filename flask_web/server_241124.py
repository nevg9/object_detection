from flask import Flask, jsonify, request
import torch
from PIL import Image
from torchvision import models, transforms
import torch.nn.functional as F
import io
import cv2
from ultralytics import YOLO
from ultralytics.nn.tasks import torch_safe_load, ClassificationModel
import torch
from safetensors.torch import load_file
from ultralytics.data.dataset import classify_transforms
from PIL import Image
import sys
from pathlib import Path
import numpy as np
import torch
import json
from utils.torch_utils import select_device, smart_inference_mode
import time
import logging
from logging.handlers import TimedRotatingFileHandler
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # 指定使用 GPU 0

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


app = Flask(__name__)
# 配置日志记录
log_directory = 'logs_241124'
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


def load_yolo_model(path, nc):
    ckpt, w = torch_safe_load(path)
    model = ckpt["model"]
    ClassificationModel.reshape_outputs(model, nc)
    for p in model.parameters():
        p.requires_grad = False  # for training
    model = model.float()
    return model


app.config["device"] = 0
model_cls_ori_path = "/mnt/data1/model_1122/image_classifier/yolov8m-cls.pt"
checkpoint_path = "/mnt/data1/model_1122/image_classifier/epoch_30/model.safetensors"
model = load_yolo_model(model_cls_ori_path, 3)
weights = load_file(checkpoint_path)
model.load_state_dict(weights)
model.to(app.config["device"])
app.config['classifier_model'] = model
app.config["torch_transforms"] = classify_transforms(size=224)
app.config['classifier_detail'] = {0: '动物', 1: '人类', 2: '无目标'}

app.config['detect_model'] = YOLO("/mnt/data1/model_1122/image_detection/best.engine", task="detect")
app.config['gender_set'] = {"雄性", "雌性"}
app.config['age_set'] = {'成体', '亚成体', "幼体"}


def process_image_classifier_image(image_binary, torch_transforms, device=0):
    try:
        nparr = np.frombuffer(image_binary, np.uint8)
        im = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 转换为PIL图像 (RGB格式)
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        # 应用转换
        sample = torch_transforms(im).to(device)
    except Exception as e:
        print(f"Error decoding image: {str(e)}")
        return None
    return sample


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
    processed_image = process_image_classifier_image(binary_data, app.config["torch_transforms"], app.config["device"])
    if processed_image is None:
        response_data = {}
        if resource_id:
            response_data['resourceId'] = resource_id
        if media_type:
            response_data['mediaType'] = media_type
        response_data["error-reason"] = "image is corrupt"
        return jsonify(response_data)
    processed_image = processed_image.unsqueeze(0)
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
    if response_data['label'] in app.config['classifier_detail']:
        response_data["desc"] = app.config['classifier_detail'][response_data['label']]
    if resource_id:
        response_data['resourceId'] = resource_id
    if media_type:
        response_data['mediaType'] = media_type
    return jsonify(response_data)


def process_predict(results):
    gender_set = app.config['gender_set']
    age_set = app.config['age_set']
    detect_class = []
    agg_detect_class = []
    classs_id_2_num = {}
    class_id_2_detail_dict = {}
    for result in results:
        # 获取边界框、置信度和类别
        boxes = result.boxes
        # 遍历每个检测框
        for box in boxes:
            # 获取边界框坐标 (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # xyxy格式
            # 获取置信度分数
            confidence = box.conf[0].item()
            # 获取预测类别的索引
            class_id = int(box.cls[0].item())
            if class_id not in classs_id_2_num:
                classs_id_2_num[class_id] = 0
            classs_id_2_num[class_id] += 1
            # 获取类别名称（如果模型中有类别名称映射）
            class_name = result.names[class_id]

            class_name_copy = class_name
            gender = "无法区分"
            for gd in gender_set:
                if gd in class_name_copy:
                    class_name_copy = class_name_copy.replace(gd, "")
                    gender = gd
            age = "无法区分"
            for ag in age_set:
                if ag in class_name_copy:
                    class_name_copy = class_name_copy.replace(ag, "")
                    age = ag
            class_id_2_detail_dict[class_id] = [class_name_copy, gender, age]
            detect_class.append((class_name, (x1, y1, x2, y2), confidence, [class_name_copy, gender, age]))
            # print(f"检测到 {class_name} - 置信度: {confidence:.2f}")
            # print(f"位置: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")
    for k, v in classs_id_2_num.items():
        agg_detect_class.append((class_id_2_detail_dict[k], v))

    return {'results': detect_class, 'agg_results': agg_detect_class}


@app.route('/predict_detect', methods=['POST'])
def predict_detect():
    start_time = time.time()  # 开始时间
    # 获取二进制图片数据
    binary_data = request.data
    resource_id = request.headers.get('resourceId')
    media_type = request.headers.get('mediaType')
    image_content = Image.open(io.BytesIO(binary_data))
    results = app.config['detect_model'].predict(image_content, conf=0.256)
    # 获取已加载的模型
    results_dict = process_predict(results)
    end_time = time.time()  # 结束时间
    processing_time = (end_time - start_time) * 1000  # 计算耗时（单位：毫秒）
    app.logger.info(f"detect processing time: {processing_time} ms")
    if resource_id:
        results_dict['resourceId'] = resource_id
    if media_type:
        results_dict['mediaType'] = media_type
    return jsonify(results_dict)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=False)
