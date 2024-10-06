# 图像分类

## yolo 图像分类模型

### 数据构造

数据构造[参考链接](https://docs.ultralytics.com/datasets/classify/#what-datasets-are-supported-by-ultralytics-yolo-for-image-classification)

参考数据构造代码：image_classifier/get_classifier_data.py

### 模型训练

[参考链接](https://docs.ultralytics.com/tasks/classify/)
[模型训练的一些参数](https://docs.ultralytics.com/modes/train/#train-settings)

### Accelerate 教程

[参考链接](https://huggingface.co/docs/accelerate/en/index)

#### accelerate使用配置文件启动的命令

accelerate launch --config_file {path/to/config/my_config_file.yaml} {script_name.py} {--arg1} {--arg2} ...
CUDA_VISIBLE_DEVICES="0" accelerate launch --config_file {path/to/config/my_config_file.yaml} {script_name.py} {--arg1} {--arg2} ...

