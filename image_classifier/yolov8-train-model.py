from ultralytics import YOLO

# Load a model
# load a pretrained model (recommended for training)
model = YOLO("/home/yuzhong/data1/models/yolo/yolov8m-cls.pt")

# Train the model
results = model.train(data="/home/yuzhong/data1/image_classifier_data",
                      epochs=100, imgsz=480, batch=64,
                      save_period=2, cache=False, workers=24, project="/home/yuzhong/data1/train_model/yolo_v8_m_classifier_model",
                      name="default_model", exist_ok=True, dropout=0.1, label_smoothing=0.1, plots=True, device=[0, 1, 2, 3, 4, 5, 6, 7])
