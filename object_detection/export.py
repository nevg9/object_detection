from ultralytics import YOLO

# 加载训练好的权重文件
model = YOLO("all_species_240916/default_args/weights/best.pt")
# 导出位tensorRT格式
model.export(format="engine")