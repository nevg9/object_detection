from ultralytics import YOLO

model = YOLO("yolo11m.yaml").load("./yolo11m.pt")  # build a new model from YAML

results = model.train(data="./all_species_240916.yaml", epochs=100, imgsz=640, batch=88, device=[0, 1, 2, 3, 4, 5, 6, 7],
                      save_period=4, workers=32, project="all_species_240916", name="default_args", exist_ok=True, verbose=True,
                      plots=True)
