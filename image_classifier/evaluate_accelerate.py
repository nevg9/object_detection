import argparse
from accelerate import Accelerator
import evaluate
import cv2
import torchvision  # scope for faster 'import ultralytics'
from torch.utils.data import DataLoader
from ultralytics.data.dataset import classify_transforms
from ultralytics.nn.tasks import torch_safe_load, ClassificationModel
import torch
from safetensors.torch import load_file
from tqdm import tqdm
from PIL import Image
from classification_dataset_our import ClassificationDatasetCustom
import csv


def get_predict_image(root, imgsz):
    base = torchvision.datasets.ImageFolder(root=root)
    samples = base.samples
    torch_transforms = classify_transforms(size=imgsz)
    for file_name, class_id in samples:
        im = cv2.imread(file_name)
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = torch_transforms(im)
        yield {"img": sample, "cls": class_id}


def load_yolo_model(path, nc):
    ckpt, w = torch_safe_load(path)
    model = ckpt["model"]
    ClassificationModel.reshape_outputs(model, nc)
    for p in model.parameters():
        p.requires_grad = False  # for training
    model = model.float()
    return model


def predict_images(model, image_paths, imgsz, device_id=0):
    model.eval()
    model.to(device_id)
    predict_class = []
    ground_truth_class = []
    with torch.no_grad():
        for im_info in tqdm(get_predict_image(image_paths, imgsz), desc="predict image"):
            im = im_info['img'].unsqueeze(0).to(device_id)
            class_id = im_info['cls']
            ground_truth_class.append(class_id)
            outputs = model(im)
            predictions = outputs.argmax(dim=-1)
            predict_class.append(predictions.item())
    return predict_class, ground_truth_class


def predict_batch_images(model, image_paths, imgsz, device_id=0, batch_size=32):
    model.eval()
    model.to(device_id)
    predict_class = []
    ground_truth_class = []

    batch_images = []
    batch_classes = []

    with torch.no_grad():
        for im_info in tqdm(get_predict_image(image_paths, imgsz), desc="predict image"):
            im = im_info['img']
            class_id = im_info['cls']

            batch_images.append(im)
            batch_classes.append(class_id)

            if len(batch_images) == batch_size:
                batch_tensor = torch.stack(batch_images).to(device_id)
                outputs = model(batch_tensor)
                predictions = outputs.argmax(dim=-1)

                # Store predictions and ground truth
                predict_class.extend(predictions.cpu().numpy().tolist())
                ground_truth_class.extend(batch_classes)

                # Clear the batch lists
                batch_images = []
                batch_classes = []

        # Process any remaining images in the batch (if total % batch_size != 0)
        if batch_images:
            batch_tensor = torch.stack(batch_images).to(device_id)
            outputs = model(batch_tensor)
            predictions = outputs.argmax(dim=-1)

            predict_class.extend(predictions.cpu().numpy().tolist())
            ground_truth_class.extend(batch_classes)

    return predict_class, ground_truth_class


def main(args):
    accelerator = Accelerator()
    with accelerator.main_process_first():
        test_dataset = ClassificationDatasetCustom(args.test_set, args)
    eval_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=32, prefetch_factor=2, pin_memory=True, drop_last=False
    )
    model = load_yolo_model(args.model_path, 3)
    weights = load_file(args.checkpoint_path)
    model.load_state_dict(weights)
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
    model.eval()
    predict_all_result = []
    references_all_result = []
    img_files_all_result = []
    metric = evaluate.load("accuracy")
    for _, batch in tqdm(enumerate(eval_dataloader), disable=not accelerator.is_local_main_process, desc="predicting"):
        with torch.no_grad():
            outputs = model(batch['img'])
        predictions = outputs.argmax(dim=-1)
        result = accelerator.gather_for_metrics((predictions, batch["cls"], batch['filename']))
        for i in range(0, len(result), 3):
            # accelerator.print(result[i:i + 3])
            predictions = result[i]
            references = result[i + 1]
            img_files = result[i + 2]
            # accelerator.print(f"predictions:{predictions}, references:{references}, img_files:{img_files}")
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
            predict_all_result.extend(predictions.cpu().numpy().tolist())
            references_all_result.extend(references.cpu().numpy().tolist())
            img_files_all_result.extend(img_files)
    eval_metric = metric.compute()
    accelerator.print(f"eval_metric:{eval_metric}")
    with open(args.evaluat_csv_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["label", "predict", "img_file"])
        for labeld, predict, imfile in zip(references_all_result, predict_all_result, img_files_all_result):
            writer.writerow([labeld, predict, imfile])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate arguments")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--test_set",
        type=str,
        default="/mnt/data1/model_1122/image_classifier/test",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/data1/model_1122/image_classifier/yolov8m-cls.pt",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default='/mnt/data1/model_1122/image_classifier/epoch_30/model.safetensors',
    )
    parser.add_argument(
        "--evaluat_csv_file",
        type=str,
        default='/mnt/data1/model_1122/image_classifier/test/evaluate_result.csv',
    )
    args = parser.parse_args()
    main(args)
