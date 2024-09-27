import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os
import shutil
"""
该代码构造数据前需要先运行构造目标检测构造数据的流程，然后从对应的目标检测数据的目录里去获取对应的数据
process_data/get_animal_image_transform_train_test_data.py，process_data/get_human_image_data.py，process_data/get_no_target_image_data.py
"""


# 定义拷贝函数
def copy_file(file_path, destination_dir):
    # 获取文件名
    file_name = os.path.basename(file_path)
    file_name_new = ""
    path_list = file_name.split(".")
    # 对以jpg进行下正规化
    if len(path_list) == 1:
        file_name_new = file_name + ".jpg"
    else:
        file_name_new = ".".join(path_list[:-1]) + ".jpg"
    # 目标路径
    dest_path = os.path.join(destination_dir, file_name_new)
    # 拷贝文件
    shutil.copy(file_path, dest_path)
    return file_name + " copied to " + destination_dir


# 获取源目录下所有文件的路径
def get_all_files(src_dir):
    return [os.path.join(src_dir, file) for file in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, file))]


# 并行拷贝函数
def parallel_copy(src_dir, dest_dir, desc, max_workers=4, batch_size=100):
    # 获取源目录中的所有文件
    files = get_all_files(src_dir)
    total_files = len(files)

    # 使用 ProcessPoolExecutor 并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 使用 tqdm 显示进度条
        with tqdm(total=total_files, desc=f"Copying {desc} images", unit="file") as pbar:
            # 批量处理文件
            for i in range(0, total_files, batch_size):
                # 提交一批任务
                batch_files = files[i:i + batch_size]
                results = list(executor.map(copy_file, batch_files, [dest_dir] * len(batch_files)))
                # 批量更新进度条
                pbar.update(len(results))


def get_train_test_images(images_dir, target_dir, sub_dir_name, max_workers=4):
    train_image_dir = os.path.join(images_dir, 'train', 'images')
    test_image_dir = os.path.join(images_dir, 'test', "images")
    if not os.path.exists(train_image_dir):
        print(f"train_image_dir {train_image_dir} 不存在")
        return
    if not os.path.exists(test_image_dir):
        print(f"test_image_dir {test_image_dir} 不存在")
        return
    target_train_dir = os.path.join(target_dir, 'train', sub_dir_name)
    if os.path.exists(target_train_dir):
        # 如果存在，删除目录
        shutil.rmtree(target_train_dir)
        print(f"目录 {target_train_dir} 已删除")
    # 创建目录
    os.makedirs(target_train_dir)
    print(f"目录 {target_train_dir} 已创建")
    target_test_dir = os.path.join(target_dir, 'test', sub_dir_name)
    if os.path.exists(target_test_dir):
        # 如果存在，删除目录
        shutil.rmtree(target_test_dir)
        print(f"目录 {target_test_dir} 已删除")
    # 创建目录
    os.makedirs(target_test_dir)
    print(f"目录 {target_test_dir} 已创建")
    parallel_copy(train_image_dir, target_train_dir, sub_dir_name + " train", max_workers)
    parallel_copy(test_image_dir, target_test_dir, sub_dir_name + " test", max_workers)


def main(animal_directory, human_directory, no_target_directory, target_dir, max_workers=4):
    get_train_test_images(animal_directory, target_dir, "animal", max_workers)
    get_train_test_images(human_directory, target_dir, "human", max_workers)
    get_train_test_images(no_target_directory, target_dir, "no_target", max_workers)


if __name__ == "__main__":
    """
    照片分类数据处理代码
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--animal_dir', type=str,
                        default="/mnt/data2/all_species_240916/animal_action", help='动物照片下载的目录')
    parser.add_argument('--human_dir', type=str,
                        default="/mnt/data2/all_species_240916/human_action", help='人类照片下载的目录')
    parser.add_argument('--no_target_dir', type=str,
                        default="/mnt/data2/all_species_240916/no_target", help='没有动物和人类的照片目录')
    parser.add_argument('--target_dir', type=str,
                        default="/mnt/data2/image_classifier_data", help='物种分类的数据目录')
    parser.add_argument('--max_workers', type=int,
                        default=6, help='多进程的进程数量')
    args = parser.parse_args()
    main(args.animal_dir, args.human_dir, args.no_target_dir, args.target_dir, args.max_workers)
