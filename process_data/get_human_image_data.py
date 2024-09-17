import random
import os
import shutil
import argparse
from process_image_data_tools import json_contents_to_path, read_images_json_file


def get_image_train_test_data_to_local_dir(image_json_actions, image_json_name, fs_dir, root_dir,
                                           test_ratio=0.1, max_workers=3):
    os.makedirs(root_dir, exist_ok=True)
    train_dir = os.path.join(root_dir, image_json_name, "train")
    test_dir = os.path.join(root_dir, image_json_name, "test")
    root_dir = os.path.join(root_dir, image_json_name)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(train_dir + "/images", exist_ok=True)
    os.makedirs(train_dir + "/labels", exist_ok=True)
    os.makedirs(test_dir + "/images", exist_ok=True)
    os.makedirs(test_dir + "/labels", exist_ok=True)
    image_id_set = {item["图片id"] for item in image_json_actions}
    image_id_list = list(image_id_set)
    random.shuffle(image_id_list)
    test_image_id_list = image_id_list[:int(len(image_id_list) * test_ratio)]
    train_image_id_list = image_id_list[int(len(image_id_list) * test_ratio):]
    print("测试集图片数量：{}, 训练集图片数量：{}.".format(len(test_image_id_list), len(train_image_id_list)))
    json_contents_to_path(image_json_actions, root_dir, test_image_id_list,
                          train_image_id_list, train_dir, test_dir, fs_dir, max_workers)


def main(file_name, root_dir, dir_name, fs_dir, max_workers=3, test_ratio=0.2):
    # 查看照片中所有的年龄和性别的标注情况
    # 查看数据所有的年龄和性别标注情况
    human_action = read_images_json_file(file_name)
    human_image_download_dir = os.path.join(root_dir, dir_name)
    print(f"human image download dir: {human_image_download_dir}")
    # 检查目录是否存在
    if os.path.exists(human_image_download_dir):
        # 如果存在，删除目录
        shutil.rmtree(human_image_download_dir)
        print(f"目录 {human_image_download_dir} 已删除")
    # 创建目录
    os.makedirs(human_image_download_dir)
    print(f"目录 {human_image_download_dir} 已创建")
    # 拉取 对应的图片数据
    get_image_train_test_data_to_local_dir(human_action, dir_name, fs_dir, root_dir, test_ratio, max_workers)


if __name__ == '__main__':
    """
    从动物的json数据中，把照片和标签信息提取出来，并把图片复制到本地目录中
    输入：动物json数据，dump的根目录，以及dump到目录的名字，网盘挂载的路径
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json_name', type=str,
                        default="/home/yuzhong/nndata/export/20240428/人为活动1.json", help='下载的图片的json文件路径')
    parser.add_argument('-r', '--root_dir', type=str,
                        default="/mnt/data2/all_species_240916", help='该数据下载的根目录名字')
    parser.add_argument('-d', '--dir', type=str,
                        default="human_action", help='该次下载的数据的目录名字')
    parser.add_argument('-f', '--fs', type=str,
                        default="/home/yuzhong/nndata/fs", help='网盘上的默认路径位置')
    parser.add_argument('-w', '--workers', type=int,
                        default=2, help='下载数据时候的进程数量')
    parser.add_argument('-t', '--test_ratio', type=float,
                        default=0.2, help='下载数据时候的进程数量')
    args = parser.parse_args()
    main(args.json_name, args.root_dir, args.dir, args.fs, args.workers, args.test_ratio)
