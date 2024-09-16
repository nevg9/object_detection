"""原始的标注数据转换成YOLOV5目标检测的标准格式
原始的数据存储格式介绍：
1. 原始数据的目标框的格式是4个元素的元祖分别为 左上x坐标，左上y坐标，右下x坐标，右下y坐标
2. 原始的坐标需要对其进行归一化到真实图片上对应的百分比位置即每个坐标需要除以65536
处理后的数据格式介绍：
1. 存储到指定的路径下：会按照指定的比例分为训练集和测试集path/train path/test
2. 对应的路径下会分为images和labels目录：
3. labels目录下会有对应images图片文件名去掉后缀后加上.txt后缀，例如images中有img.jpg,则对应的label目录下会有img.txt对应的标注数据
4. 标注数据的格式为：每个object各自为一行，标注框对应的内容为：classIndex,x_center,y_center,w,h（这些都是根据图片的宽高进行归一化的即都是0-1之间）
"""

import json
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil
import os
import yaml
import argparse


def transform_position(w, h, position_str, positions_list):
    """从原始数据的标注的框位置转化为真实图片标注框的位置，图片位置记录方式为 左上x坐标，左上y坐标，右下x坐标，右下y坐标"""
    position_str = position_str.strip("()")
    position_str_split = position_str.split("),(")
    for string in position_str_split:
        number_strings = string.split(",")
        try:
            numbers = [int(num) for num in number_strings]
        except Exception:
            return False
        if len(numbers) != 4:
            return False
        to_zero_possition = [i if i > 0 else 0 for i in numbers]
        new_possions = [y / 65536 * w if x % 2 == 0 else y /
                        65536 * h for x, y in enumerate(to_zero_possition)]
        if abs(new_possions[0] - new_possions[2]) < 2 or abs(new_possions[1] - new_possions[3]) < 2:
            continue

        positions_list.append(new_possions)
    if len(positions_list) == 0:
        return False
    return True


def parse_raw_data(file_name):
    new_image_datas = {}
    error_possition = set()
    error_image = set()
    total_image = set()
    for line in open(file_name):
        line = line.strip()
        image_data = json.loads(line)
        total_image.add(image_data["图片id"])
        try:
            image = Image.open(image_data["本地路径"])
        except Exception:
            print("error:", line)
            error_possition.add(image_data["图片id"])
            continue
        w, h = image.size
        # dict_data["图片id"] = image_data["图片id"]
        # dict_data["本地路径"] = image_data["本地路径"]
        object_data = {}
        object_data["物种名称"] = image_data["物种名称"]
        object_data["性别"] = image_data["性别"]
        object_data["年龄"] = image_data["年龄"]
        object_data["物种id"] = image_data["物种id"]
        positions_list = []
        if not transform_position(w, h, image_data["位置坐标"], positions_list):
            # print("error possition:", line)
            error_possition.add(image_data["图片id"])
            continue
        object_data["位置坐标"] = positions_list

        if image_data["图片id"] not in new_image_datas:
            new_dict = {}
            new_dict["图片id"] = image_data["图片id"]
            new_dict["图片高度"] = h
            new_dict["图片宽度"] = w
            new_dict["本地路径"] = image_data["本地路径"]
            new_dict["保护地id"] = image_data["保护地id"]
            new_dict["保护地名称"] = image_data["保护地名称"]
            new_dict["objects"] = [object_data]
            new_image_datas[image_data["图片id"]] = new_dict
        else:
            data_content = new_image_datas[image_data["图片id"]]
            data_content["objects"].append(object_data)

    print("图片总数：{}, 图片打开失败的数量为：{}, 图片中位置信息有问题的数量为：{}, 没问题的图片数量: {}.".format(
        len(total_image), len(error_image), len(error_possition), len(new_image_datas)))
    return new_image_datas


def read_class_names(filename):
    class_names = []
    for line in open(filename):
        name = line.strip()
        class_names.append(name)
    return class_names


def read_species_config(filename):
    f = open(filename, 'r')
    species_config = yaml.load(f,  Loader=yaml.FullLoader)
    return species_config


def transform_to_yolo_positions(positions_lists, w, h):
    """将图片框的 左上x坐标，左上y坐标，右下x坐标，右下y坐标格式，转换为yolo框标定的格式：x_center, y_center, width, height(归一化)"""
    new_positions = []
    for positions_list in positions_lists:
        x_center = (positions_list[2] + positions_list[0]) / 2
        y_center = (positions_list[3] + positions_list[1]) / 2
        width = positions_list[2] - positions_list[0]
        height = positions_list[3] - positions_list[1]

        # normalize
        x_center = x_center / w
        y_center = y_center / h
        width = width / w
        height = height / h
        new_positions.append([x_center, y_center, width, height])
    return new_positions


def generate_all_keys(object_data):
    """生成key的逻辑是年龄-性别-物种，年龄-物种，性别-物种，物种"""
    keys = set()
    age = object_data["年龄"] if object_data["年龄"] else "none"
    gender = object_data["性别"] if object_data["性别"] else "none"
    species = object_data["物种名称"] if object_data["物种名称"] else "none"
    keys.add(age + gender + species)
    keys.add(age + species)
    keys.add(gender + species)
    keys.add(species)
    return keys


def is_in_species(species_config, object_data):
    if object_data["物种名称"] in species_config:
        species_candidate_names_set = generate_all_keys(object_data)
        for x in species_config[object_data["物种名称"]]:
            if isinstance(x, dict):
                for inner_key in x.keys():
                    for inner_value in x[inner_key]:
                        if inner_value in species_candidate_names_set:
                            return True, inner_key
            else:
                if x in species_candidate_names_set:
                    return True, x
        if object_data["物种名称"] in species_config["同类物种"]:
            return True, species_config["同类物种"][object_data["物种名称"]]
        else:
            return True, object_data["物种名称"]

    return False, None


def species_config_to_class_names(species_config):
    """
    input:
        species_config 物种的配置信息
        train_image_info 训练数据中哪些类别是训练数据的
    output:
        根据配置信息和训练数据中有的类别信息，返回训练数据含有的对应的配置的类别class
    """
    class_names_set = set()
    class_names = []
    for k, v in species_config.items():
        if k == "同类物种":
            continue
        for x in v:
            if isinstance(x, dict):
                for item in x.keys():
                    if item not in class_names_set:
                        class_names_set.add(item)
                        class_names.append(item)
            else:
                if x not in class_names_set:
                    class_names_set.add(x)
                    class_names.append(x)
        if k in species_config["同类物种"]:
            if species_config["同类物种"][k] not in class_names_set:
                class_names_set.add(species_config["同类物种"][k])
                class_names.append(species_config["同类物种"][k])
        else:
            if k not in class_names_set:
                class_names_set.add(k)
                class_names.append(k)

    return class_names


def species_config_and_train_data_to_class_names(species_config, train_image_info):
    """
    input:
        species_config 物种的配置信息
        train_image_info 训练数据中哪些类别是训练数据的
    output:
        根据配置信息和训练数据中有的类别信息，返回训练数据含有的对应的配置的类别class
    """
    class_names_set = set()
    class_names = []
    for k, v in species_config.items():
        if k == "同类物种":
            continue
        for x in v:
            if isinstance(x, dict):
                for item in x.keys():
                    if item not in class_names_set and item in train_image_info:
                        class_names_set.add(item)
                        class_names.append(item)
            else:
                if x not in class_names_set and x in train_image_info:
                    class_names_set.add(x)
                    class_names.append(x)
        if k in species_config["同类物种"]:
            name = species_config["同类物种"][k]
            if name not in class_names_set and name in train_image_info:
                class_names_set.add(name)
                class_names.append(name)
        else:
            if k not in class_names_set and k in train_image_info:
                class_names_set.add(k)
                class_names.append(k)

    return class_names


def write_class_names(class_names, path, species_config_file):
    if not os.path.exists(path):
        os.makedirs(path)
    file = species_config_file.rsplit('.', 1)[0].rsplit('/', 1)[-1]
    f = open(path + "/" + file + ".txt", 'w')
    index = 0
    for name in class_names:
        f.write(str(index) + ": " + name + "\n")
        index += 1


def label_transform_yolo_format_and_split(species_config_file, image_datas, dataset_path, test_ratio=0.2):
    species_config = read_species_config(species_config_file)
    species_error_info = dict()
    train_image_info = dict()
    new_image_datas = {}
    for k, v in image_datas.items():
        new_objects = []
        for x in v["objects"]:
            flag, new_species = is_in_species(species_config, x)
            if flag:
                x["species"] = new_species
                new_objects.append(x)
                train_image_info[new_species] = train_image_info.setdefault(new_species, 0) + 1
            else:
                # 为了打印log查看有多少标注信息可能有问题
                if x["物种名称"] in species_config:
                    if x["物种名称"] not in species_error_info:
                        species_error_info[x["物种名称"]] = {'年龄': {}, '性别': {}, 'total': 0}
                    species_error_info[x["物种名称"]]['total'] += 1
                    species_error_info[x["物种名称"]]['年龄'][x["年龄"]] = species_error_info[x["物种名称"]]['年龄'].\
                        setdefault(x["年龄"], 0) + 1
                    species_error_info[x["物种名称"]]['性别'][x["性别"]] = species_error_info[x["物种名称"]]['性别'].\
                        setdefault(x["性别"], 0) + 1
        if new_objects:
            new_image_datas[k] = v
            for x in new_objects:
                x["yolo_positions"] = transform_to_yolo_positions(x["位置坐标"], v["图片宽度"], v["图片高度"])
            new_image_datas[k]["objects"] = new_objects

    imageIDs = [id for id in new_image_datas.keys()]
    X_train, X_test = train_test_split(imageIDs, test_size=test_ratio)
    class_names = species_config_and_train_data_to_class_names(species_config, train_image_info)
    write_class_names(class_names, dataset_path, species_config_file)
    for k, v in species_error_info.items():
        print('species error info:', k, v)

    if os.path.exists(dataset_path + "/train/"):
        shutil.rmtree(dataset_path + "/train/")
    if os.path.exists(dataset_path + "/test/"):
        shutil.rmtree(dataset_path + "/test/")
    train_images = {}
    test_images = {}
    for k, v in new_image_datas.items():
        if k in X_train:
            train_images[k] = v
            move_image_to_directory(dataset_path + "/train/", v, class_names)
        else:
            test_images[k] = v
            move_image_to_directory(dataset_path + "/test/", v, class_names)
    print("train_images size:{},test_images size:{},X_train:{},X_test:{}.".format(len(train_images), len(test_images),
                                                                                  len(X_train), len(X_test)))
    print("all images instaces info:{}".format(train_image_info))
    return train_images, test_images


def move_image_to_directory(path, datas, class_names):
    label_dir = path + "/labels"
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    image_dir = path + "/images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    image_and_label_name = datas["图片id"].split(".")[0]
    try:
        shutil.move(datas["本地路径"], image_dir + "/" + image_and_label_name + ".jpg")
        with open(label_dir + "/" + image_and_label_name + ".txt", "w") as f:
            for x in datas["objects"]:
                class_index = class_names.index(x["species"])
                for pos in x["yolo_positions"]:
                    f.write(str(class_index) + " " + str(pos[0]) + " " + str(pos[1]) + " " +
                            str(pos[2]) + " " + str(pos[3]) + "\n")
    except Exception:
        print("error:", datas["图片id"])


def main(file_name, species_config_file, target_path, test_ratio=0.2):
    new_image_datas = parse_raw_data(file_name)
    yolo_image_datas = label_transform_yolo_format_and_split(species_config_file, new_image_datas, target_path, test_ratio)
    print("训练集图片数量:{},测试集图片数量:{}".format(len(yolo_image_datas[0]), len(yolo_image_datas[1])))
    return yolo_image_datas


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--file', type=str,
                        required=True, help='输入原始标注数据的json文件')
    parser.add_argument('-s', '--sfile', type=str,
                        required=True, help='需要进行识别的物种类别配置yaml文件')
    parser.add_argument('-p', '--path', type=str,
                        required=True, help='yolo格式的图片和label存储的路径')
    parser.add_argument('-t', '--ratio', type=float,
                        default=0.1, help='训练数据中测试集的比例')
    args = parser.parse_args()
    main(args.file, args.sfile, args.path, args.ratio)
