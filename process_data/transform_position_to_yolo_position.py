import json
from PIL import Image
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial


# 该函数的输入是process_data/get_animal_image_transform_train_test_data.py调整后的json文件
def read_image_data_json(json_file_path):
    # image_id -> local_path, {物种名称 : yolo坐标}
    image_id_local_path_and_position_list_map = {}
    invalid_position_num = 0
    for line_number, line in enumerate(open(json_file_path), 1):
        line = line.strip()
        try:
            data = json.loads(line)
        except Exception as e:
            print(f"json.loads(line) error,exception:{e}")
            print(line_number, line)
            continue
        image_id = data['图片id']
        # local_path = data['本地路径']
        # 临时加的代码
        local_path = data['本地路径'].replace("animal_action", "animal_action.bak")
        species_name = data['species_name']
        # 无位置坐标
        if len(data['位置坐标']) < 3:
            invalid_position_num += 1
            continue
        if image_id not in image_id_local_path_and_position_list_map:
            image_id_local_path_and_position_list_map[image_id] = ["", {}]
            image_id_local_path_and_position_list_map[image_id][0] = local_path
            image_id_local_path_and_position_list_map[image_id][1] = {}
            image_id_local_path_and_position_list_map[image_id][1][species_name] = [data["位置坐标"]]
        else:
            if local_path != image_id_local_path_and_position_list_map[image_id][0]:
                print("local_path != image_id_local_path_and_position_list_map[image_id][0]")
                print(local_path)
                print(image_id_local_path_and_position_list_map[image_id][0])
                exit(-1)
            if species_name in image_id_local_path_and_position_list_map[image_id][1]:
                image_id_local_path_and_position_list_map[image_id][1][species_name].append(data["位置坐标"])
            else:
                image_id_local_path_and_position_list_map[image_id][1][species_name] = [data["位置坐标"]]
    # 是否输出下一个image中是否有不同的动物
    # for image_id, value in image_id_local_path_and_position_list_map.items():
    #     species_info = value[1]
    #     species_name = set()
    #     for species in species_info.keys():
    #         species_name.add(species.split(',')[0])
    #     if len(species_name) > 1:
    #         print("image_id:", image_id)
    #         print("species_name:", species_name)
    #         print("local_path:", value[0])
    print(f"invalid_position_num: {invalid_position_num}")
    return image_id_local_path_and_position_list_map


def transform_position(w, h, position_str_list, positions_list):
    """从原始数据的标注的框位置转化为真实图片标注框的位置，图片位置记录方式为 左上x坐标，左上y坐标，右下x坐标，右下y坐标"""
    for position_str in position_str_list:
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
        return True
    return True


def parse_raw_data_get_positions(animal_action):
    new_image_datas = {}
    error_possition = set()
    error_image = set()
    total_image = set()
    for image_data in open(animal_action):
        total_image.add(image_data["图片id"])
        try:
            image = Image.open(image_data["本地路径"])
        except Exception:
            print(f"error:{image_data}")
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


def read_name_to_class_id_file(file_name):
    name_to_class_map = {}
    for line in open(file_name):
        line = line.strip()
        if len(line) == 0:
            continue
        line_split = line.rsplit(":", 1)
        name_to_class_map[line_split[0]] = int(line_split[1])
    return name_to_class_map


def get_image_class_and_yolo_position(image_item, name_to_class_map):
    local_path = image_item[0]
    class_and_positions_map = image_item[1]
    try:
        image = Image.open(local_path)
    except Exception as e:
        print(f"error open image:{local_path},exception:{e}")
        return False
    if len(image_item) != 3:
        image_item.append({})
    class_id_to_positions_map = image_item[2]
    w, h = image.size
    name_class_to_position_list = {}
    for name, position_str_list in class_and_positions_map.items():
        positions_list = []
        if name not in name_to_class_map:
            continue
        class_id = name_to_class_map[name]
        if not transform_position(w, h, position_str_list, positions_list):
            # print("error possition:", line)
            print(f"error transform_position:{local_path},name:{name},position_str_list:{position_str_list}")
            continue
        if len(positions_list) == 0:
            continue
        if class_id not in name_class_to_position_list:
            name_class_to_position_list[class_id] = positions_list
        else:
            name_class_to_position_list[class_id].extend(positions_list)
    if len(name_class_to_position_list) == 0:
        return True
    name_class_to_yolo_position_list = {}
    for class_id, positions_list in name_class_to_position_list.items():
        yolo_position_list = transform_to_yolo_positions(positions_list, w, h)
        name_class_to_yolo_position_list[class_id] = yolo_position_list
    if len(name_class_to_yolo_position_list) == 0:
        return True
    else:
        for k, v in name_class_to_yolo_position_list.items():
            if k in class_id_to_positions_map:
                class_id_to_positions_map[k].extend(v)
            else:
                class_id_to_positions_map[k] = v
    return True


def process_image_data(image_id_local_path_and_position_list_map, name_to_class_map):

    for image_id, value in image_id_local_path_and_position_list_map.items():
        if len(value) == 0:
            continue
        if not get_image_class_and_yolo_position(value, name_to_class_map):
            print(f"error get_image_class_and_yolo_position image_id:{image_id}")
            continue


def save_yolo_lable_txt(image_id_local_path_and_position_list_map):
    for _, v in image_id_local_path_and_position_list_map.items():
        local_path = v[0]
        class_id_to_positions_map = v[2]
        split_list = local_path.split("/")
        root_dir = "/".join(split_list[:-2])
        image_name = split_list[-1]
        file_path = root_dir + "/labels/" + image_name.split('.')[0] + ".txt"
        with open(file_path, "w") as f:
            for class_id, positions_list in class_id_to_positions_map.items():
                for position in positions_list:
                    f.write(f"{class_id} {position[0]} {position[1]} {position[2]} {position[3]}\n")


def main(root_dir, class_info_file, json_file):
    image_id_local_path_and_position_list_map = read_image_data_json(json_file)
    name_to_class_map = read_name_to_class_id_file(class_info_file)
    process_image_data(image_id_local_path_and_position_list_map, name_to_class_map)
    not_have_position_num = 0
    with open(f"{root_dir}/animal_json_process_with_yolo_position.json", "w") as f:
        for k, v in image_id_local_path_and_position_list_map.items():
            if len(v[2]) == 0:
                not_have_position_num += 1
                continue
            f.write(json.dumps({k: v}, ensure_ascii=False) + "\n")

    print(f"not_have_position_num: {not_have_position_num}")
    save_yolo_lable_txt(image_id_local_path_and_position_list_map)


if __name__ == "__main__":
    """
    动物标注的位置信息转换成yolo格式的位置信息
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root_dir', type=str,
                        default="/mnt/data2/all_species_240916/animal_action.bak", help='动物照片下载的目录')
    parser.add_argument('-c', '--class_info_file', type=str,
                        default="/mnt/data2/all_species_240916/animal_action.bak/species_name_to_class_gt_10.txt", help='物种类别信息转id的码表')
    parser.add_argument('-j', '--json_file', type=str,
                        default="/mnt/data2/all_species_240916/animal_action.bak/success.json", help='动物照片下载成功的照片信息')
    args = parser.parse_args()
    main(args.root_dir, args.class_info_file, args.json_file)
