import random
from PIL import Image
import os
import shutil
import argparse
import csv
from process_image_data_tools import json_contents_to_path, read_images_json_file


def get_all_age_gender_set(animal_action):
    age_set = set()
    gender_set = set()
    print(f'total image num: {len(animal_action)}')
    for item in animal_action:
        age_set.add(item['年龄'])
        gender_set.add(item['性别'])
    return age_set, gender_set


def get_clean_species_statistic_info(file_name):
    species_map_to_count = {}
    with open(file_name, 'r') as file:
        next(file)  # 跳过第一行
        for line in file:
            line = line.strip()
            if line == "":
                continue
            data = line.split(",")
            if len(data) < 6:
                print(f"error format {line}")
                continue
            _, _, species, gender, age, count = data[0:6]
            if age == "幼体":
                if gender != "无法区分":
                    print(
                        f"{species} age is 幼体 need transfer gender {gender} to 无法区分")
                    gender = "无法区分"
            if "指名亚种" in species:
                species_column = species.split(" ")
                new_species = species_column[0]
                key = new_species + ',' + gender + ',' + age
                print(
                    f"orig species {species},{gender},{age} transfer to {key}")
            else:
                key = species + ',' + gender + ',' + age
            if key not in species_map_to_count:
                species_map_to_count[key] = 0
            species_map_to_count[key] += int(count)
    split_key_separator = ","
    species_map_to_count_sort = dict(sorted(species_map_to_count.items(), key=lambda item:
                                            (item[0].split(split_key_separator)[0],
                                             item[0].split(
                                                 split_key_separator)[1],
                                             item[0].split(split_key_separator)[2])))
    total_label_num = 0
    with open("species_statistic_info_new.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ID",	"标签",	"物种",	"性别",	"年龄", '数量'])
        id = 1
        for key, value in species_map_to_count_sort.items():
            species, gender, age = key.split(',')
            species_detail = species
            if gender != '无法区分':
                species_detail += gender
            if age != '无法区分':
                species_detail += age
            writer.writerow([id, species_detail, species, gender, age, value])
            id += 1
            total_label_num += value
    print(f"total_label_num is {total_label_num}")


def check_label_data(file_name, species_to_another_species, species_map_to_count):
    with open(file_name, 'r') as file:
        next(file)  # 跳过第一行
        for line in file:
            line = line.strip()
            if line == "":
                continue
            data = line.split(",")
            if len(data) < 5:
                print(f"error format {line}")
                continue
            species_map_to_count[data[2] + ',' + data[3] + ',' + data[4]] = 0
            if data[3] == "无法区分" and data[4] == "无法区分":
                if data[1] != data[2]:
                    print(f"ID {data[0]},species {data[1]} to {data[2]}")
                    species_to_another_species[data[1]] = data[2]
            if data[3] != "无法区分":
                if data[3] not in data[1]:
                    print(
                        f"ID {data[0]},species name {data[1]} not macth gender {data[3]}")
            if data[4] != "无法区分":
                if data[4] not in data[1]:
                    print(
                        f"ID {data[0]},species name {data[2]} not macth age {data[4]}")


def noomalize_species(item, species_to_another_species, gender_map, age_map):
    age = item['年龄']
    gender = item['性别']
    species = item['物种名称']
    if "指名亚种" in species:
        species_column = species.split(" ")
        old_species = species
        species = species_column[0]
        print(f"orig species {old_species} transfer to {species}")
    if species in species_to_another_species:
        species = species_to_another_species[species]

    if age not in age_map:
        age = "无法区分"
    else:
        age = age_map[age]
    if age == "幼体":
        if gender != "无法区分":
            # print(f"{species} age is 幼体 need transfer gender {gender} to 无法区分")
            gender = "无法区分"
    if gender not in gender_map:
        gender = "无法区分"
    else:
        gender = gender_map[gender]
    return species, gender, age


def get_species_num(animal_action, species_to_another_species, species_map_to_count, gender_map, age_map, not_found_species):
    all_species_num_map = {}
    print(f'total image num: {len(animal_action)}')
    for item in animal_action:
        species, gender, age = noomalize_species(
            item, species_to_another_species, gender_map, age_map)
        key = species + ',' + gender + ',' + age
        item['species_name'] = key
        # 在标注数据中存在的数据但是没在专家提供的表格中也需要作为训练数据
        if key not in species_map_to_count:
            not_found_species.add(key)
            species_map_to_count[key] = 0
            # print(f"{key} not found in species_map_to_count")
            # continue  # 跳过该数据项
        species_map_to_count[key] += 1
        if species not in all_species_num_map:
            all_species_num_map[species] = 0
        all_species_num_map[species] += 1
    return all_species_num_map


def species_statistic_info_to_csv(species_map_to_count, all_species_num_map, image_download_dir, filter_num, least_num_not_to_test_key_set):
    print(f"species_statistic_info_to_csv,image_download_dir:{image_download_dir}")
    split_key_separator = ","
    species_map_to_count_sort = dict(sorted(species_map_to_count.items(), key=lambda item:
                                            (item[0].split(split_key_separator)[0],
                                             item[0].split(
                                                 split_key_separator)[1],
                                             item[0].split(split_key_separator)[2])))
    species_name_to_class = {}
    with open(image_download_dir + "/species_num.txt", 'w') as f:
        for k, v in all_species_num_map.items():
            f.write(f"{k}:{v}\n")
    with open(image_download_dir + f"/species_statistic_info_ge_{filter_num}.csv", "w", newline='') as csvfile, \
            open(image_download_dir + f"/species_name_info_ge_{filter_num}.txt", "w") as txtfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ID", "标签", "物种", "性别", "年龄", '数量'])
        id = 1
        for key, value in species_map_to_count_sort.items():
            if key == "":
                continue
            species, gender, age = key.split(',')
            if species == '' or species == "不认识":
                continue
            # 只要大类的数据够了就可以了
            if species not in all_species_num_map:
                continue
            if all_species_num_map[species] < filter_num:
                continue
            if value <= 0:
                continue
            if value <= filter_num:
                least_num_not_to_test_key_set.add(key)
            species_detail = species
            if gender != '无法区分':
                species_detail += gender
            if age != '无法区分':
                species_detail += age
            writer.writerow([id, species_detail, species, gender, age, value])
            txtfile.write(f"{id - 1}: {species_detail}\n")
            id += 1
            species_name_to_class[key] = id - 1
    with open(image_download_dir + f"/species_name_to_class_gt_{filter_num}.txt", 'w') as f:
        for key, value in species_name_to_class.items():
            f.write(f"{key}:{value}\n")
    with open(image_download_dir + f"/species_key_le_{filter_num}.txt", 'w') as f:
        for key in least_num_not_to_test_key_set:
            f.write(f"{key}\n")
    return species_name_to_class


def test_image_json_have_dumplicate_image_id(image_json):
    image_id_num = {}
    for item in image_json:
        id = item["图片id"]
        if id in image_id_num:
            image_id_num[id] += 1
        else:
            image_id_num[id] = 1
    ge_2_num = 0
    for id, num in image_id_num.items():
        if num > 1:
            # print(id, num)
            ge_2_num += 1
    print(f"ge_2_num:{ge_2_num}")


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


def get_animal_image_train_test_data_to_local_dir(image_json_actions, image_json_name, fs_dir, root_dir, least_num_not_to_test_image_id_set,
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
    test_image_id_list_temp = image_id_list[:int(len(image_id_list) * test_ratio)]
    train_image_id_list = image_id_list[int(len(image_id_list) * test_ratio):]
    test_image_id_list = []
    adjust_num = 0
    for i in test_image_id_list_temp:
        if i in least_num_not_to_test_image_id_set:
            train_image_id_list.append(i)
            adjust_num += 1
        else:
            test_image_id_list.append(i)
    print("测试集图片数量：{}, 训练集图片数量：{}, adjust test to train num is {}.".format(len(test_image_id_list), len(train_image_id_list), adjust_num))
    json_contents_to_path(image_json_actions, root_dir, test_image_id_list,
                          train_image_id_list, train_dir, test_dir, fs_dir, max_workers)


def get_no_test_image_ids_set(animal_action, least_num_not_to_test_key_set, species_to_another_species, gender_map, age_map):
    least_num_not_to_test_image_id_set = set()
    for item in animal_action:
        species, gender, age = noomalize_species(
            item, species_to_another_species, gender_map, age_map)
        key = species + ',' + gender + ',' + age
        if key in least_num_not_to_test_key_set:
            least_num_not_to_test_image_id_set.add(item["图片id"])
    return least_num_not_to_test_image_id_set


def main(file_name, root_dir, dir_name, fs_dir, class_least_num, max_workers=3):
    # 查看照片中所有的年龄和性别的标注情况
    # 查看数据所有的年龄和性别标注情况
    animal_action = read_images_json_file(file_name)
    age_set, gender_set = get_all_age_gender_set(animal_action)
    print("年龄标注情况:", age_set)
    print("性别标注情况:", gender_set)
    # 未在这个map中有的都是无法区分
    gender_map = {"雄": "雄性", "雄+": "雄性", '雌带幼仔': '雌性', "雌": "雌性", "雄，另外一头不确定": "雄性", '': '无法区分'}
    # 未在这个map中有的都是无法区分
    age_map = {'': '无法区分', '成年': '成体', '亚成体': '亚成体', '幼体': "幼体", '成年;亚成体': '成体'}

    not_found_animal = set()
    species_to_another_species = {}
    species_map_to_count = {}
    image_download_dir = os.path.join(root_dir, dir_name)
    print(f"image download dir: {image_download_dir}")
    # 检查目录是否存在
    if os.path.exists(image_download_dir):
        # 如果存在，删除目录
        shutil.rmtree(image_download_dir)
        print(f"目录 {image_download_dir} 已删除")
    # 创建目录
    os.makedirs(image_download_dir)
    print(f"目录 {image_download_dir} 已创建")
    # 根据专家填写的分类码表，检查数据集的分类并把一些分类转换成另一个分类
    check_label_data("./物种分类码表-野生动物.csv",
                     species_to_another_species, species_map_to_count)
    check_label_data("./物种分类码表-家养动物.csv",
                     species_to_another_species, species_map_to_count)
    print(f"species_to_another_species:\n{species_to_another_species}")
    # 查看照片中所有的年龄和性别的标注情况, 并把对应的分类写到对应的json item中
    all_species_num_map = get_species_num(
        animal_action, species_to_another_species, species_map_to_count, gender_map, age_map, not_found_animal)

    print(
        f'len(species_map_to_count):{len(species_map_to_count)},len(not_found_animal):{len(not_found_animal)},all species:{len(all_species_num_map)}')
    # 根据动物大类过滤掉数量少的动物类别数据
    least_num_not_to_test_key_set = set()
    _ = species_statistic_info_to_csv(
        species_map_to_count, all_species_num_map, image_download_dir, class_least_num, least_num_not_to_test_key_set)
    print(f"least_num_not_to_test_key_set num: {len(least_num_not_to_test_key_set)}")
    # 得到不能分到test的图片id
    least_num_not_to_test_image_id_set = get_no_test_image_ids_set(animal_action, least_num_not_to_test_key_set, species_to_another_species, gender_map, age_map)
    print(f"least_num_not_to_test_image_id_set num: {len(least_num_not_to_test_image_id_set)}")
    test_image_json_have_dumplicate_image_id(animal_action)
    # 拉取 animal 数据
    get_animal_image_train_test_data_to_local_dir(animal_action, dir_name, fs_dir, root_dir, least_num_not_to_test_image_id_set, 0.1, max_workers)


if __name__ == '__main__':
    """
    从动物的json数据中，把照片和标签信息提取出来，并把图片复制到本地目录中
    输入：动物json数据，dump的根目录，以及dump到目录的名字，网盘挂载的路径
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json_name', type=str,
                        default="/home/yuzhong/nndata/export/20240428/物种图片1.json", help='下载的图片的json文件路径')
    parser.add_argument('-r', '--root_dir', type=str,
                        default="/mnt/data2/all_species_240916", help='该数据下载的根目录名字')
    parser.add_argument('-d', '--dir', type=str,
                        default="animal_action", help='该次下载的数据的目录名字')
    parser.add_argument('-f', '--fs', type=str,
                        default="/home/yuzhong/nndata/fs", help='网盘上的默认路径位置')
    parser.add_argument('-w', '--workers', type=int,
                        default=2, help='下载数据时候的进程数量')
    parser.add_argument('-c', '--class_least_num', type=int,
                        default=10, help='一个类型最少需要多少张图片')
    args = parser.parse_args()
    main(args.json_name, args.root_dir, args.dir, args.fs, args.class_least_num, args.workers)
