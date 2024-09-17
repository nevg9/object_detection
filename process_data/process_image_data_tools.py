import random
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import os
import shutil
import hashlib
import json


def get_file_path(content, root_dir):
    # 计算图片id的md5
    pic_id = content["图片id"]
    md5hash = hashlib.md5(pic_id.encode('utf-8'))
    md5 = md5hash.hexdigest()

    # root_dir = "/home/yuzhong/nndata/fs"

    first = md5[0]
    second = md5[1:3]
    third = md5[3:6]
    path = "/".join([root_dir, first, second, third, pic_id])
    # print(path)
    my_file = Path(path)
    if my_file.is_file():
        content["有效路径"] = path
        return path
    else:
        new_path = "/".join([root_dir, first, second, third, md5])
        # print(new_path)
        my_file = Path(new_path)
        if my_file.is_file():
            content["有效路径"] = new_path
            return new_path
        else:
            content["无效路径"] = path + "\t" + new_path
            return None


def copy_image_to_local(content, root_dir):
    try:
        Image.open(content["有效路径"])
        content["本地路径"] = root_dir + "/" + content["有效路径"].split("/")[-1]
        shutil.copyfile(content["有效路径"], content["本地路径"])
        return True
    except Exception:
        content["本地路径"] = "corrupted"
        # json.dump(content, copy_file_failed_fd, ensure_ascii=False)
        # copy_file_failed_fd.write("\n")
        return False


def process_content(content, test_image_id_list, train_image_id_list, train_dir, test_dir, fs_dir):
    path = get_file_path(content, fs_dir)
    if content["图片id"] in test_image_id_list:
        content["本地路径"] = test_dir + "/" + "images"
    elif content["图片id"] in train_image_id_list:
        content["本地路径"] = train_dir + "/" + "images"
    else:
        content["本地路径"] = "not_exist"
        return content, 'not_exist'
    if path:
        if copy_image_to_local(content, root_dir=content["本地路径"]):
            return content, 'success'
        else:
            return content, 'corrupted'
    else:
        return content, 'not_exist'


def json_contents_to_path(actions, root_dir, test_image_id_list_temp, train_image_id_list, train_dir, test_dir, fs_dir, max_workers=3):
    process_func = partial(process_content, test_image_id_list=test_image_id_list_temp, train_image_id_list=train_image_id_list,
                           train_dir=train_dir, test_dir=test_dir, fs_dir=fs_dir)

    write_images = 0
    path_not_exist = 0
    image_corrupted = 0
    invalid_json = []
    json_file = root_dir + "/success.json"
    with ProcessPoolExecutor(max_workers=max_workers) as executor, open(json_file, 'w') as json_file_fd:
        futures = [executor.submit(process_func, content)
                   for content in actions]

        for future in tqdm(as_completed(futures), total=len(actions)):
            content, status = future.result()
            if status == 'success':
                json.dump(content, json_file_fd, ensure_ascii=False)
                json_file_fd.write("\n")
                write_images += 1
            elif status == 'corrupted':
                invalid_json.append(content)
                image_corrupted += 1
            else:  # 'not_exist'
                invalid_json.append(content)
                path_not_exist += 1

    print(f"成功写入的照片数量:{write_images},"
          f"路径非法总数:{path_not_exist},"
          f"照片损坏总数:{image_corrupted}.")

    with open(os.path.join(root_dir, "failed.json"), "w") as json_file:
        json.dump(invalid_json, json_file, ensure_ascii=False)


def read_images_json_file(json_file):
    with open(json_file, 'r') as f:
        actions = json.load(f)
        print(f"total image num: {len(actions)}")
        return actions
