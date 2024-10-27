# object detection

## 增加无目标的照片训练图像识别模型

使用shell命令拷贝无目标和人的照片到训练识别模型的数据目录下，执行如下命令:

```shell
find . -maxdepth 1 -type f -iname "*.JPG" | sort | awk 'BEGIN {srand(42)} {print rand(), $0}' | sort -k1,1n | cut -d' ' -f2- | head -n 54000 | xargs -I {} cp {} /mnt/data2/all_species_240916/animal_action/train/images

find . -maxdepth 1 -type f -iname "*.JPG" | sort | awk 'BEGIN {srand(42)} {print rand(), $0}' | sort -k1,1n | cut -d' ' -f2- | tail -n 6000 | xargs -I {} cp {} /mnt/data2/all_species_240916/animal_action/test/images

```

## 移除掉非法数据

使用以下命令进行处理：

```shell
python remove_dir_invalid_image.py -d /mnt/data2/all_species_240916/animal_action
```
