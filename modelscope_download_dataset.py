import time
from modelscope.hub.snapshot_download import dataset_snapshot_download

times = 0

while True:
    times += 1
    try:
        dataset_snapshot_download('AI-ModelScope/dclm-baseline-1.0', local_dir='/home/yuzhong/data2/dclm-baseline-1')
    except Exception as e:
        print(f"Failed to download dataset, retrying times {times}... {e}")
    if times > 100:
        break
    time.sleep(5)
