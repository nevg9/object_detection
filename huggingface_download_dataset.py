import time
from huggingface_hub import snapshot_download

times = 0

while True:
    times += 1
    try:
        snapshot_download('mlfoundations/dclm-baseline-1.0', repo_type='dataset', local_dir='/mnt/data2/dclm-baseline-1', resume_download=True, 
                          max_workers=16)
        break
    except Exception as e:
        print(f"Failed to download dataset, retrying times {times}... {e}")
    if times > 100:
        break
    time.sleep(5)
