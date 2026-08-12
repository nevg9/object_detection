# 8 卡兼容推理服务

本目录是 `flask_web/server_241124.py` 的新增多 GPU 实现。旧文件不会被修改或删除；新旧服务在正式切换前必须通过结果一致性检查。

## 对外接口兼容

以下三个路径与旧服务保持一致：

- `POST /predict_classifier`
- `POST /predict_detect`
- `POST /predict_video`

请求体仍然是图片或视频的原始二进制数据。可选请求头仍然是 `resourceId` 和 `mediaType`。分类、检测的响应体直接由兼容 GPU worker 产生，网关不会重新组装，因此字段名、层级、类型和可选字段与旧服务一致。

分类响应示例：

```json
{
  "prediction": 0.9981,
  "label": 0,
  "desc": "动物",
  "resourceId": "photo-001",
  "mediaType": "photo"
}
```

检测响应示例：

```json
{
  "results": [
    ["小麂雄性成体", [123.0, 45.0, 456.0, 378.0], 0.91, ["小麂", "雄性", "成体"]]
  ],
  "agg_results": [
    [["小麂", "雄性", "成体"], 1]
  ],
  "resourceId": "photo-001",
  "mediaType": "photo"
}
```

## 目录结构与调度方式

- `gateway.py`：公网兼容入口、请求落盘、图片/视频公平调度和视频结果汇总。
- `worker.py`：单 GPU 模型服务。每个进程加载一份分类模型和 TensorRT 检测模型。
- `scheduler.py`：8 个 GPU worker 的任务分配；图片和视频帧同时积压时按 1:1 交替派发，单类流量可使用全部 8 卡。
- `spool.py`、`cleanup.py`：请求缓存、50GB 磁盘低水位和定时清理。
- `consistency_check.py`：逐字段比较新旧服务结果。
- `deploy/`：Nginx、systemd、环境变量和 logrotate 配置。

请求缓存在 `/mnt/data2/object_detection/spool`。请求结束时立即删除；每 15 分钟清理一次未锁定且超过 2 小时的孤儿文件。日志保留 7 天。

## 创建独立运行环境

不要修复或覆盖当前 `torch2` 环境。它的 Python 3.10 可执行文件已经被升级操作删除，当前旧服务只能依靠内存中的进程继续运行。

这台服务器当前采用下面的兼容快照方式。新 Python 环境和包目录都是独立路径；`cp -al` 只创建硬链接，不复制约 9.5GB 的包内容，也不会修改旧目录。即使以后删除旧目录，快照中的硬链接仍然有效。

```bash
/home/yuzhong/miniconda3/bin/conda create \
  -p /home/yuzhong/.venvs/object-detection \
  python=3.10.15 pip=24.2 -y

mkdir -p /home/yuzhong/.venvs/object-detection-legacy-packages
cp -al \
  /home/yuzhong/miniconda3/envs/torch2/lib/python3.10/site-packages/. \
  /home/yuzhong/.venvs/object-detection-legacy-packages/

/home/yuzhong/.venvs/object-detection/bin/pip install \
  --no-deps gunicorn==23.0.0

export PYTHONPATH=/home/yuzhong/.venvs/object-detection-legacy-packages
/home/yuzhong/.venvs/object-detection/bin/python -m pip check
```

如果不能使用现有包快照，也可以按锁文件从头安装。PyAV 14.4.0 已从公开 PyPI 下架，需要先从旧包目录无损重建本机 wheel：

```bash

av_wheel_stage="$(mktemp -d /tmp/od-av-wheel.XXXXXX)"
mkdir -p /home/yuzhong/.wheelhouse
cp -a /home/yuzhong/miniconda3/envs/torch2/lib/python3.10/site-packages/av "${av_wheel_stage}/"
cp -a /home/yuzhong/miniconda3/envs/torch2/lib/python3.10/site-packages/av.libs "${av_wheel_stage}/"
cp -a /home/yuzhong/miniconda3/envs/torch2/lib/python3.10/site-packages/av-14.4.0.dist-info "${av_wheel_stage}/"
/home/yuzhong/.venvs/object-detection/bin/wheel pack \
  "${av_wheel_stage}" -d /home/yuzhong/.wheelhouse
/home/yuzhong/.venvs/object-detection/bin/pip install \
  /home/yuzhong/.wheelhouse/av-14.4.0-*.whl

/home/yuzhong/.venvs/object-detection/bin/pip install \
  -r /mnt/data1/code/object_detection/flask_web/multi_gpu_server/requirements.lock

sudo mkdir -p /mnt/data2/object_detection/spool /mnt/data2/object_detection/logs
sudo chown -R yuzhong:yuzhong /mnt/data2/object_detection
chmod 700 /mnt/data2/object_detection/spool
chmod 750 /mnt/data2/object_detection/logs
```

验证依赖和 GPU：

```bash
set -a
source /mnt/data1/code/object_detection/flask_web/multi_gpu_server/deploy/object-detection.env.example
set +a
/home/yuzhong/.venvs/object-detection/bin/python -c 'import torch, flask, tensorrt, ultralytics; print(torch.__version__, torch.cuda.device_count())'
nvidia-smi -L
```

## 手动启动

先加载配置，然后分别启动 8 个 worker 和网关。worker 只监听本机端口 `5100` 到 `5107`，网关监听 `5090`。

```bash
set -a
source /mnt/data1/code/object_detection/flask_web/multi_gpu_server/deploy/object-detection.env.example
set +a

for gpu in 0 1 2 3 4 5 6 7; do
  /mnt/data1/code/object_detection/flask_web/multi_gpu_server/scripts/run_worker.sh "$gpu" &
done

/mnt/data1/code/object_detection/flask_web/multi_gpu_server/scripts/run_gateway.sh
```

检查状态：

```bash
curl -s http://127.0.0.1:5090/health/ready
curl -s http://127.0.0.1:5100/health/ready
```

## 自己请求一张照片

请求格式与旧服务完全相同。分类：

```bash
curl -X POST 'http://127.0.0.1:5090/predict_classifier' \
  -H 'Content-Type: application/octet-stream' \
  -H 'resourceId: photo-001' \
  -H 'mediaType: photo' \
  --data-binary '@/path/to/photo.jpg'
```

检测：

```bash
curl -X POST 'http://127.0.0.1:5090/predict_detect' \
  -H 'Content-Type: application/octet-stream' \
  -H 'resourceId: photo-001' \
  -H 'mediaType: photo' \
  --data-binary '@/path/to/photo.jpg'
```

也可以使用仓库中的 Python 示例：

```bash
/home/yuzhong/.venvs/object-detection/bin/python -m flask_web.multi_gpu_server.request_example \
  /path/to/photo.jpg \
  --url http://127.0.0.1:5090 \
  --endpoint predict_classifier

/home/yuzhong/.venvs/object-detection/bin/python -m flask_web.multi_gpu_server.request_example \
  /path/to/photo.jpg \
  --url http://127.0.0.1:5090 \
  --endpoint predict_detect
```

## 新旧结果一致性验证

旧服务保持在 `http://127.0.0.1:5002`，新服务先通过旁路端口提供访问，例如 `http://127.0.0.1:15002`。准备多张包含动物、人、空白和多目标场景的图片，然后执行：

```bash
/home/yuzhong/.venvs/object-detection/bin/python -m flask_web.multi_gpu_server.consistency_check \
  --old-url http://127.0.0.1:5002 \
  --new-url http://127.0.0.1:15002 \
  --images /path/to/compatibility-images
```

同时验证现有视频样本：

```bash
/home/yuzhong/.venvs/object-detection/bin/python -m flask_web.multi_gpu_server.consistency_check \
  --old-url http://127.0.0.1:5002 \
  --new-url http://127.0.0.1:15002 \
  --video /mnt/data1/animal_sample250.mp4
```

检查器要求 HTTP 状态、Content-Type、JSON 键、列表顺序、标签、类别、坐标、聚合结果和可选请求头回传完全一致。浮点推理值允许 `1e-5` 绝对误差；视频的 `processing_ms` 必须存在且为数字，但耗时数值本身不做相等要求。任何不一致都会返回非零退出码，禁止切换生产入口。

## 并发压测

```bash
/home/yuzhong/.venvs/object-detection/bin/python -m flask_web.multi_gpu_server.load_test \
  --url http://127.0.0.1:15002 \
  --images /path/to/load-test-images \
  --concurrency 64 \
  --qps 40 \
  --duration 900 \
  --classifier-ratio 0.6
```

验收标准是 p95 不超过 1000ms、错误率低于 0.1%。同时检查 8 个 `worker-*-access.log` 和 `nvidia-smi`，确认每张卡都收到任务且任务量偏差不超过 20%。

## systemd、Nginx 和清理任务

安装配置前，先把 Nginx 配置中的 `listen 5002` 临时改成 `listen 15002`，确保不影响旧服务。以下操作需要管理员权限：

```bash
sudo mkdir -p /etc/object-detection
sudo cp flask_web/multi_gpu_server/deploy/object-detection.env.example /etc/object-detection/object-detection.env
sudo cp flask_web/multi_gpu_server/deploy/object-detection-worker@.service /etc/systemd/system/
sudo cp flask_web/multi_gpu_server/deploy/object-detection-gateway.service /etc/systemd/system/
sudo cp flask_web/multi_gpu_server/deploy/object-detection-cleanup.service /etc/systemd/system/
sudo cp flask_web/multi_gpu_server/deploy/object-detection-cleanup.timer /etc/systemd/system/
sudo cp flask_web/multi_gpu_server/deploy/object-detection.target /etc/systemd/system/
sudo cp flask_web/multi_gpu_server/deploy/object-detection.logrotate /etc/logrotate.d/object-detection
sudo cp flask_web/multi_gpu_server/deploy/nginx-object-detection.conf /etc/nginx/conf.d/object-detection.conf

sudo systemctl daemon-reload
sudo systemctl enable --now object-detection.target
sudo systemctl enable --now object-detection-cleanup.timer
sudo nginx -t
sudo systemctl reload nginx
```

所有一致性检查和压测通过后，停止旧进程，把 Nginx 监听端口从 `15002` 改回 `5002`，再次执行 `nginx -t` 后 reload。不要在新环境和旁路服务验证完成前停止旧进程。

常用排查命令：

```bash
systemctl status 'object-detection-worker@*.service' object-detection-gateway.service
journalctl -u object-detection-gateway.service -n 100 --no-pager
tail -f /mnt/data2/object_detection/logs/gateway.log
ls -lh /mnt/data2/object_detection/spool
```
