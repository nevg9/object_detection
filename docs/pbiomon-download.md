# Pbiomon 图片、视频和最终鉴定归档

部署目标：`image0_processing`，数据根目录 `/mnt/data2/pbiomon_yolo`。
本工具归档原始数据，不转换未知坐标、不导出 YOLO 标签、不启动训练。
本轮用户指定截止日期为 **2026-11-01（含当天，北京时间，最终鉴定首次完成日期）**。

## 数据与网络

- API：`https://open.pbiomon.cn/api/infcam/final-identifications/query`。
- 每页 200 条，不按物种或 AI 一致性裁剪（`aiConsistency=-1`）。
- 默认同时下载 `PHOTO` 和 `VIDEO`，4 个工作线程，其中一个优先处理视频。
- 资源下载只允许 HTTPS `resource.pbiomon.cn:443`，curl 强制
  `--resolve resource.pbiomon.cn:443:10.249.0.110 --noproxy '*'`。
- 保持原 URL 的 Host/SNI 和证书校验；不跟随重定向，不回退公网。
- API 请求间隔至少 6.5 秒，429 后等待 65 秒重新签名。
- API 密钥经交互式隐藏输入读取；后台进程只在内存中保留密钥。
  不写入代码、参数、日志、配置或状态文件。退出后恢复任务需要重新输入。
- 原始 API 页和 SQLite 包含资源下载地址，属于内部数据；不要把这些原始
  内容直接粘贴到公开日志。`status`、事件日志和下载回执不输出下载地址。

## 状态与完整性

- 资源主键：`(resourceType, resourceId)`；物种明细主键：
  `(resourceType, resourceId, speciesId)`。
- 每页原始 JSON 先压缩、原子保存，再在一个事务中更新所有记录及下一游标。
- 重复返回已有资源时仍处理全部物种标注，已完成文件不会重置为待下载。
- 全局进程锁防止同一根目录运行两个主进程；任务领取用数据库事务串行化。
- 文件写入 `.part`，完整接收、校验后进入内容哈希存储；同内容文件复用。
- 回执先于数据库完成标记保存；重启可用回执恢复，而不再次请求文件。
- 重启检查完成文件的存在和大小；mtime 改变时重新校验 SHA256。
- 图片使用 Pillow 检查格式和尺寸；视频有 ffprobe 时检查容器和视频流。
  容器检查不等于逐帧解码，更不等于框已与视频帧正确对齐。
- 图片默认上限 256 MiB，视频默认上限 16 GiB，可用命令参数调整。
  超限或无法校验的资源保留明确失败状态，不计作完成。
- 数据盘剩余低于 100 GiB 时暂停文件下载并保留任务，不自动删除数据。
- `raw/annotations/.../current.json` 汇总目前观察到的物种和最终框；同目录
  保留生成过的内容版本。原始 API 页保留每次抓到的原始响应。
- 当前 API 没有快照及变更游标。汇总只代表已观察记录，不承诺平台某时刻的
  全量快照；不能仅因某次未出现某物种就自动删除本地历史标注。
- `complete=false` 和 `coordinateUnit=UNKNOWN` 原样保留，不写空 YOLO 标签。
  下载数量与可用于检测训练的数量分开统计。

## 查询分片

首次无开始日期时，按月从截止月份向历史查询至 2020 年 1 月，另外有一个
无开始日期、截止 2019-12-31 的分片，覆盖更早数据，不假定最早年份。
超过 1000 页的大日期分片自动一分为二重扫，已下载文件仍复用。
无法进一步按日拆分而超过游标预算时，明确记录错误；不会跳过后宣称全量完成。
游标失效可重扫分片，当前接口无法保证动态变更完全不漏。

后续指定新的 `--start-date` / `--end-date` 启动新的扫描，包含上一次截止日或
更长重叠窗口。旧日期回填、修订、撤销仍需要平台变更机制或滚动历史核对。
扫描进度和文件下载进度分开保存；下载结束日期不能代替查询截止日期。

## 运行

命令在服务器上执行。代码位于数据根目录的 `bin/pbiomon_sync.py`。

```bash
python3 /mnt/data2/pbiomon_yolo/bin/pbiomon_sync.py run \
  --root /mnt/data2/pbiomon_yolo \
  --mode history --end-date 2026-11-01 --include PHOTO VIDEO \
  --private-ip 10.249.0.110 --workers 4 --reserve-gib 100 --detach
```

交互输入密钥，父进程返回后台 PID 和 run_id。同一日期范围重复执行会恢复
未完成任务，不清空数据库或图片库。完成后启动同范围新扫描可用 `--new-scan`；
`--retry-failed` 显式恢复失败下载。没有访问地址的资源等待后续采集更新。

查询范围全部保存成功后才推进 `catalog_end_date`。`initial_requested_end_date`
保留首轮请求截止日；程序完成日期不替代查询截止日期。后续补拉可执行：

```bash
python3 /mnt/data2/pbiomon_yolo/bin/pbiomon_sync.py run \
  --root /mnt/data2/pbiomon_yolo --mode incremental \
  --end-date YYYY-MM-DD --include PHOTO VIDEO --detach
```

默认从上次成功截止日前 6 天开始（含截止日在内重扫 7 天），补至新截止日。
旧原图复用，物种标注仍更新。`auto` 模式优先恢复同截止日的未完成查询，
有历史水位时按增量处理。历史查询未完成时不允许跳过它推进增量水位。

```bash
python3 /mnt/data2/pbiomon_yolo/bin/pbiomon_sync.py status --root /mnt/data2/pbiomon_yolo
python3 /mnt/data2/pbiomon_yolo/bin/pbiomon_sync.py verify --root /mnt/data2/pbiomon_yolo --limit 1000 --hash
python3 /mnt/data2/pbiomon_yolo/bin/pbiomon_sync.py stop --root /mnt/data2/pbiomon_yolo
```

`stop` 请求优雅退出，保留查询进度、回执、已完成文件和未完成任务。

## 验证

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_pbiomon_sync.py -v
```

测试只使用临时目录和构造记录，覆盖跨页归并、并发领取、事务回滚、回执恢复、
已完成文件复用、原始坐标保留以及内网路由约束。线上小范围试运行和正式下载
结果保存在服务器 `reports/`。不运行仓库现有会修改或删除数据的清理脚本。
