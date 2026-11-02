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

## 资源失败与重试

- 资源 HTTP 404/410 首次失败后进入 `unavailable`，保留标注和失败原因，
  不进入高频重试队列。这表示该 URL 当前不可用，不是对物理存储的永久不存在判定。
- 资源 HTTP 401/403 进入 `refresh_required`，等待源链接刷新，避免反复请求过期链接。
- 资源 HTTP 429 至少等待 65 秒；429/503 等提供的 `Retry-After` 秒数或 HTTP 日期会被尊重。
- 网络错误、408、5xx 进行 30、60、120、240、300 秒级退避并加抖动，
  每个链接最多尝试 5 次。其他确定性 4xx、证书和路由错误不做普通热重试。
- 新资源优先处理；有持续新任务时，每九个新任务给一个已到期的暂时故障重试机会。
  无新任务时正常处理到期重试，不让重试永久饥饿。
- 升级时将旧账本中的 404/410 重试或失败项迁为 `unavailable`，不删除资源或标注。
- 后续采集到变更后的下载 URL 时，自动恢复一次新链接尝试。旧链接的在途失败
  不会把已经刷新的新链接标记为不可用。
- 普通 `--retry-failed` 不重新激活 `unavailable`；平台修复源文件后可显式加
  `--recheck-unavailable` 复查一轮。默认不会对这些链接反复轮询。
- 新下载回执增加实际请求 URL 的 SHA256 和资源元数据哈希，形成资源、源链接与文件
  之间的可审计关联，仍不将完整下载 URL 写入回执或日志。旧回执不会伪造补填此信息。

## 关联与源链接审计

```bash
python3 /mnt/data2/pbiomon_yolo/bin/pbiomon_audit.py --root /mnt/data2/pbiomon_yolo
```

工具先复制一致的本地 SQLite 快照，再逐条核对标注与原始 API 页、资源主键、
汇总物种集合、文件回执和完整文件 SHA256。它还在内网独立复取受限样本比对哈希，
生成图片缩略图及视频时间点样本供视觉核查。并发更新的 `current.json` 不会被
误判为旧快照错配；核对时同时查找其不可变汇总版本。尚未归并的标注单独统计。
旧回执没有源链接哈希时会明确记为 legacy，不能事后伪造其请求证据；新回执的
链接与资源元数据哈希必须作为一对出现在同一资源的原始 API 记录中。

`pbiomon_diagnose_404.py` 是手动有界诊断：抽取失败链接与成功对照，使用 HTTPS
和正确 Host/SNI，对内外网各最多读取 1 KiB 进行比较，不输出链接。
该工具不用于正式下载，正式下载始终保持内网路由。

这类审计验证的是已观察数据的对应关系；不能据此宣称 API 返回物种和框在视觉上
逐张正确。坐标系未确认时，不把诊断假设当作训练转换规则。

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
