# Pbiomon 图片、视频和最终鉴定归档

部署目标：`image0_processing`，数据根目录 `/mnt/data2/pbiomon_yolo`。
本工具归档原始数据，不转换未知坐标、不导出 YOLO 标签、不启动训练。
本轮用户指定截止日期为 **2026-11-01（含当天，北京时间，最终鉴定首次完成日期）**。
以下完整标注约定依据用户提供的 v1.4.0 `openapi.yaml` 与 `接口文档.md`。
其中 OpenAPI 的 `info.version` 为 1.4.0，Markdown 文档页首仍写 1.3.0；
本次按两份文档均已定义的 `finalIdentification.attributes[]` 说明实现，
不将文档示例中的性别、年龄等演示值视为线上真实标注。

代码主干为 `main`，本地和服务器均跟踪 `origin/main`。服务器 Git 工作副本为
`/mnt/data2/pbiomon_yolo/code/repository`；`bin/` 中正式入口指向该工作副本的工具文件。
已验证的改动先在本地提交并推送 canonical `origin/main`，服务器再执行
`git pull --ff-only`。数据、账本、回执和日志放在工作副本外；源码行为变更后才按需
恢复进程载入新版，单纯分支指引调整且运行源码字节相同时不必重启下载。

## 数据与网络

- API：`https://open.pbiomon.cn/api/infcam/final-identifications/query`。
- 每页 200 条，不按物种或 AI 一致性裁剪（`aiConsistency=-1`）。
- 默认同时下载 `PHOTO` 和 `VIDEO`，4 个工作线程，其中一个优先处理视频。
- **默认保存接口返回的全部标注信息**，不需要另加性别、年龄等字段开关。
  `--include` 控制媒体下载类型，查询到的所有物种标注仍完整归档。
- 资源下载只允许 HTTPS `resource.pbiomon.cn:443`，curl 强制
  `--resolve resource.pbiomon.cn:443:10.249.0.110 --noproxy '*'`。
- 保持原 URL 的 Host/SNI 和证书校验；不跟随重定向，不回退公网。
- API 请求间隔至少 6.5 秒，429 后等待 65 秒重新签名。
- API 密钥经交互式隐藏输入读取；后台进程只在内存中保留密钥。
  不写入代码、参数、日志、配置或状态文件。退出后恢复任务需要重新输入。
- 原始 API 页和 SQLite 包含资源下载地址，属于内部数据；不要把这些原始
  内容直接粘贴到公开日志。`status`、事件日志和下载回执不输出下载地址。

## 完整标注字段（v1.4.0）

每条“资源 × 物种”记录完整保存 `finalIdentification` 和 `ai`，包括鉴定 ID、
首次完成时间、鉴定人、全部属性、备注、全部框及视频帧信息、AI 物种和一致性。
不按字段白名单裁剪这两个对象；未来新增的嵌套字段也原样保留。
原始响应在 `raw/pages/`，资源元信息和物种明细保存在 SQLite；
逐资源汇总在 `raw/annotations/{PHOTO或VIDEO}/{资源key前2位}/{资源key}/current.json`。

性别、年龄等字段的汇总 JSON 路径为
`annotations[].finalIdentification.attributes[]`，每项包含：

| 字段 | 含义 |
| --- | --- |
| `gender` | 性别 |
| `age` | 年龄 |
| `postureBehavior` | 姿势行为 |
| `generalBehavior` | 一般行为 |
| `breedingBehavior` | 繁殖行为 |
| `groupingBehavior` | 聚群行为 |
| `photoQuality` | 照片质量 |

图片和视频都保存这些字段。`attributes[]` 按原鉴定项顺序完整保存，不合并、
不去重、不截断；多项、重复项、七项全为 `null` 的项均保留。
属性属于当前资源中的当前物种，文档没有提供属性项与某个框、某帧或某个动物
个体的一对一关联，不能自行按数组下标配对；属性项数和 `boxCount` 都不能
直接当作个体数。

旧版响应缺少 `attributes` 与新版返回全空属性是不同情况。缺失字段不伪造，
`null` 不猜测、不用空字符串代替。只有线上接口实际返回的属性才能补全，
所有字段都有保存也不意味着每个字段都有非空业务值。

## 统一存储与后续 YOLO 数据集

### 全量下载和补标注共用一份归档

这是当前实现的共同格式，不是为补标注另建的一套格式。两种模式都调用
`Store._ingest_record()` 保存完整记录，再由同一个 `Store.export_annotations()`
生成逐资源 JSON；`--annotations-only` 只改变媒体处理，不改变标注结构与位置。

```text
/mnt/data2/pbiomon_yolo/
  state/state.sqlite
  raw/
    objects/<文件SHA256前2位>/<文件SHA256>.<扩展名>
    receipts/<资源key前2位>/<资源key>.json
    pages/<run_id>/<task_id>/<页号>-<响应摘要>.json.gz
    annotations/
      PHOTO/<资源key前2位>/<资源key>/current.json
      VIDEO/<资源key前2位>/<资源key>/current.json
      # 每个 current.json 同目录保留不可变的 <汇总摘要>.json 版本
  reports/status.json
```

资源 key 由 `(resourceType, resourceId)` 计算；物种记录仍用三元组
`(resourceType, resourceId, speciesId)`。不要用原文件名作为关联键：原文件名
可能重复，多个资源还可能共享同一内容哈希文件。

| 内容 | SQLite 当前记录 | 逐资源 JSON（两种模式相同，`schemaVersion=1`） |
| --- | --- | --- |
| 资源身份和拍摄/监测单元信息 | `resources.metadata_json` | `resource`，含 `resourceType`、`resourceId` 等原字段 |
| 原始媒体关联 | `resources.file_path`、`sha256`、尺寸及状态 | `file.path`、`sha256`、`byteSize`、`width`、`height`、`status` |
| 每个物种的全部标注 | `annotations.annotation_json` | `annotations[]` 中完整的 `speciesId`、`speciesName`、`finalIdentification`、`ai` |
| 性别、年龄、行为、质量 | 同一 `annotation_json` 内的原属性数组 | `annotations[].finalIdentification.attributes[]` |
| 图片框或视频逐帧框 | 同一 `annotation_json` 内的 `boxes[]` | 保留原坐标、单位、参考尺寸、`frameId`、`tsMs`、完整性和来源标记 |
| 标注版本与原始响应来源 | `annotation_hash`、时间和 `source_page` | `provenance[]` 中的物种、标注摘要、首次观察/变更时间及源页 |
| 图片 EXIF、视频流/容器信息 | `resources.media_json` | 下载回执的 `media`；当前汇总 JSON 的 `file` 不重复存这些信息 |

`file.path`、`provenance[].sourcePage` 都相对归档根目录。后续读取器只需按
`file.status == "done"` 选择有已下载媒体的记录，使用同一 JSON 解析逻辑，
不需要知道记录来自全量下载还是补标注。
补标注更新原路径下的 `current.json` 并保留旧版本，不改原图/视频路径、
SHA256 或媒体完成时间；资源上各物种的属性分别保存，不跨物种合并。

SQLite 是可查询的当前台账，`current.json` 是当前可读汇总，原始页与不可变
汇总版本用于追溯。新训练版本应冻结输入的不可变汇总路径及各物种
`annotationHash`，不能只记录会随下一轮补标注变化的 `current.json` 路径。

### YOLO 训练使用独立、可复现的导出目录

以下是约定的后续导出设计；当前下载工具不会创建这些训练文件或启动训练：

```text
exports/yolo/<数据集版本>/
  data.yaml
  classes.json
  images/{train,val,test}/<样本key>.jpg
  labels/{train,val,test}/<样本key>.txt
  metadata/samples.jsonl
  metadata/export.json
  reports/excluded.jsonl
```

训练图片和同名标签遵循 [Ultralytics 检测数据格式](https://docs.ultralytics.com/datasets/detect/)：
标签每个目标一行 `class_id x_center y_center width height`，类别从 0 开始，
坐标以对应训练图片的宽高归一化到 0—1。性别、年龄和其他业务属性保留在
完整源 JSON 与 `metadata/samples.jsonl` 中，不给标准检测 TXT 任意追加列；
标准物种检测训练不会自动学习这些额外属性。

导出约定：

- `classes.json` 固定 `speciesId -> class_id` 及显示名称的映射，`data.yaml`
  使用同一映射。已有类别 ID 不因物种名称修订或扫描顺序改变而重新排列。
- 图片样本 key 使用资源 key；视频帧样本使用 `<资源key>__f<原始frameId>`。
  原视频继续归档在 `raw/objects/`，检测训练使用对应的解码帧图片。
- 视频只在已确认的原始帧上使用该帧的框，保留从 0 开始的原 `frameId`、
  原 `tsMs`、实际解码时间及帧图片摘要；可变帧率时不能仅按 `时间 × FPS`
  推算帧序号，不把视频级物种标注复制成每一帧的检测框。
- 每条样本 metadata 记录样本 key、资源类型/ID、源媒体路径及 SHA256、
  不可变标注版本路径、各物种 `annotationHash`、全部原属性和框、导出框与
  源框下标的对应关系、坐标/方向变换规则、视频帧信息及所属数据划分。
  属性仍位于各自的物种记录中，不能把属性数组按下标强配给某个框。
- `export.json` 记录归档根、源扫描、导出代码提交、类别映射、筛选规则、
  坐标规则、抽帧规则及数据划分规则。导出版本完成后不随新一轮补标注修改；
  需要新标注时生成新版本。
- 同一原视频及相同内容的媒体/帧不能同时分入训练集和验证/测试集。
  可进一步按监测单元或拍摄事件分组，避免高度相近的素材跨集合。
- 图片可引用原媒体；需要解码、方向修正或其他图像变换时生成派生图片，
  不把训练预处理写回归档。原视频不为每个数据集版本再复制一份。

### 当前 YOLO 导出边界

目前接口的 `coordinateUnit=UNKNOWN`、参考画面尺寸未知，不能直接假设
`left/top/width/height` 是原图像素坐标后除以宽高。先确认坐标单位、参考
画面、图片 EXIF 方向/视频旋转、帧编号与时间含义，再按已确认规则换算，
验证叠加结果并在导出 metadata 中记录证据和变换。原始标注值保持不变。

只将媒体存在、坐标与帧映射可信且框标注满足本次训练要求的样本纳入检测集。
有物种但无框、缺框、不完整框、未标注的视频帧不能自动当作无动物负样本；
同一图片中已知存在却缺框的物种也不能静默丢弃。排除/待复核原因写入
`reports/excluded.jsonl`，这类素材仍可保留作分类等任务的候选数据。

性别/年龄等目前是“资源 × 物种 × 原鉴定项”属性，文档没有保证它与单个
框对应。若后续要训练逐个动物的性别/年龄预测，需要先获得可靠的属性与框
关联；现在先完整保存属性，不能把一个物种的某个属性套给所有同物种动物。

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
- `raw/annotations/.../current.json` 汇总目前观察到的物种、全部鉴定属性及最终框；同目录
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

### 已下载媒体只补标注

沿用 `/mnt/data2/pbiomon_yolo` 的原 SQLite 台账；新建空目录无法复用原媒体关联。
首次补全本轮历史范围执行：

```bash
python3 /mnt/data2/pbiomon_yolo/bin/pbiomon_sync.py run \
  --root /mnt/data2/pbiomon_yolo \
  --mode history --end-date 2026-11-01 --new-scan \
  --annotations-only --detach
```

这里保留原截止日，且不指定开始日，重查截至 2026-11-01 的全部历史鉴定。
接口按**首次鉴定完成时间**筛选，没有属性修改时间或按资源 ID 批量查询接口；
只查 2026-11-02 之后、使用默认 7 天重叠窗口，或者接着旧扫描的末尾游标，
都不能补齐更早记录中本次新增的属性。若历史归档范围已扩大，使用实际需要
回查的截止日，不把上面日期改成属性修改日期。

本模式只请求鉴定 API、更新原始页与标注台账并输出 JSON：

- 按 `(resourceType, resourceId, speciesId)` 更新当前标注，已有版本保留；
  重跑不会将相同属性反复追加进数组。
- 保持已有媒体文件、下载回执、路径、SHA256、完成时间和队列状态。
  不启动媒体下载线程，不做媒体修复/重新校验，也不因链接变化重试 404。
- 范围内新发现的资源同样保留其标注，媒体记为 `not_requested`，不请求媒体。
  下次普通下载扫描重新遇到它们时，才按下载类型加入媒体队列。
- 不能与 `--retry-failed`、`--recheck-unavailable` 或非零 `--max-downloads` 混用。
  同一根目录仍使用全局进程锁，不能与普通下载同时运行；也不能在恢复同一个
  未完成扫描时静默切换媒体模式。
- 需交互输入 API 密钥，原有媒体不需要再次传输。

暂停后恢复同一个标注扫描时，使用相同日期和 `--annotations-only`，去掉
`--new-scan`；完成后确需再次回查，再加 `--new-scan` 从各分片第一页开始。
若已有另一个未完成的目录扫描，先恢复完成它，不能强行跳过推进水位。

`status` 中应看到 `annotations_only=true`、`download_workers=0`、
`completed_downloads_this_process=0`。扫描完成且标注 JSON 全部输出后，
`annotation_files_pending=0`；本模式的 `run.status` 只表示本次标注扫描结果，
全局 `resources` 中以前的媒体失败或待下载项仍如实显示。
`complete` 也不证明线上已升级为 v1.4.0 或全部属性都有值，需另行核对字段。

### 常规下载（默认带全部标注）

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
后续普通下载持续按上面的完整字段约定保存全部标注；历史属性初次补齐后，
更早记录的新修订仍需定期按历史范围回查，普通增量不保证覆盖这些修订。

```bash
python3 /mnt/data2/pbiomon_yolo/bin/pbiomon_sync.py status --root /mnt/data2/pbiomon_yolo
python3 /mnt/data2/pbiomon_yolo/bin/pbiomon_sync.py verify --root /mnt/data2/pbiomon_yolo --limit 1000 --hash
python3 /mnt/data2/pbiomon_yolo/bin/pbiomon_sync.py stop --root /mnt/data2/pbiomon_yolo
```

`stop` 请求优雅退出，保留查询进度、回执、已完成文件和未完成任务。

### 补标注验收

先确认线上 API 已返回 `attributes[]`，再运行历史扫描。扫描结束后，
对已下载媒体的标注可用以下只读检查区分“字段尚未补到”“字段全空”和“已有值”；
不输出媒体地址或个人信息，也不读取媒体内容：

```bash
python3 - <<'PY'
import collections
import json
import sqlite3

fields = ('gender', 'age', 'postureBehavior', 'generalBehavior',
          'breedingBehavior', 'groupingBehavior', 'photoQuality')
db = sqlite3.connect('file:/mnt/data2/pbiomon_yolo/state/state.sqlite?mode=ro', uri=True)
counts = collections.Counter()
for (raw,) in db.execute('''
    SELECT a.annotation_json FROM annotations a
    JOIN resources r USING(resource_type, resource_id) WHERE r.status='done'
'''):
    counts['downloaded_media_species_records'] += 1
    final = json.loads(raw)['finalIdentification']
    if 'attributes' not in final:
        counts['missing_attributes_field'] += 1
        continue
    items = final['attributes']
    if (not isinstance(items, list) or not items or
        any(not isinstance(item, dict) or any(key not in item or
            (item[key] is not None and (not isinstance(item[key], str) or not item[key].strip()))
            for key in fields) for item in items)):
        counts['invalid_attributes_structure'] += 1
        continue
    counts['records_with_attributes'] += 1
    has_value = any(item[key] is not None for item in items for key in fields)
    counts['records_with_values' if has_value else 'records_all_attributes_null'] += 1
db.close()
print(json.dumps(dict(counts), ensure_ascii=False, indent=2))
PY
```

输出计数的单位是“资源 × 物种”记录，不是文件数；未出现的计数项视为 0。
若仍有缺失，核对原始响应与扫描范围：线上未返回、历史记录已被撤销或不再
返回等情况不能在本地推测补齐。全空属性合法，不能为了验收而填造性别或年龄。
本次功能测试使用临时数据；部署源码和测试通过不代表生产历史标注已补完。

## 验证

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_pbiomon_sync.py -v
```

测试只使用临时目录和构造记录，覆盖跨页归并、并发领取、事务回滚、回执恢复、
已完成文件复用、原始坐标保留以及内网路由约束。线上小范围试运行和正式下载
结果保存在服务器 `reports/`。不运行仓库现有会修改或删除数据的清理脚本。
新增检查覆盖 v1.4 多项/重复/全空属性、旧记录原位补字段、图片和视频文件及
回执不变、所有已有媒体队列状态不变、只补标注模式无媒体下载器或工作线程、
有旧媒体积压时标注扫描仍能结束，以及恢复扫描时的模式冲突保护。
同一组最终图片/视频标注通过普通下载或历史补标注后，资源关联路径、完整
标注内容和汇总结构保持一致；原始源页和观察时间按各自采集过程如实记录。
