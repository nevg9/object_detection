# **object_detection**

## 需要安装的一些包

1. 安装 ultralytics包
https://github.com/ultralytics/ultralytics

## Pbiomon 图片、视频及完整标注归档

使用 `tools/pbiomon_sync.py`，部署与运行方法见 [下载说明](docs/pbiomon-download.md)。
默认同时下载图片、原视频及接口返回的全部标注，包括性别、年龄、行为、质量、
最终物种、全部框、备注和 AI 信息。历史媒体补标注可用 `--annotations-only`，
沿用原台账并重扫原鉴定日期范围，不重新下载媒体。
