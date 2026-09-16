# GRPO 指标图解手册

打开 [index.html](index.html)，或下载这个单独文件后用浏览器打开。无需安装依赖或登录；所有曲线、教程、样式、交互与源码片段均已内嵌，可离线阅读。

针对 SwanLab run yyq90/GRPO-Qwen3.8-Smoke/psmh3kg8，包含 92 项训练指标和 12 项系统指标。当前训练数据截至 step 73；准确抓取时间见页面顶部。系统指标独立按时间采样。

## 功能

中文及英文指标搜索、九类导航、12 项入门指标、逐点查询、五点平滑、含零纵轴、GRPO 优势模拟、三类 KL 对照、组合诊断、源码片段、逐项 CSV 与整体 JSON 导出、打印。

## 文件与重建

- index.html：直接交付的独立页面。
- snapshot.json：公开 API 数据与配置快照。
- catalog.py：104 项中文定义、读法、误区和来源。
- template.html：前端模板。
- fetch_snapshot.py：只读获取公开实验，不登录，不修改远端。
- build.py：检查指标和引用覆盖，再生成 HTML。

在本目录运行 python3 fetch_snapshot.py，然后 python3 build.py 可更新快照与重建。仅使用 Python 标准库。构建会读取当前工作区及 catalog.py 所列本次实验的 reward、agent、SwanLab SDK 源码；其他机器直接打开 HTML 即可，无需这些源码。若项目新增指标，构建会因缺少对应解释而停止，需先补充词典。

可选本地服务：python3 -m http.server 8765 --bind 127.0.0.1
浏览器访问 http://localhost:8765。远程机器需转发端口，或直接下载 HTML 后双击。

## 口径与限制

指标清单来自项目范围，并已逐项确认该 run 有非空数据。曲线是本次 API 返回的快照，可能经过服务端采样；抓取不是原子操作，各指标末尾时间可能不同。缺失值不补零。

运行元信息没有 Git commit；当前本地源码不能视为已证明与运行时代码完全一致。HTML 内嵌实际读取的源码片段及其文件 SHA-256，便于核对。奖励衡量参考答案一致性，不是经真实 API 验证的业务成功率。

该页面由 AI 辅助制作，未创建 PR 或提交。

## 内嵌字体

包含 Noto Sans SC 的 WOFF2 字形子集，来源 https://github.com/google/fonts/tree/main/ofl/notosanssc ，采用 SIL Open Font License 1.1，完整许可证在 FONT-LICENSE.txt。该字体已内嵌 HTML，离线显示不依赖外部字体服务。后续加入新的汉字时，应重新生成字形子集。
