# Repository Guidelines

## 项目结构与模块组织
- 入口：`main.py`（启动 Tornado，端口 8888）、`server.py`（WebSocket/HTTP 路由与处理）。
- 核心：`associate/`（多视角关联与告警）、`components/`（YOLO加载、跟踪、日志）、`entries/`（`DetectionEntry`、`AssociateResult` 等数据结构）。
- 配置与资源：
  - `constant.py`（全局常量）
  - `config/drowning.yaml`（溺水检测规则与阈值）
  - `config/tracker.yaml`（滑动窗口与生命周期配置）
  - `models/`（YOLO 权重）、`video/`（本地媒体）、`logs/`、`track_logs/`
- 客户端脚本：`client_test.py`、`alert_client_test.py`、`associate_client_test.py`（手动验证 WebSocket）。
- 第三方：跟踪相关实现位于 `boxmot/`。

## 构建、测试与开发命令
- 创建环境
  - `python -m venv venv && source venv/bin/activate`（Windows: `venv\Scripts\activate`）
  - `pip install -r requirements.txt`
  - 可选：CUDA 检查 `python cuda_test.py`
- 运行服务
  - `python main.py`
  - 数据源管理（HTTP）：
    - 查询：`curl -s localhost:8888/source`
    - 新增：`curl -X PUT -H 'Content-Type: application/json' -d '{"key":"cam1","source":"video/sample.mp4"}' localhost:8888/source`
    - 删除：`curl -X DELETE -H 'Content-Type: application/json' -d '{"key":"cam1"}' localhost:8888/source`
- 冒烟测试
  - 视频流：连接 `ws://localhost:8888/video`
  - 融合/告警：`python client_test.py` / `python alert_client_test.py`

## 代码风格与命名规范
- Python 3，PEP 8，4 空格缩进，尽量使用类型注解。
- 模块/函数使用 `snake_case`；类使用 `CamelCase`；常量使用 `UPPER_SNAKE`（见 `constant.py`）。
- 日志：使用 `components/logger.py` 的 `setup_logger`；库代码避免直接 `print`。
- 路径尽量相对仓库根；`components/yolo_model.py` 默认从 `models/` 加载并 `.to('cuda')`。

## 溺水规则配置（config/drowning.yaml）
- 位置：`config/drowning.yaml`
- 读取与覆盖：`components/drowning_detector.py:71` 加载 YAML；实例化参数如 `fps`、`use_original_bbox` 可覆盖配置。
- 触发判定：`required_rule_count` 条规则触发，或（启用时）掉落事件单独触发（`enable_drop_event_single`）。
- 关键阈值：
  - `thresholds.underwater_streak_s`（规则4，连续水下时长，秒）
  - `thresholds.head_underwater_ratio`（规则1，水下占比）
  - `thresholds.velocity.mean_v_lt/mean_a_lt`（规则2，速度/加速度下限）
  - 其它见文件内中文注释。
- 误报调优建议：
  - 提高 `thresholds.underwater_streak_s`（已默认 5.0）
  - 提高 `thresholds.head_underwater_ratio`（如 0.8）
  - 提高 `required_rule_count`（如 5）
  - 增加 `min_observations`、`min_duration_s` 以过滤短暂波动
  - 必要时将 `enable_drop_event_single` 置为 `false`

## 滑动窗口配置（config/tracker.yaml）
- 位置：`config/tracker.yaml`
- 加载：`components/track_lifecycle.py:102` 初始化时读取。
- 关键项：
  - `max_window_size`（窗口最大观测条数）
  - `frame_window_span`（仅保留最近 N 帧，留空禁用帧裁剪）
  - `ttl`（窗口超时秒）、`snapshot_interval`（快照间隔秒）、`save_dir`（日志目录）
- 调参提示：
  - 若“连续水下”误报多，可适度减小 `frame_window_span` 或 `max_window_size`，限制统计时间范围。

## 告警与节流
- 溺水检测调用：`associate/associate.py:132`；检测器 hook：`components/track_window_manager.py:257`。
- 告警节流：`associate/associate.py:51` 设置 `alarm_throttle_sec`（同一 track 在窗口内仅受理一次）。
- 事件链路：accept → queue → start/push → finish；日志输出路径：`logs/alert_events.jsonl`。

## 测试指南
- 当前无正式单测；根目录脚本可作冒烟测试。
- 新增测试推荐使用 `pytest`，置于 `tests/`，命名 `test_*.py`；运行 `pytest -q`。
- 涉及 GPU 的测试需注明设备前提，并尽量提供 CPU 回退路径。

## 提交与 Pull Request 指南
- 提交信息用祈使句；建议遵循 Conventional Commits（如 `feat:`、`fix:`、`refactor:`），题目≤72字符，建议包含作用域（如 `associate:`）。
- PR 应包含：变更说明、动机、复现步骤、相关配置（`constant.py`/`config/*.yaml`）与环境信息（CUDA、torch、模型文件）。
- 行为变更需同步更新文档/注释；新增端点请附示例。

## 安全与配置提示
- 勿提交大型/二进制文件：权重（`*.pt`/`*.pth`）、视频、日志；`.gitignore` 已忽略，含 `source` 与 `watch_client`。
- `server.py` 允许跨域（`check_origin=True`），对外部署前务必收紧策略。
- 无 GPU 时可将 `components/yolo_model.py` 改为 `.to('cpu')` 或按可用性切换。

## 日志与溯源工具
- 事件日志：`logs/alert_events.jsonl`（由 `components/event_logger.py` 自动写入）。
- 管理脚本：`python scripts/drownlog.py <recent|show|stats>`
  - 近期：`python scripts/drownlog.py recent -n 10`
  - 查看链路：`python scripts/drownlog.py show --event <event_id>`
  - 统计：`python scripts/drownlog.py stats --since 7d [--json]`
  - 导出报告：`python scripts/drownlog.py export --event <event_id> -o report.json --pretty`
