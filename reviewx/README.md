# ReviewX

ReviewX 用于审核 LabelMe 实例分割标注，组合确定性规则与 SAM 3 预测结果，输出可追溯的问题清单和可视化证据。工具只读原始数据，不修改图片或 JSON。

## 审核能力

### 确定性检查

- 图片与 JSON 配对、文件名及尺寸一致性；
- LabelMe 结构、类别名称和多边形合法性；
- 空掩码、微小多边形及越界坐标；
- 同类别重复实例；
- 不同类别使用近似相同的掩码。

### SAM 3 交叉检查

| 能力 | 判定依据 | 问题类型 |
|---|---|---|
| 漏标候选 | 模型目标未匹配到人工实例，且满足置信度与最小面积要求 | 疑似漏标 |
| 掩码偏差 | 同类别人工掩码与模型掩码的 IoU 低于阈值 | 人工掩码与模型掩码差异较大 |
| 类别冲突 | 人工实例未匹配同类别结果，却与其他已审核类别高度重合 | 疑似类别错误 |
| 错误拆分 | 一个模型目标覆盖多个同类别人工实例 | 疑似同一目标被拆成多个实例 |
| 错误合并 | 一个人工实例覆盖多个同类别模型目标 | 疑似多个目标被合成一个实例 |

模型检查仅覆盖 `model_audit: true` 的类别，结果统一标记为 `REVIEW`。SAM 3 不负责解释业务语义，也不能替代人工验收标准。

## 数据约定

数据目录采用 LabelMe 常见布局，图片与 JSON 位于同一相对目录且文件名主干一致：

```text
dataset/
  batch_a/
    image_001.jpg
    image_001.json
```

当前解析要求：

- 标注位于 `shapes`；
- 类别名称位于 `shapes[].label`，区分大小写；
- `shape_type` 为 `polygon`；
- 多边形至少包含 3 个有效坐标点。

## 快速使用

修改 `configs/reviewx.example.json` 中的数据路径、类别和提示词。全量确定性检查：

```bash
python reviewx.py \
  --config configs/reviewx.example.json \
  --static-only
```

抽样执行确定性检查和 SAM 3 检查：

```bash
python reviewx.py \
  --config configs/reviewx.example.json \
  --max-images 100
```

确认提示词、阈值和误报水平后执行全量检查：

```bash
python reviewx.py --config configs/reviewx.example.json
```

`--max-images N` 按路径排序截取前 N 张。增加 `--no-visualizations` 可只生成 JSON、CSV 和 HTML 文本报告。

## 配置

项目提供两份配置：

- `configs/reviewx.example.json`：包含 6 个规范示例类别的通用模板；
- `configs/reviewx.1009.json`：1009 数据集的完整业务配置。

### 全局参数

| 参数 | 含义 | 示例值 |
|---|---|---:|
| `dataset_dir` | LabelMe 数据集根目录 | `../data/labelme_dataset` |
| `output_dir` | 报告和缓存目录 | `../runs/reviewx/example` |
| `checkpoint` | SAM 3 权重路径 | `../assets/models/sam3.pt` |
| `device` | 推理设备 | `cuda` |
| `resolution` | 模型输入分辨率 | `1008` |
| `min_polygon_area` | 最小多边形面积，单位为像素 | `4.0` |
| `duplicate_iou` | 重复或跨类别近似掩码阈值 | `0.9` |
| `match_iou` | 人工实例与模型实例的匹配阈值 | `0.5` |
| `low_mask_iou` | 掩码偏差告警阈值 | `0.65` |
| `missing_min_area_ratio` | 漏标候选的最小图像面积占比 | `0.0002` |
| `visualize_pass` | 是否为无问题图片生成整图可视化 | `false` |

相对路径以配置文件所在目录为基准解析。

### 类别参数

```json
{
  "name": "person",
  "prompts": ["person", "pedestrian"],
  "model_audit": true,
  "confidence_threshold": 0.60
}
```

| 参数 | 含义 |
|---|---|
| `name` | 与 LabelMe `label` 完全一致的类别名称 |
| `prompts` | 独立执行的 SAM 3 文本提示词列表 |
| `model_audit` | 是否为该类别启用模型交叉检查 |
| `confidence_threshold` | 该类别模型候选的最低置信度 |

多个提示词的结果会按掩码 IoU 去重。业务定义强、视觉定义弱的类别不宜直接启用模型审核，例如可通行区域、细小杂物或依赖场景规则的类别。

## 输出

每次运行创建独立时间戳目录；SAM 3 预测缓存在数据集输出目录下：

```text
runs/reviewx/<dataset>/
  cache/<config_signature>/
  <timestamp>/
    index.html
    summary.json
    report.json
    images.csv
    issues.csv
    visualizations/
      <relative_image_dir>/<image_stem>/
        overview.jpg
        issue_001.jpg
```

| 文件 | 内容 |
|---|---|
| `index.html` | 可按状态、问题类型和图片路径筛选的审核页面 |
| `summary.json` | 数据规模、状态分布、问题分布和实际检查范围 |
| `report.json` | 图片级审核结果及完整问题字段 |
| `images.csv` | 每张图片的状态、问题类型和可视化路径 |
| `issues.csv` | 每个问题的类别、原因、指标和证据图路径 |
| `overview.jpg` | 原图尺度的问题定位图 |
| `issue_*.jpg` | 单问题局部证据图 |

建议以 `index.html` 作为人工复核入口，以 `issues.csv` 进行任务分派或二次处理。

## 状态语义

| 状态 | 含义 |
|---|---|
| `PASS` | 已执行的检查未发现问题 |
| `REVIEW` | 存在重复、冲突或模型候选，需要人工判断 |
| `FAIL` | 存在格式、配对、尺寸或几何错误 |

状态优先级为 `FAIL > REVIEW > PASS`。`PASS` 只覆盖报告中列出的检查项和启用模型审核的类别，不代表整张图片在业务语义上绝对正确。

## 证据图

证据图根据问题类型自适应布局：

- 模型对比：原图、人工掩码、SAM 3 掩码和差异图；
- 人工标注冲突：两个类别掩码及重叠区域；
- 单掩码或格式问题：仅保留相关面板。

颜色约定：

| 颜色 | 含义 |
|---|---|
| 黄色框 | 问题区域 |
| 洋红色 | SAM 3 掩码或仅模型覆盖区域 |
| 绿色 | 人工与模型一致区域 |
| 红色 | 仅人工覆盖区域，或第一人工掩码 |
| 蓝色 | 第二人工掩码 |
| 黄色填充 | 两个人工掩码的重叠区域 |

图中使用类别标签，不使用类别 ID 或人工实例编号。

## 推荐流程

1. 运行 `--static-only`，优先修复 `FAIL`；
2. 使用 `--max-images` 评估提示词、置信度和误报；
3. 固化配置后运行全量 SAM 3 检查；
4. 在 `index.html` 中复核 `REVIEW`；
5. 修订标注后重新运行确定性检查。

ReviewX 不自动修改原始标注。

## 缓存

缓存签名由权重文件、推理分辨率、类别提示词和置信度配置共同决定。配置不变时重复运行会复用预测；修改上述任一项会自动创建新的缓存空间。删除 `cache/` 只会触发重新推理，不影响报告或原始数据。
