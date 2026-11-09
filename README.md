# TAMA：基于级联条件多任务学习的目标感知多语言辱骂检测

本仓库提供论文《TAMA: Target-Aware Multilingual Abuse Detection by Cascaded Conditional Multi-Task Learning》的官方代码与数据集。

## 安装依赖

使用提供的 YAML 文件创建并激活 Conda 环境：

```bash
conda env create --file=TAMA.yaml
conda activate TAMA
```

## 数据集

`data` 文件夹包含实验使用的数据集，按照 8:1:1 的比例划分为训练集、验证集和测试集。

- 训练集：`data/train.jsonl`
- 验证集：`data/dev.jsonl`
- 测试集：`data/test.jsonl`

注意：`no-target.jsonl` 包含未指定目标对象的帖子，不用于训练。

### 数据字段说明

| 字段 | 说明 |
| :--- | :--- |
| `id` | 推文的唯一标识符，即 Tweet ID。 |
| `target` | 与该样本配对的指定 Twitter 用户，即作为目标对象的公众人物。 |
| `T1_label` | 任务一（T1）的三分类标签：`non-abusive`（非辱骂）、`targeted-abusive`（针对指定目标的辱骂）或 `unidentified-targets`（目标不明确）。 |
| `T2_label` | 任务二（T2）的细粒度辱骂类型标签，仅在 `T1_label` 为 `targeted-abusive` 时适用。 |
| `T3_loc` | 辱骂短语在原始文本中的字符级位置区间，格式为 `start-end`，例如 `25-41`。 |

## Cascaded-MTL 代码结构

| 目录 | 说明 |
| :--- | :--- |
| `01_Single_Task` | 小模型单任务微调的实验代码，各文件名对应相应任务。 |
| `02_CASCADED-MTL` | 本文提出的 CASCADED-MTL 多任务框架实验代码，各文件名对应相应模块。 |
| `03_zero-shot` | 大语言模型单任务零样本评估的实验代码，各文件名对应相应任务。 |
| `04_fine-tune` | 大语言模型单任务微调的实验代码，各文件名对应相应任务。 |

## 运行示例

### 小模型单任务微调

在 `01_Single_Task` 目录下运行：

```bash
python t1.py
python t2.py
python t3.py
```

### 级联多任务学习框架

在 `02_CASCADED-MTL` 目录下运行：

```bash
python MTL_Base.py
python CTF.py
python CTF+TAG.py
python CTF+TAG+LGSD.py
```

### 大语言模型零样本评估

在 `03_zero-shot` 目录下运行：

```bash
python zero_t1.py
python zero_t2.py
python zero_t3.py
```

### 大语言模型单任务微调

在 `04_fine-tune` 目录下运行：

```bash
python fine-tune_t1.py
python fine-tune_t2.py
python fine-tune_t3.py
```

注：原文在 `T2_label` 的说明中使用了 `tri_label`，这里根据字段表统一为 `T1_label`，请以实际数据字段为准。
