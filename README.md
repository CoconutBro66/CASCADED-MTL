# TAMA: Target-Aware Multilingual Abuse Detection by Cascaded Conditional Multi-Task Learning


## Install Dependencies
Create a conda environment using the provided .yaml file.
```
conda env create --file=TAMA.yaml
conda activate TAMA
```


## Dataset
data文件夹中包含了我们实验所用到的数据集，其中`no-target.jsonl`为无指定目标数据集，不用于训练。<br>
所有数据集已经按8:1:1的比例划分好：
- **Training Set:** `data/train.jsonl` <br>
- **Dev Set:** `data/dev.jsonl` <br>
- **Test Set:** `data/test.jsonl` <br>

## Source Code Structure
完整实验可按文件夹名中的序号进行，也可以选择执行脚本
<div>
  
| # | Description |
|-------|-------|
| 01 | 01为单任务的小模型微调实验代码，文件名分别对应各任务的实验代码 |
| 02 | 02为我们提出的CASCADED-MTL多任务框架实验代码，文件名分别对应各模块的实验代码 |
| 03 | 03为单任务零样本大模型实验代码，文件名分别对应各任务的实验代码 |
| 04 | 04为单任务大模型微调实验代码，文件名分别对应各任务的实验代码 |
</div>

### Running the Experiments
各代码可在上述环境中直接运行，只需要更改模型及数据路径。
代码运行结束后会保存相应的测试集评估结果。


