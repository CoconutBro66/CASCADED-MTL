# CASCADED-MTL


## Install Dependencies
Create a conda environment using the provided .yaml file.

## Source Code Structure
完整实验可按文件夹名中的序号进行，也可以选择执行脚本
<div>
  
| # | Description |
|-------|-------|
| 01 | 其中，01为单任务的小模型微调实验代码，文件名分别对应各任务的实验代码 |
| 02 | 02为我们提出的CASCADED-MTL多任务框架实验代码，文件名分别对应各模块的实验代码 |
| 03 | 03为单任务零样本大模型实验代码，文件名分别对应各任务的实验代码 |
| 04 | 04为单任务大模型微调实验代码，文件名分别对应各任务的实验代码 |
</div>

### Running the Experiments
各代码可在上述环境中直接运行，只需要更改模型及数据路径。

在这里看怎么使用代码
## Dataset
Training Set:
Dev Set:
Test Set:

## CASCADED-MTL
我们的CASCADED-MTL代码位于，其中MTL-Base是基础框架代码，CTF.py，CTF+TAG.py，CTF+TAG+LGSD.py分别为其同名的模块代码。

## Single Task
我们的单任务基线代码位于，t1.py，t2.py，t3.py分别代表各同名的单任务基线代码。

## Zero-shot LLMs
我们的Zero-shot代码位于，zero_t1.py，zero_t2.py，zero_t3.py分别代表各同名的单任务zero-shot代码。

## Fine-tune 
我们的Fine-tune代码位于，fine-tune_t1.py，fine-tune_t2.py，fine-tune_t3.py分别代码各同名的单任务fine-tune代码。

