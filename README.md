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

其中各字段名解释如下

| Field | Description |
|---|---|
| `id` | The unique identifier of the tweet (Tweet ID). |
| `target` | The designated Twitter user (public figure) that this instance is paired with. |
| `T1_label` | The Task 1 (T1) tri-class label: `non-abusive`, `targeted-abusive`, or `unidentified-targets`. |
| `T2_label` | The Task 2 (T2) fine-grained abuse type label (only applicable when `tri_label` is `targeted-abusive`). |
| `T3_loc` |  Character-level span position of an abusive phrase in the original text, formatted as `start-end` (e.g., `25-41`). |

*Note: The dataset format we uploaded is already compatible with the code we uploaded. The only thing missing is the "text" field and the "content" field inside "T3_loc". After downloading the original text using the ID, simply fill these in by adding the text content. The "text" field should be inserted right after the "id" field, and the "context" field should be inserted in "T3_loc" right before "char_pos".* <br>


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
各文件运行代码命令如下

**01_Single_Task:** `python t1.py`  `python t2.py`  `python t3.py` <br>
**02_CASCADED-MTL:** `python MTL_Base.py`  `python CTF.py`  `python CTF+TAG.py`  `python CTF+TAG+LGSD.py` <br>
**03_zero-shot:** `python zero_t1.py`  `python zero_t2.py`  `python zero_t3.py` <br>
**04_fine-tune:** `python fine-tune_t1.py`  `python fine-tune_t2.py`  `python fine-tune_t3.py` <br>






