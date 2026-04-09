# TAMA: Target-Aware Multilingual Abuse Detection by Cascaded Conditional Multi-Task Learning
This is the official code and dataset for the paper **TAMA: Target-Aware Multilingual Abuse Detection by Cascaded Conditional Multi-Task Learning**.

## Install Dependencies
Create a conda environment using the provided .yaml file.
```
conda env create --file=TAMA.yaml
conda activate TAMA
```


## Dataset
The `data` folder contains the datasets used in our experiment, with split into 8:1:1 ratio. <br>

- **Training Set:** `data/train.jsonl` <br>
- **Dev Set:** `data/dev.jsonl` <br>
- **Test Set:** `data/test.jsonl` <br>

*Note: 'no-target.jsonl' contains posts without a specified target and is not used for training.* <br>

### Dataset description

| Field | Description |
|---|---|
| `id` | The unique identifier of the tweet (Tweet ID). |
| `target` | The designated Twitter user (public figure) that this instance is paired with. |
| `T1_label` | The Task 1 (T1) tri-class label: `non-abusive`, `targeted-abusive`, or `unidentified-targets`. |
| `T2_label` | The Task 2 (T2) fine-grained abuse type label (only applicable when `tri_label` is `targeted-abusive`). |
| `T3_loc` |  Character-level span position of an abusive phrase in the original text, formatted as `start-end` (e.g., `25-41`). |


### Cascaded-MTL Code Structure


| # | Description |
|-------|-------|
| 01 | 01 contains the experimental code for single-task fine-tuning of small models, with filenames corresponding to the code for each task. |
| 02 | 02 contains the experimental code for our proposed CASCADED-MTL multi-task framework, with filenames corresponding to the code for each module. |
| 03 | 03 contains the experimental code for single-task zero-shot evaluation with large language models, with filenames corresponding to the code for each task. |
| 04 | 04 contains the experimental code for single-task fine-tuning of large language models, with filenames corresponding to the code for each task. |


### Examples to run code:

**01_Single_Task:** `python t1.py`  `python t2.py`  `python t3.py` <br>
**02_CASCADED-MTL:** `python MTL_Base.py`  `python CTF.py`  `python CTF+TAG.py`  `python CTF+TAG+LGSD.py` <br>
**03_zero-shot:** `python zero_t1.py`  `python zero_t2.py`  `python zero_t3.py` <br>
**04_fine-tune:** `python fine-tune_t1.py`  `python fine-tune_t2.py`  `python fine-tune_t3.py` <br>



