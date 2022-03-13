# ConGen
This repository contains codes and scripts for the paper ConGen: Unsupervised Control and Generalization Distillation For Sentence Representation.

## Installation
```
pip install -e .
``` 

## Usage
### Train
```bash
>> bash train_congen.sh
```

### Evaluation
Our evaluation code for sentence embeddings is based on a modified version of [SentEval](https://github.com/facebookresearch/SentEval) and [SimCSE](https://github.com/princeton-nlp/SimCSE).

Before evaluation, please download the evaluation datasets by running
```bash
cd SentEval/data/downstream/
bash download_dataset.sh
```

Then come back to the root directory, you can evaluate any `sentence transformers` models using SimCSE evaluation code. For example,
```bash
python evaluation.py \
    --model_name_or_path "your-model-path" \
    --task_set sts \
    --mode test
```
