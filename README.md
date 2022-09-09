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

### Training data
Thai: [Thai Wikipedia](https://github.com/PyThaiNLP/ThaiWiki-clean/releases/tag/20210620?fbclid=IwAR2_CtHJ_6od9z5-0hsolwcNYJH03e5qk_XXkoxDpOQivmo8QreYFQS3JuQ).
Eval: [STS-B-TH](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/sts-test_th.csv)

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
