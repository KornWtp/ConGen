# ConGen
This repository contains codes and scripts for the paper ConGen: Unsupervised Control and Generalization Distillation For Sentence Representation.

## Installation
```
pip install -e .
``` 

## Usage
### Train
For example,
```bash
python main.py \
    --model_save_path "your-output-model-path" \
    --teacher_model_name_or_path princeton-nlp/unsup-simcse-roberta-large \
    --student_model_name_or_path nreimers/BERT-Tiny_L-2_H-128_A-2 \
    --train_dataset_path "your-train-data-path" \
    --dev_dataset_path "your-validation-data-path" \
    --train_batch_size 128 \
    --inference_batch_size 128 \
    --eval_batch_size 128 \
    --max_seq_length 128 \
    --num_epochs 20 \
    --learning_rate 5e-5 \
    --teacher_temp 0.1 \
    --student_temp 0.09 \
    --queue_size 16384 
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