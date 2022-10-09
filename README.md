# ConGen
This repository contains codes and scripts for the paper ConGen: Unsupervised Control and Generalization Distillation For Sentence Representation (Finding of EMNLP 2022).

## Installation
```
pip install -e .
``` 

## Models in paper (Small to Large)
- [ConGen-BERT-Tiny]()
- [ConGen-BERT-Mini]()
- [ConGen-TinyBERT-L4]()
- [ConGen-MiniLM-L3]()
- [ConGen-MiniLM-L6]()
- [ConGen-BERT-Small]()
- [ConGen-MiniLM-L12]()
- [ConGen-TinyBERT-L6]()
- [ConGen-BERT-base]()
- [ConGen-RoBERTa-base]()
- [ConGen-Multilingual-DistilBERT]()
- [ConGen-Multilingual-MiniLM-L12]()

## Usage
### Training data
We use the training data from [BSL](https://drive.google.com/file/d/19O2NArJz_RlVNNGRbBnnWxNMW-7HaFZ8/view?usp=sharing) (Access is requested). 

### Development data
We use sts-b development set from [sentence transformer](https://sbert.net/datasets/stsbenchmark.tsv.gz).

### Parameters
The full model parameters:



For finetuning model parameters: 
```
learning_rate_all=(3e-4 5e-4 1e-4 1e-5)
queue_sizes=(262144 131072 65536 16384 1024)
teacher_temps=(0.01 0.03 0.05 0.07 0.09)
student_temps=(0.01 0.03 0.05 0.07 0.09)
```

### Train
Please set the model parameter before training. 
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
