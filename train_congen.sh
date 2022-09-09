#!/bin/bash

# please see Appendix A.1 for the full setup
CUDA_VISIBLE_DEVICES=$1 # GPU device number
python main.py \
    --model_save_path "your-output-model-path" \
    --teacher_model_name_or_path princeton-nlp/unsup-simcse-roberta-large \ # Default teacher For Thai!! please use "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    --student_model_name_or_path nreimers/BERT-Tiny_L-2_H-128_A-2 \ # compressed model or large model or WangchanBERTa
    --train_data_path "your-train-data-path" \ # attached link
    --dev_data_path "your-validation-data-path" \ # STS-B dev set
    --train_batch_size 128 \
    --eval_batch_size 128 \
    --max_seq_length 128 \
    --num_epochs 20 \
    --learning_rate 5e-4 \ # see Appendix A.1
    --teacher_temp 0.05 \ # see Appendix A.1
    --student_temp 0.07 \ # see Appendix A.1
    --queue_size 65536 # see Appendix A.1
