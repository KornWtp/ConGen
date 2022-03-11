#!/bin/bash

CUDA_VISIBLE_DEVICES=$1
python src/main.py \
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