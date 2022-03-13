import os
import argparse 
import logging
from datetime import datetime
import io
import math
import numpy as np
import random
from glob import glob 
import pickle

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import models
from sentence_transformers import LoggingHandler, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from sentence_transformers_congen import SentenceTransformer, losses

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--model_save_path",
					type=str,
					default=None,
					required=True,
					help="The output directory where the model checkpoints will be written.")
parser.add_argument("--train_data_path",
					type=str,
					default=None,
					required=True,
					help="The directory of train data.")
parser.add_argument("--dev_data_path",
					type=str,
					default=None,
					required=True,
					help="The directory of dev data.")
parser.add_argument("--teacher_model_name_or_path",
					type=str,
					default=None,
					required=True,
					help="The teacher model checkpoint for weights initialization.")
parser.add_argument("--student_model_name_or_path",
					type=str,
					default=None,
					required=True,
					help="The student model checkpoint for weights initialization.")
parser.add_argument("--train_batch_size", 
					type=int, 
					default=32,
					help="Batch size for training.")
parser.add_argument("--eval_batch_size", 
					type=int, 
					default=32,
					help="Batch size for evaluation.")
parser.add_argument("-- ",
					type=int,
					default=16,
					help="Batch size at inference.")
parser.add_argument("--max_seq_length",
					type=int,
					default=128,
					help="Student model max. lengths for inputs (number of word pieces).")							
parser.add_argument("--num_epochs",
					type=int,
					default=3,
					help="Total number of training epochs to perform.")
parser.add_argument("--learning_rate",
					type=float,
					default=5e-5,
					help="The initial learning rate for Adam.")
parser.add_argument("--student_temp",
					type=float,
					default=0.1,
					help="Temperature for student encoder.")
parser.add_argument("--teacher_temp",
					type=float,
					default=0.05,
					help="Distillation temperature.")
parser.add_argument("--queue_size",
					type=int,
					default=1000,
					help="The size of instance queue")
parser.add_argument("--gpu_device",
					type=int,
					default=0,
					help="gpu device number")
parser.add_argument("--early_stopping_patience",
					type=int,
					default=7,
					help="Early stopping criteria: patience") 
parser.add_argument("--seed",
					type=int,
					default=1000,
					help="The seed value")					
parser.add_argument("--queue_random",
					type=int,
					default=0,
					help="Random instance queue or not")	

args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

logging.info(f"Loading teacher model: {args.teacher_model_name_or_path}")
teacher_model = SentenceTransformer(args.teacher_model_name_or_path)

logging.info("Preparing training dataset")
all_pairs = open(args.train_data_path, mode="rt", encoding="utf-8").readlines()
all_pairs = [sample.strip().split('\t') for sample in all_pairs]
# Two lists of sentences
sents1 = [p[0] for p in all_pairs]
sents2 = [p[1] for p in all_pairs]


try:
	filename = open("data/sents1_encoded.pkl", "rb")
	sents1_encoded = pickle.load(filename)
	filename.close()
except:
	sents1_encoded = teacher_model.encode(sents1, convert_to_tensor=True, normalize_embeddings=True, device=device)
	filename = 'data/sents1_encoded.pkl'
	pickle.dump(sents1_encoded, open(filename, 'wb'), protocol=4)
teacher_dimension = sents1_encoded.shape[1]
logging.info(f"Teacher dimension size:{teacher_dimension}")

logging.info(f"Loading student model: {args.student_model_name_or_path}")
student_word_embedding_model = models.Transformer(args.student_model_name_or_path, max_seq_length=args.max_seq_length)
student_dimension = student_word_embedding_model.get_word_embedding_dimension()
student_pooling_model = models.Pooling(student_dimension)

if teacher_dimension != student_dimension:
	dense_model = models.Dense(in_features=student_dimension, out_features=teacher_dimension, activation_function=nn.Tanh())
	student_model = SentenceTransformer(modules=[student_word_embedding_model, student_pooling_model, dense_model])
else:
	student_model = SentenceTransformer(modules=[student_word_embedding_model, student_pooling_model])

logging.info(f"Create instance queue")
text_in_queue = np.random.RandomState(16349).choice(sents1, args.queue_size, replace=False)
train_samples = []
instance_queue = []
text_in_q_set = set(text_in_queue)
for s1, s2, s1_encoded in zip(sents1, sents2, sents1_encoded): 
	if s1 not in text_in_q_set:
		train_samples.append(InputExample(texts=[s1, s2], label=s1_encoded))
	else:
		instance_queue.append(s1)
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.train_batch_size)

instance_queue_encoded = teacher_model.encode(instance_queue, 
									convert_to_tensor=True,
									normalize_embeddings=True, 
									device=device)

training_loss = losses.ConGenLoss(instanceQ_encoded=instance_queue_encoded,  
								model=student_model,
								student_temp=args.student_temp, 
								teacher_temp=args.teacher_temp)

del instance_queue, sents1_encoded, teacher_model, instance_queue_encoded					

warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs * 0.1)  # 10% of train data for warm-up
evaluation_steps = 512

logger.info("Load evaluator for STSBenchmark") 
dev_samples = []
with io.open(args.dev_data_path, "r", encoding="utf-8") as f:
	for line in f:
		text = line.strip().split("\t")
		if text[0] == 'dev':
			sentence1 = text[6]
			sentence2 = text[7]
			score = float(text[5]) / 5.0  #Normalize score to range 0 ... 1
			dev_samples.append(InputExample(texts=[sentence1, sentence2], label=score))

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=args.eval_batch_size, name='sts-dev')

logger.info("Start training")
start = datetime.now()
student_model.fit(train_objectives=[(train_dataloader, training_loss)],
		evaluator=dev_evaluator,
		epochs=args.num_epochs,
		warmup_steps=warmup_steps,
		evaluation_steps=evaluation_steps,
		output_path=args.model_save_path,
		optimizer_params={"lr": args.learning_rate, 'eps': 1e-6, 'correct_bias': False},
		use_amp=True,
		early_stopping_patience=args.early_stopping_patience)
stop = datetime.now()
run_time = stop - start
logger.info("Training time: " + str(run_time) + " s")
