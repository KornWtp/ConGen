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
| Models  | Teacher Temp | Student Temp | Queue Size | Learning Rate |   
| --------------------- | ----- | ----- | -----| ----|
|BERT-Tiny              | 0.05  | 0.05  | 16384| 5e-4|
|BERT-Mini              | 0.05  | 0.07  | 16384| 3e-4| 
|Tiny-BERT-L4           | 0.05  | 0.05  | 65536| 1e-4| 
|MiniLM-L3              | 0.05  | 0.07  | 16384| 5e-4| 
|MiniLM-L6              | 0.05  | 0.07  | 65536| 3e-4|   
|BERT-Small             | 0.05  | 0.07  | 65536| 3e-4|  
|MiniLM-L12             | 0.05  | 0.07  | 16384| 5e-5|  
|Tiny-BERT-L6           | 0.05  | 0.07  | 65536| 5e-5| 
|BERT-base              | 0.05  | 0.07  | 65536| 5e-5| 
|RoBERTa-base           |  0.1  |  0.1  |  1024| 5e-5| 
|Multilingual-DistilBERT| 0.05  | 0.07  | 65536| 3e-4|  
|Multilingual-MiniLM-L12| 0.05  | 0.07  | 65536| 3e-4|  


For finetuning model parameters: 
```
learning_rate_all=(3e-4 5e-4 1e-4 3e-5 5e-5 1e-5)
queue_sizes=(262144 131072 65536 16384 1024)
teacher_temps=(0.01 0.03 0.05 0.07 0.09 0.1)
student_temps=(0.01 0.03 0.05 0.07 0.09 0.1)
```

### Train
Please set the model's parameter before training.
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

## Results
<table class="tg">
<thead>
  <tr>
    <th class="tg-7btt">Model</th>
    <th class="tg-7btt">STS-12</th>
    <th class="tg-7btt"><span style="font-style:normal">STS-13</span></th>
    <th class="tg-7btt"><span style="font-style:normal">STS-14</span></th>
    <th class="tg-7btt"><span style="font-style:normal">STS-15</span></th>
    <th class="tg-7btt"><span style="font-style:normal">STS-16</span></th>
    <th class="tg-7btt"><span style="font-style:normal">STS-B</span></th>
    <th class="tg-7btt">SICK-R</th>
    <th class="tg-7btt">Avg.</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">BERT-Tiny</td>
    <td class="tg-c3ow">72.18</td>
    <td class="tg-nbj5"><span style="background-color:#FFF">81.12</span></td>
    <td class="tg-c3ow">75.45</td>
    <td class="tg-c3ow">83.22</td>
    <td class="tg-c3ow">77.89</td>
    <td class="tg-c3ow">79.03</td>
    <td class="tg-c3ow">69.05</td>
    <td class="tg-c3ow">76.85</td>
  </tr>
  <tr>
    <td class="tg-0pky">BERT-Mini</td>
    <td class="tg-c3ow">74.17</td>
    <td class="tg-c3ow">82.69</td>
    <td class="tg-c3ow">76.58</td>
    <td class="tg-c3ow">84.30</td>
    <td class="tg-c3ow">78.23</td>
    <td class="tg-c3ow">80.84</td>
    <td class="tg-c3ow">69.82</td>
    <td class="tg-c3ow">78.09</td>
  </tr>
  <tr>
    <td class="tg-0pky">Tiny-BERT-L4</td>
    <td class="tg-c3ow">74.3</td>
    <td class="tg-c3ow">83.07</td>
    <td class="tg-c3ow">77.37</td>
    <td class="tg-c3ow">84.70</td>
    <td class="tg-c3ow">79.06</td>
    <td class="tg-c3ow">80.99</td>
    <td class="tg-c3ow">70.26</td>
    <td class="tg-c3ow">78.54</td>
  </tr>
  <tr>
    <td class="tg-0pky">MiniLM-L3</td>
    <td class="tg-c3ow">74.00</td>
    <td class="tg-c3ow">82.93</td>
    <td class="tg-c3ow">76.58</td>
    <td class="tg-c3ow">84.35</td>
    <td class="tg-c3ow">78.57</td>
    <td class="tg-c3ow">81.00</td>
    <td class="tg-c3ow">70.09</td>
    <td class="tg-c3ow">78.22</td>
  </tr>
  <tr>
    <td class="tg-0pky"><span style="font-weight:400;font-style:normal">MiniLM-L6</span></td>
    <td class="tg-c3ow">75.06</td>
    <td class="tg-c3ow">83.86</td>
    <td class="tg-c3ow">77.29</td>
    <td class="tg-c3ow">85.01</td>
    <td class="tg-c3ow">79.67</td>
    <td class="tg-c3ow">81.92</td>
    <td class="tg-c3ow">70.89</td>
    <td class="tg-c3ow">79.10</td>
  </tr>
  <tr>
    <td class="tg-0pky">BERT-Small</td>
    <td class="tg-c3ow">74.50</td>
    <td class="tg-c3ow">83.58</td>
    <td class="tg-c3ow">77.29</td>
    <td class="tg-c3ow">84.83</td>
    <td class="tg-c3ow">79.72</td>
    <td class="tg-c3ow">81.93</td>
    <td class="tg-c3ow">70.55</td>
    <td class="tg-c3ow">78.91</td>
  </tr>
  <tr>
    <td class="tg-0pky"><span style="font-weight:400;font-style:normal">MiniLM-L12</span></td>
    <td class="tg-c3ow">75.25</td>
    <td class="tg-c3ow">84.61</td>
    <td class="tg-c3ow">78.27</td>
    <td class="tg-c3ow">85.51</td>
    <td class="tg-c3ow">80.52</td>
    <td class="tg-c3ow">82.32</td>
    <td class="tg-c3ow">71.32</td>
    <td class="tg-c3ow">79.68</td>
  </tr>
  <tr>
    <td class="tg-0pky"><span style="font-weight:400;font-style:normal">Tiny-BERT-L6</span></td>
    <td class="tg-c3ow">75.53</td>
    <td class="tg-c3ow">84.76</td>
    <td class="tg-c3ow">78.33</td>
    <td class="tg-c3ow">85.72</td>
    <td class="tg-c3ow">80.42</td>
    <td class="tg-c3ow">82.25</td>
    <td class="tg-c3ow">71.12</td>
    <td class="tg-c3ow">79.73</td>
  </tr>
  <tr>
    <td class="tg-0pky">BERT-base</td>
    <td class="tg-c3ow">75.58</td>
    <td class="tg-c3ow">85.13</td>
    <td class="tg-c3ow">78.54</td>
    <td class="tg-c3ow">85.75</td>
    <td class="tg-c3ow">81.12</td>
    <td class="tg-c3ow">82.81</td>
    <td class="tg-c3ow">71.47</td>
    <td class="tg-c3ow">80.06</td>
  </tr>
  <tr>
    <td class="tg-0pky">RoBERTa-base</td>
    <td class="tg-c3ow">75.32</td>
    <td class="tg-c3ow">84.56</td>
    <td class="tg-c3ow">77.26</td>
    <td class="tg-c3ow">85.33</td>
    <td class="tg-c3ow">81.34</td>
    <td class="tg-c3ow">82.67</td>
    <td class="tg-c3ow">72.00</td>
    <td class="tg-c3ow">79.78</td>
  </tr>
</tbody>
</table>
