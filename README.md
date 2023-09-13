# ConGen
Implementation of [ConGen: Unsupervised Control and Generalization Distillation For Sentence Representation (Finding of EMNLP 2022)](https://aclanthology.org/2022.findings-emnlp.483.pdf).

## Citation
```
@inproceedings{limkonchotiwat-etal-2022-congen,
    title = "{ConGen}: Unsupervised Control and Generalization Distillation For Sentence Representation",
    author = "Limkonchotiwat, Peerat  and
      Ponwitayarat, Wuttikorn  and
      Lowphansirikul, Lalita and
      Udomcharoenchaikit, Can  and
      Chuangsuwanich, Ekapol  and
      Nutanong, Sarana",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    year = "2022",
    publisher = "Association for Computational Linguistics",
}
```

## Announcement (2023)
- We have a new version of ConGen: [SCT](https://github.com/mrpeerat/SCT/tree/main) (published at TACL2023).
- The SCT method outperforms ConGen on distillation settings.
- This method is also effective for a small model to learn sentence embedding without the teacher model!  

## Installation
```
git clone https://github.com/KornWtp/ConGen.git
cd ConGen
pip install -e .
``` 

## Our models (Small to Large)
- [ConGen-BERT-Tiny](https://huggingface.co/kornwtp/ConGen-BERT-Tiny)
- [ConGen-BERT-Mini](https://huggingface.co/kornwtp/ConGen-BERT-Mini)
- [ConGen-TinyBERT-L4](https://huggingface.co/kornwtp/ConGen-TinyBERT-L4)
- [ConGen-MiniLM-L3](https://huggingface.co/kornwtp/ConGen-MiniLM-L3)
- [ConGen-MiniLM-L6](https://huggingface.co/kornwtp/ConGen-MiniLM-L6)
- [ConGen-BERT-Small](https://huggingface.co/kornwtp/ConGen-BERT-Small)
- [ConGen-MiniLM-L12](https://huggingface.co/kornwtp/ConGen-MiniLM-L12)
- [ConGen-TinyBERT-L6](https://huggingface.co/kornwtp/ConGen-TinyBERT-L6)
- [ConGen-BERT-base](https://huggingface.co/kornwtp/ConGen-BERT-base)
- [ConGen-RoBERTa-base](https://huggingface.co/kornwtp/ConGen-RoBERTa-base)
- [ConGen-Multilingual-DistilBERT](https://huggingface.co/kornwtp/ConGen-Multilingual-DistilBERT)
- [ConGen-Multilingual-MiniLM-L12](https://huggingface.co/kornwtp/ConGen-Multilingual-MiniLM-L12)

## Usage
### Training data
We use the training data from [BSL's paper](https://aclanthology.org/2021.acl-long.402.pdf): [monolingual version](https://drive.google.com/file/d/1HeqsEChDr7i_kxbdJvmVaRMSFKDRnFBY/view?usp=sharing)
and [multilingual version](https://drive.google.com/file/d/19O2NArJz_RlVNNGRbBnnWxNMW-7HaFZ8/view?usp=sharing). 

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


### Train your own model
Please set the model's parameter before training.
```bash
>> bash train_congen.sh
```

For finetuning model parameters: 
```
learning_rate_all=(3e-4 5e-4 1e-4 3e-5 5e-5 1e-5)
queue_sizes=(262144 131072 65536 16384 1024)
teacher_temps=(0.01 0.03 0.05 0.07 0.09 0.1)
student_temps=(0.01 0.03 0.05 0.07 0.09 0.1)
```

# Evaluation
Our evaluation code for sentence embeddings is based on a modified version of [SentEval](https://github.com/facebookresearch/SentEval) and [SimCSE](https://github.com/princeton-nlp/SimCSE).

Before evaluation, please download the evaluation datasets by running
```bash
cd SentEval/data/downstream/
bash download_dataset.sh
```

## Evaluation - Notebook
Please see https://github.com/KornWtp/ConGen/tree/main/notebook

## Evaluation - Python
Then come back to the root directory, you can evaluate any `sentence transformers` models using SimCSE evaluation code. For example,
```bash
python evaluation.py \
    --model_name_or_path "your-model-path" \
    --task_set sts \
    --mode test
```

## Main results - STS
In our paper, we average score over three models and shown as follows:
<table class="tg">
<thead>
  <tr>
    <th class="tg-7btt" rowspan="2">Methods</th>
    <th class="tg-7btt" colspan="10">Semantic Textual Similarity (STS) average scores</th>
  </tr>
  <tr>
    <th class="tg-7btt">BERT<br>Tiny</th>
    <th class="tg-7btt"><span style="font-weight:700;font-style:normal">BERT</span><br><span style="font-weight:700;font-style:normal">Mini</span></th>
    <th class="tg-7btt"><span style="font-weight:700;font-style:normal">Tiny</span><br><span style="font-weight:700;font-style:normal">BERT-L4</span></th>
    <th class="tg-7btt"><span style="font-weight:700;font-style:normal">MiniLM</span><br><span style="font-weight:700;font-style:normal">L3</span></th>
    <th class="tg-7btt"><span style="font-weight:700;font-style:normal">MiniLM</span><br><span style="font-weight:700;font-style:normal">L6</span></th>
    <th class="tg-7btt"><span style="font-weight:700;font-style:normal">BERT</span><br>Small</th>
    <th class="tg-7btt"><span style="font-weight:700;font-style:normal">MiniLM</span><br><span style="font-weight:700;font-style:normal">L12</span></th>
    <th class="tg-7btt"><span style="font-weight:700;font-style:normal">Tiny</span><br><span style="font-weight:700;font-style:normal">BERT-L6</span></th>
    <th class="tg-7btt"><span style="font-weight:700;font-style:normal">BERT</span><br><span style="font-weight:700;font-style:normal">Base</span></th>
    <th class="tg-7btt"><span style="font-weight:700;font-style:normal">RoBERTa</span><br><span style="font-weight:700;font-style:normal">Base</span></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">#Param (M)</td>
    <td class="tg-c3ow">4</td>
    <td class="tg-c3ow">11</td>
    <td class="tg-c3ow">14</td>
    <td class="tg-c3ow">17</td>
    <td class="tg-c3ow">22</td>
    <td class="tg-c3ow">29</td>
    <td class="tg-c3ow">33</td>
    <td class="tg-c3ow">67</td>
    <td class="tg-c3ow">109</td>
    <td class="tg-c3ow">125</td>
  </tr>
  <tr>
    <td class="tg-fymr" colspan="11"><span style="font-style:normal">Finetuning-based</span></td>
  </tr>
  <tr>
    <td class="tg-fymr">Teacher</td>
    <td class="tg-f8tv" colspan="10"><span style="font-weight:400">SimCSE-Unsup-RoBERTa-large: 78.90</td>
  </tr>
  <tr>
    <td class="tg-0pky">Sup-SimCSE</td>
    <td class="tg-8bgf">72.35</td>
    <td class="tg-8bgf">76.52</td>
    <td class="tg-8bgf">78.19</td>
    <td class="tg-8bgf">76.49</td>
    <td class="tg-8bgf">78.86</td>
    <td class="tg-8bgf">78.59</td>
    <td class="tg-8bgf">80.48</td>
    <td class="tg-8bgf">81.23</td>
    <td class="tg-8bgf">81.57</td>
    <td class="tg-8bgf">82.52</td>
  </tr>
  <tr>
    <td class="tg-0pky">Unsup-SimCSE</td>
    <td class="tg-c3ow">64.47</td>
    <td class="tg-c3ow">65.94</td>
    <td class="tg-c3ow">67.91</td>
    <td class="tg-c3ow">55.10</td>
    <td class="tg-c3ow">59.15</td>
    <td class="tg-c3ow">69.13</td>
    <td class="tg-c3ow">67.90</td>
    <td class="tg-c3ow">73.67</td>
    <td class="tg-c3ow">76.25</td>
    <td class="tg-c3ow">77.10</td>
  </tr>
  <tr>
    <td class="tg-fymr" colspan="11">Distillation-based</td>
  </tr>
  <tr>
    <td class="tg-0pky">L2 </td>
    <td class="tg-c3ow">73.32</td>
    <td class="tg-c3ow">76.07</td>
    <td class="tg-c3ow">77.03</td>
    <td class="tg-c3ow">76.66</td>
    <td class="tg-c3ow">77.51</td>
    <td class="tg-c3ow">77.30</td>
    <td class="tg-c3ow">78.79</td>
    <td class="tg-c3ow">78.95</td>
    <td class="tg-c3ow">78.97</td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">79.00</span></td>
  </tr>
  <tr>
    <td class="tg-0pky">Making </td>
    <td class="tg-c3ow">70.76</td>
    <td class="tg-c3ow">74.42</td>
    <td class="tg-c3ow">76.39</td>
    <td class="tg-c3ow">75.34</td>
    <td class="tg-c3ow">74.74</td>
    <td class="tg-c3ow">76.92</td>
    <td class="tg-c3ow">76.91</td>
    <td class="tg-c3ow">78.67</td>
    <td class="tg-c3ow">78.07</td>
    <td class="tg-c3ow">79.06</td>
  </tr>
  <tr>
    <td class="tg-0pky">SKD</td>
    <td class="tg-c3ow">68.83</td>
    <td class="tg-c3ow">72.02</td>
    <td class="tg-c3ow">73.05</td>
    <td class="tg-c3ow">72.66</td>
    <td class="tg-c3ow">73.59</td>
    <td class="tg-c3ow">75.06</td>
    <td class="tg-c3ow">74.58</td>
    <td class="tg-c3ow">77.62</td>
    <td class="tg-c3ow">78.05</td>
    <td class="tg-c3ow">77.44</td>
  </tr>
  <tr>
    <td class="tg-0pky">CKD</td>
    <td class="tg-c3ow">76.19</td>
    <td class="tg-c3ow">76.59</td>
    <td class="tg-c3ow">77.48</td>
    <td class="tg-c3ow">77.14</td>
    <td class="tg-c3ow">77.90</td>
    <td class="tg-c3ow">76.97</td>
    <td class="tg-c3ow">77.92</td>
    <td class="tg-c3ow">78.29</td>
    <td class="tg-c3ow">78.54</td>
    <td class="tg-c3ow">78.34</td>
  </tr>
  <tr>
    <td class="tg-fymr" colspan="11">Our propose method</td>
  </tr>
  <tr>
    <td class="tg-0pky"><span style="font-style:normal">ConGen</span></td>
    <td class="tg-7btt">76.85</td>
    <td class="tg-7btt">78.09</td>
    <td class="tg-7btt">78.54</td>
    <td class="tg-7btt">78.22</td>
    <td class="tg-7btt">79.10</td>
    <td class="tg-7btt">78.91</td>
    <td class="tg-7btt">79.68</td>
    <td class="tg-7btt">79.73</td>
    <td class="tg-7btt">80.06</td>
    <td class="tg-7btt">79.78</td>
  </tr>
</tbody>
</table>

## Full results
<table class="tg">
<thead>
  <tr>
    <th class="tg-7btt">Models</th>
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

# We have Thai sentence embedding models from ConGen!!
- *Unsupervised learning*: [ConGen-simcse-model-roberta-base-thai](https://huggingface.co/kornwtp/ConGen-simcse-model-roberta-base-thai). Teacher model: [simcse-model-roberta-base-thai](https://huggingface.co/mrp/simcse-model-roberta-base-thai). Student model: [WangchanBERTa](https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased)
- *Weakly supervised learning*: [ConGen-paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/kornwtp/ConGen-paraphrase-multilingual-mpnet-base-v2/tree/main). Teacher model: [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2). Student model: [WangchanBERTa](https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased)
- Training data: we do backtranslatation from TH-to-EN-to-TH from [scb_mt_enth_2020's model](https://huggingface.co/datasets/scb_mt_enth_2020). The translation dataset: [back translated machine translation of SCB](https://drive.google.com/file/d/1u7kCk9xpTfQkxpJ0zfILpo9SR5KNMfaj/view?usp=share_link)
- We evaluate on two task benchmark tasks, such as [Thai semantic textual similarity benchmark](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark#thai-semantic-textual-similarity-benchmark) and [Thai transfer benchmark](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark#thai-transfer-benchmark)

## Hyper-Parameters

<table>
    <thead>
        <tr>
            <th>Parameters</th>
            <th>Models</th>
            <th>Teacher Temp</th>
            <th>Student Temp</th>
            <th>Queue Size</th>
            <th>Learning Rate</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2><30M</td>
            <td><a href="https://huggingface.co/kornwtp/ConGen-WangchanBERT-Tiny">ConGen-WangchanBERT-Tiny</a></td>
            <td align="center"> 0.01</td>
            <td align="center">0.01</td>
            <td align="center">65536</td>
            <td align="center">3e-4</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/kornwtp/ConGen-WangchanBERT-Small">ConGen-WangchanBERT-Small</a></td>
            <td align="center">0.05</td>
            <td align="center">0.09</td>
            <td align="center">65536</td>
            <td align="center">5e-4</td>
        </tr>
        <tr>
            <td rowspan=2>>100M</td>
            <td><a href="https://huggingface.co/kornwtp/ConGen-simcse-model-roberta-base-thai">ConGen-simcse-model-roberta-base-thai</a></td>
            <td align="center">0.05</td>
            <td align="center">0.03</td>
            <td align="center">65536</td>
            <td align="center">3e-4</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/kornwtp/ConGen-paraphrase-multilingual-mpnet-base-v2">ConGen-paraphrase-multilingual-mpnet-base-v2</a></td>
            <td align="center">0.05</td>
            <td align="center">0.05</td>
            <td align="center">262144</td>
            <td align="center">1e-4</td>
        </tr>
    </tbody>
</table>


## Thai semantic textual similarity benchmark

<table>
    <thead>
        <tr>
            <th>Parameters</th>
            <th>Models</th>
            <th>Spearman's Correlation (*100)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2><30M</td>
            <td><a href="https://huggingface.co/kornwtp/ConGen-WangchanBERT-Tiny">ConGen-WangchanBERT-Tiny</a></td>
            <td align="center">66.43</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/kornwtp/ConGen-WangchanBERT-Small">ConGen-WangchanBERT-Small</a></td>
            <td align="center">70.65</td>
        </tr>
        <tr>
            <td rowspan=2>>100M</td>
            <td><a href="https://huggingface.co/kornwtp/ConGen-simcse-model-roberta-base-thai">ConGen-simcse-model-roberta-base-thai</a></td>
            <td align="center">66.21</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/kornwtp/ConGen-paraphrase-multilingual-mpnet-base-v2">ConGen-paraphrase-multilingual-mpnet-base-v2</a></td>
            <td align="center">76.56</td>
        </tr>
    </tbody>
</table>

## Thai transfer benchmark

### Wisesight

<table>
    <thead>
        <tr>
            <th>Parameters</th>
            <th>Models</th>
            <th>Acc (*100)</th>
            <th>F1 (*100, weighted)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2><30M</td>
            <td><a href="https://huggingface.co/kornwtp/ConGen-WangchanBERT-Tiny">ConGen-WangchanBERT-Tiny</a></td>
            <td align="center">61.55</td>
            <td align="center">62.19</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/kornwtp/ConGen-WangchanBERT-Small">ConGen-WangchanBERT-Small</a></td>
            <td align="center">64.77</td>
            <td align="center">65.30</td>
        </tr>
        <tr>
            <td rowspan=2>>100M</td>
            <td><a href="https://huggingface.co/kornwtp/ConGen-simcse-model-roberta-base-thai">ConGen-simcse-model-roberta-base-thai</a></td>
            <td align="center">65.07</td>
            <td align="center">65.28</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/kornwtp/ConGen-paraphrase-multilingual-mpnet-base-v2">ConGen-paraphrase-multilingual-mpnet-base-v2</a></td>
            <td align="center">67.84</td>
            <td align="center">68.31</td>
        </tr>
    </tbody>
</table>

### Wongnai

<table>
    <thead>
        <tr>
            <th>Parameters</th>
            <th>Models</th>
            <th>Acc (*100)</th>
            <th>F1 (*100, weighted)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2><30M</td>
            <td><a href="https://huggingface.co/kornwtp/ConGen-WangchanBERT-Tiny">ConGen-WangchanBERT-Tiny</a></td>
            <td align="center">42.67</td>
            <td align="center">44.78</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/kornwtp/ConGen-WangchanBERT-Small">ConGen-WangchanBERT-Small</a></td>
            <td align="center">43.38</td>
            <td align="center">45.99</td>
        </tr>
        <tr>
            <td rowspan=2>>100M</td>
            <td><a href="https://huggingface.co/kornwtp/ConGen-simcse-model-roberta-base-thai">ConGen-simcse-model-roberta-base-thai</a></td>
            <td align="center">41.32</td>
            <td align="center">41.57</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/kornwtp/ConGen-paraphrase-multilingual-mpnet-base-v2">ConGen-paraphrase-multilingual-mpnet-base-v2</a></td>
            <td align="center">47.22</td>
            <td align="center">48.63</td>
        </tr>
    </tbody>
</table>

### Generated Review

<table>
    <thead>
        <tr>
            <th>Parameters</th>
            <th>Models</th>
            <th>Acc (*100)</th>
            <th>F1 (*100, weighted)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2><30M</td>
            <td><a href="https://huggingface.co/kornwtp/ConGen-WangchanBERT-Tiny">ConGen-WangchanBERT-Tiny</a></td>
            <td align="center">54.26</td>
            <td align="center">52.69</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/kornwtp/ConGen-WangchanBERT-Small">ConGen-WangchanBERT-Small</a></td>
            <td align="center">58.22</td>
            <td align="center">57.03</td>
        </tr>
        <tr>
            <td rowspan=2>>100M</td>
            <td><a href="https://huggingface.co/kornwtp/ConGen-simcse-model-roberta-base-thai">ConGen-simcse-model-roberta-base-thai</a></td>
            <td align="center">49.81</td>
            <td align="center">47.94</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/kornwtp/ConGen-paraphrase-multilingual-mpnet-base-v2">ConGen-paraphrase-multilingual-mpnet-base-v2</a></td>
            <td align="center">58.00</td>
            <td align="center">56.80</td>
        </tr>
    </tbody>
</table>
