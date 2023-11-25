# ParsingDST: an framework of In-context Learning that parsing dialogue as an intermediatestate state for DST

This is the pytorch implementation of
**Semantic Parsing by Large Language Models for Intricate Updating Strategies of Zero-Shot Dialogue State Tracking**. 
[[PDF]](https://arxiv.org/abs/2310.10520), which is accepted as a **Findings paper of EMNLP 2023**.

If you find this repo helpful, please cite the following paper:
<pre>
@article{wu2023semantic,
  title={Semantic Parsing by Large Language Models for Intricate Updating Strategies of Zero-Shot Dialogue State Tracking},
  author={Wu, Yuxiang and Dong, Guanting and Xu, Weiran},
  journal={arXiv preprint arXiv:2310.10520},
  year={2023}
}
</pre>

## 🎥 Environment
Install PyTorch, Huggingface transformers, and openai.

(optional) Put your OpenAI API key in `config.py` to use ChatGPT and else models.

## 🍯 Data
We follow the pipeline of [MultiWoz 2.4 repo](https://github.com/smartyfh/MultiWOZ2.4/) for data preprocessing.
We modified a bit to unify the ontology between MultiWOZ 2.1 and 2.4
To download and create the dataset
```console
cd data
python create_data.py --main_dir mwz21 --mwz_ver 2.1 --target_path mwz2.1  # for MultiWOZ 2.1
python create_data.py --main_dir mwz24 --mwz_ver 2.4 --target_path mwz2.4  # for MultiWOZ 2.4
```

### 🎯 preprocess the dataset
Run the following script to sample and preprocess the few-shot and full-shot training sets, dev set and test set. 
For few-shot experiments, the retriever is trained on the selection pool. So we have save the selection pool for each of the experiment.
`data/sample.py` samples and processes the training sets.
All the processed data will be saved in the `data` folder.
```console
./preprocess.sh
```

## 🎯 In-Context Learning Experiments
### Zero-shot experiment
Run the zero-shot experiment on MultiWOZ 2.1 by
```console
python run_zeroshot_experiment.py --output_dir ./expts/zero-shot --mwz_ver 2.1
```

### Analyze using the running log
Notice that the only difference between MultiWOZ 2.1 and 2.4 are the labels of dev and test set. So, there is no need to run the same experiment again for 2.1 and 2.4. Instead, we can get the MultiWOZ 2.4 scores with the running log on MultiWOZ 2.1. 

Get the per-domain result on MultiWOZ 2.1 by
```console
python evaluate_run_log_by_domain.py --running_log expts/zero-shot/running_log.json --test_fn data/mw24_100p_test.json --mwz_ver 2.1
```

To get result on MultiWOZ 2.4, change to `--mwz_ver 2.4`.
