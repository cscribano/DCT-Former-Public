# DCT-Former: Efficient Self-Attention with Discrete Cosine Transform [PAPER](https://arxiv.org/pdf/2203.01178.pdf)

## Requirements
* Create a conda envrionment using the provided `envrionment.yml` in `data` as described [HERE](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
```
conda env create -f environment.yml
```
  
## Dataset

### Pretraining Dataset

The pre-processing stages are taken from [academic-budget-bert](https://github.com/IntelLabs/academic-budget-bert),
additional information is available in `data/README.md`

* Download wikipedia dump from https://dumps.wikimedia.org/ and pre-process it using [Wikiextractor.py](https://github.com/attardi/wikiextractor)
* ```python process_data.py -f enwiki-latest-pages-articles.xml -o <output_dir> --type wiki```
* Initial Sharding:
```bash
python shard_data.py \
    --dir <path_to_text_files> \
    -o <output_dir> \
    --num_train_shards 256 \
    --num_test_shards 128 \
    --frac_test 0.1
```
* Samples Generation:
```
python generate_samples.py \
    --dir <path_to_shards> \
    -o <output_path> \
    --dup_factor 10 \
    --seed 42 \
    --do_lower_case 1 \
    --masked_lm_prob 0.15 \ 
    --max_seq_length 128 \
    --model_name bert-base-uncased \
    --max_predictions_per_seq 20 \
    --n_processes 4
```

### Finetuning Dataset
For finetuining the "Large Movie Review" dataset is used, which is freely available [HERE](https://ai.stanford.edu/~amaas/data/sentiment/)

## Training

### Pretraining (English Wikipedia)
* Adjust the `.json` file in `experiments/paper_pretrain` according to the experiment you want to run.
* Change `data_root` to point to the output directory of `generate_samples.py`
* change `/data/logs` to the desired logging directory
* To train on <num_gpus> on the same machine: ```python -m torch.distributed.launch --nproc_per_node=<num_gpus> --master_addr="127.0.0.1" --master_port=1234 main.py --exp_name=paper_pretrain/<experiment_name> --seed=6969```

When the trining is complete run:
```
python -m torch.distributed.launch --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=1234 main.py --exp_name=<exp_log_path> --conf_file_path=<log_dir> --mode=test
```
To compute the pretraining metrics (Accuracy) on the validation set.

### Finetuning (ImDB)
* Adjust the `.json` file in `experiments/paper_finetune` according to the experiment you want to run.
* Change `data_root` to point to the output directory `aclImdb`
* Change `pretrain_ck` to point to the intended pretrain checkpoint
* change `/data/logs` to the desired logging directory
* To train on <num_gpus> on the same machine: ```python -m torch.distributed.launch --nproc_per_node=<num_gpus> --master_addr="127.0.0.1" --master_port=1234 main.py --exp_name=paper_finetune/<experiment_name> --seed=6969```

## Acknowledgments
* Training BERT with Compute/Time (Academic) Budget: https://github.com/IntelLabs/academic-budget-bert
* Nystromformer: https://github.com/mlpen/Nystromformer
* torch-dct: https://github.com/zh217/torch-dct
* Deep Speed examples: https://github.com/microsoft/DeepSpeedExamples

## Reference (Pre-print)
```
@article{scribano2022dct,
  title={DCT-Former: Efficient Self-Attention with Discrete Cosine Transform},
  author={Scribano, Carmelo and Franchini, Giorgia and Prato, Marco and Bertogna, Marko},
  journal={arXiv e-prints},
  pages={arXiv--2203},
  year={2022}
}
```

