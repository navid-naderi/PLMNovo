# PLMNovo

Implementation Code of [PLMNovo: Protein Language Model–Aligned Spectra Embeddings for *De Novo* Peptide Sequencing](https://www.biorxiv.org/content/10.1101/2025.10.01.679857v1), a novel constrained learning method to incorporate pre-trained protein language models (PLMs) into deep learning-based *de novo* peptide sequencing (Preprint Coming Soon).

![Overview of PLMNovo.](https://github.com/navid-naderi/PLMNovo/blob/main/assets/fig_overview_plmnovo.png)

## Abstract
We consider the problem of *de novo* peptide sequencing in tandem mass spectrometry, where the goal is to predict the underlying peptide sequence given a spectrum's fragment peaks and precursor information. We present PLMNovo, a constrained learning framework that leverages pre-trained protein language models (PLMs) to guide the training process. In particular, we cast peptide-spectrum matching as a constrained optimization problem that enforces alignment between spectrum and peptide embeddings produced by a spectrum encoder and a PLM, respectively. We use a Lagrangian primal-dual algorithm to train the spectrum encoder and the peptide decoder by solving the proposed constrained learning problem, while optionally fine-tuning the pre-trained PLM. Through numerical experiments on established benchmarks, we demonstrate that PLMNovo outperforms several state-of-the-art deep learning-based *de novo* sequencing algorithms.

## Setting Up the Environment
Create a new virtual Conda environment, called `plmnovo`, with the required libraries using the following commands:
```
conda create -n plmnovo python=3.10
conda activate plmnovo
pip install -r requirements.txt
```

## Running PLMNovo

### Toy Dataset
You can train PLMNovo on a small train/test dataset (provided under `data/sample`) by running the following command:
```
python plmnovo.py train data/sample/train.mgf --validation_peak_path data/sample/test.mgf
```
Due to the small size of the dataset (1000 training samples and 128 test samples), modify the parameter `val_check_interval` in `config.yaml` as well as `log_every_n_steps` in `denovo/model_runner.py` to ensure proper validation and logging.

### Downloading the MSKB Dataset
The `download_mskb_data.sh` script can be used to download the MSKB datasets using for training and validating PLMNovo. The data will be downloded into `data/mgf_data/` by running
```
bash download_mskb_data.sh
```

### Training on the MSKB Dataset
Once the dataset is downloaded, run the following command to train PLMNovo:
```
python plmnovo.py train data/mgf_data/mskb_final/mskb_final.train.mgf --validation_peak_path data/mgf_data/mskb_final/mskb_final.test.mgf
```
Runs will be logged into [Weights and Biases](https://wandb.ai/site), and the best models (with the lowest validation loss) will be saved under `checkpoints`. You can modify the PLMNovo hyperparameters in `config.yaml`.

## Evaluating the Trained Models
The model checkpoints saved under `checkpoints` can be used for evaluation. As an example, you can evaluate a model `checkpoints/RUN_NAME/best_model.ckpt` on the multi-enzyme test set by running
```
python plmnovo.py evaluate data/mgf_data/multi_enzyme_simple/multi-enzyme-simple.test.mgf --model checkpoints/RUN_NAME/best_model.ckpt
```

## Alternative PLMs and Pooling Strategies
The current implementation of PLMNovo support two PLMs (ESM-2 8M and ESM-2 650M) and one pooling mechanism (average pooling). Alternative PLMs can be also be used with PLMNovo by modifying `denovo/model.py`, especially following similar steps as in lines 218-229. Also, alternative pooling mechanisms can be added to PLMNovo by updating the `pooling` method in `denovo/model.py`.


## Acknowledgments
This repository is built upon the [Casanovo GitHub repository](https://github.com/Noble-Lab/casanovo).


## Citation

If you use PLMNovo, please cite our preprint using the following BibTeX format:
```
@article{naderializadeh2025_plmnovo,
  title={Protein Language Model-Aligned Spectra Embeddings for De Novo Peptide Sequencing},
  author={NaderiAlizadeh, Navid and Dallago, Christian and Soderblom, Erik J and Soderling, Scott H},
  journal={bioRxiv},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```
