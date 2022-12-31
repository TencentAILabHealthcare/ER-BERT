# ER-BERT
ER-BERT is a pre-trained model for the de novo generation of epitope-binding TCR, which is composed of two BERT modules: EpitopeBERT and ReceptorBERT.
## Installation
To install the required packages for running ER-BERT, please use the following command
```bash
pip install -r requirements.txt
```
## How to train and use ER-BERT
The training of ER-BERT for TCR generation is composed of two steps: MAA Task and Seq2Seq task. The details of each training are in the `Code/config` folder. For example, for the FMFM tokenizer, all the related config files are in the `Code/config/FMFM` folder. Here we use the FMFM tokenizer as an example to show how to use ER-BERT for TCR generation. Note that all the commands are run in the `Code` folder.
### 1. MAA Task
The MAA task is used for the self-training of EpitopeBERT and ReceptorBERT. The training command for EpitopeBERT is
```bash
python bert_pretrain_maa_main.py --config config/FMFM/bert_pretrain_maa_FMFM_epitope.json
```
The training command for ReceptorBERT is
```bash
python bert_pretrain_maa_main.py --config config/FMFM/bert_pretrain_maa_FMFM_beta.json
```
After the training, the pre-trained EpitopeBERT and ReceptorBERT will be saved in the `../Result/checkpoints/BERT-Epitope-Pretrain-FMFM-MAA/XXXX_XXXXXX` and `../Result/checkpoints/BERT-Beta-Pretrain-FMFM-MAA/XXXX_XXXXXX` folder, where `XXXX_XXXXXX` is the timestamp of the training.
### 2. Seq2Seq Task
The Seq2Seq task is used for the training of TCR generataion of ER-BERT (training of ERTransformer). Before running the Seq2Seq task, please copy the path of pre-trained EpitopeBERT (`../Result/checkpoints/BERT-Epitope-Pretrain-FMFM-MAA/XXXX_XXXXXX`) and ReceptorBERT (`../Result/checkpoints/BERT-Beta-Pretrain-FMFM-MAA/XXXX_XXXXXX`) to replace the corresponding file path in the config file `bert_finetuning_er_seq2seq_FMFM.json`. In detail: please replace the "epitope_tokenizer_dir" and "EpitopeBert_dir" using `../Result/checkpoints/BERT-Epitope-Pretrain-FMFM-MAA/XXXX_XXXXXX`; replace the "receptor_tokenizer_dir" and "ReceptorBert_dir" using `../Result/checkpoints/BERT-Beta-Pretrain-FMFM-MAA/XXXX_XXXXXX`. The training command for ER-BERT is
```bash
python bert_finetuning_seq2seq_main.py --config config/FMFM/bert_finetuning_er_seq2seq_FMFM.json
```
After the training, the trained ERTransformer will be saved in the `../Result/checkpoints/ERTransformer-Finetuning-Seq2seq-FMFM/XXXX_XXXXXX` folder.
### 3. BSP Task
The BSP task is used for the preparation of the external discriminator based on ER-BERT. Same as the training in the Seq2Seq task, before running the BSP task, please copy the path of pre-trained EpitopeBERT (`../Result/checkpoints/BERT-Epitope-Pretrain-FMFM-MAA/XXXX_XXXXXX`) and ReceptorBERT (`../Result/checkpoints/BERT-Beta-Pretrain-FMFM-MAA/XXXX_XXXXXX`) to replace the corresponding file path in the config file `bert_finetuning_er_seq2seq_FMFM.json`.
The training command for the BSP task is
```bash
python bert_finetuning_bsp_main.py --config config/FMFM/bert_finetuning_bsp_FMFM.json
```
After the training, the trained EpitopeBERT and ReceptorBERT will be saved in the `../Result/checkpoints/BERT-Finetuning-BSP-FMFM/XXXX_XXXXXX/EpitopeBERT` and `../Result/checkpoints/BERT-Finetuning-BSP-FMFM/XXXX_XXXXXX/ReceptorBERT` folder, resepectively. And the best model will be saved as `../Result/checkpoints/BERT-Finetuning-BSP-FMFM/XXXX_XXXXXX/model_best.pth`.
### 4. Utilization
The utilization of the trained ER-BERT on a new dataset (here we use MIRA as an example) is composed of three steps: 1) We first use 20% of the TCRs in MIRA for the fine-tuning of ERTransformer; 2) We need to train an external discriminator uilizing 80% of the TCRs based on the ER-BERT trained after the BSP task. 3) Then, we use the fine-tuned ERTransformer to generate TCRs for each epitope in MIRA and use the external discriminator to determine whether the generated TCRs could bind to the given epitope. The details are as follows:
#### 1). Fine-tune ERTransformer. 
Before running the fine-tuning, please replace some related paths in the `bert_finetuning_er_seq2seq_FMFM_MIRA.json` file. In detail, please replace the "epitope_tokenizer_dir" and "EpitopeBert_dir" using `../Result/checkpoints/BERT-Epitope-Pretrain-FMFM-MAA/XXXX_XXXXXX`; replace the "receptor_tokenizer_dir" and "ReceptorBert_dir" using `../Result/checkpoints/BERT-Beta-Pretrain-FMFM-MAA/XXXX_XXXXXX`; replace the "resume" using `../Result/checkpoints/ERTransformer-Finetuning-Seq2seq-FMFM/XXXX_XXXXXX`. The fine-tuning command is
```bash
python bert_finetuning_seq2seq_MIRA.py --config config/FMFM/bert_finetuning_er_seq2seq_FMFM_MIRA.json
```
After the fine-tuning, the fine-tuned ERTransformer on MIRA will be saved in the `../Result/checkpoints/MIRA-ERTransformer-Finetuning-Seq2seq-FMFM/XXXX_XXXXXX` folder.
#### 2). Train external discriminator. 
Before running the training of external discriminator, please replace some related paths in the `external_MIRA_binding_FMFM.json`. In detail, please replace the "epitope_tokenizer_dir" using `../Result/checkpoints/BERT-Epitope-Pretrain-FMFM-MAA/XXXX_XXXXXX`, and the "EpitopeBert_dir" using `../Result/checkpoints/BERT-Finetuning-BSP-FMFM/XXXX_XXXXXX/EpitopeBERT`; replace the "receptor_tokenizer_dir" using `../Result/checkpoints/BERT-Beta-Pretrain-FMFM-MAA/XXXX_XXXXXX`, and the "ReceptorBert_dir" using `../Result/checkpoints/BERT-Finetuning-BSP-FMFM/XXXX_XXXXXX/ReceptorBERT`. The training command is
```bash
python external_MIRA_binding.py --config config/FMFM/external_MIRA_binding_FMFM.json
```
After the training, the trained external discriminator will be saved as `../Result/checkpoints/External-MIRA-BSP-FMFM/XXXX_XXXXXX/model_best.pth`, with the EpitopeBERT saved in the `../Result/checkpoints/External-MIRA-BSP-FMFM/XXXX_XXXXXX/EpitopeBERT` folder and the ReceptorBERT saved in the `../Result/checkpoints/External-MIRA-BSP-FMFM/XXXX_XXXXXX/ReceptorBERT` folder.
#### 3). TCR generation and binding determination. 
Before running the TCR generation and binding determination, please replace some related paths in the `seq2seq_MIRA_FMFM.json`. In detail, for the configuration of ERTransformer, please replace the "epitope_tokenizer_dir" under "data_loader/args" using `../Result/checkpoints/BERT-Epitope-Pretrain-FMFM-MAA/XXXX_XXXXXX`, and the "receptor_tokenizer_dir" under "data_loader/args" using `../Result/checkpoints/BERT-Beta-Pretrain-FMFM-MAA/XXXX_XXXXXX`, and replace the "resume" using `../Result/checkpoints/MIRA-ERTransformer-Finetuning-Seq2seq-FMFM/XXXX_XXXXXX`. \
For the external discriminator, please use your desired one first (here use the ER-BERT with the FMFM tokenizer fine-tuned on MIRA), and then replace "epitope_tokenizer_dir" under "data_loader_discriminator/args" and "epitope_tokenizer_dir" under "discriminator/args" using `../Result/checkpoints/BERT-Epitope-Pretrain-FMFM-MAA/XXXX_XXXXXX`, and replace "receptor_tokenizer_dir" under "data_loader_discriminator/args" and "ReceptorBert_dir" under "discriminator/args" using `../Result/checkpoints/BERT-Beta-Pretrain-FMFM-MAA/XXXX_XXXXXX`; and replace "discriminator_resume" using `../Result/checkpoints/External-MIRA-BSP-FMFM/XXXX_XXXXXX/model_best.pth`. The command is
```bash
python evaluate_seq2seq_MIRA.py --config config/FMFM/seq2seq_MIRA_FMFM.json
```
## Model availability
ER-BERT with two tokenizers on all the three tasks (MAA, Seq2Seq, and BSP) on the comprehensive training dataset are available on Zenodo: https://doi.org/10.5281/zenodo.7494046. And you can fine-tuning it on your own dataset as we described in the above section (Section 4. Utilization).
## Data availability
Due to the space limitation, we present part of data used in this project in the folder `ProcessedData`. For the full data, please contact us.
## Contact
If you have any questions, please contact us via email: 
- [Jiannan Yang](mailto:jiannan.yang@my.cityu.edu.hk)
- [Bing He](mailto:hebinghb@gmail.com)
"# ER-BERT" 
