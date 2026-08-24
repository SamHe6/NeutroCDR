# Welcome to NeutroCDR: Pairing CDRs with Antigenic Mutants for Cross-attention Prediction of Antibody Neutralization

Recognition of neutralizing antibodies is critically important for antibody drug development. However, the
continuing emergence of antigen mutants/variants poses challenges to the evaluation of antibody neutralization. Antigenic
mutations may alter antibody–antigen recognition, causing the same antibody to exhibit different neutralizing behaviors
against different antigen mutants. Large-scale experimental evaluation of antibody–mutants combinations is costly, while
existing sequence-based methods rarely cross-model both antibody complementarity-determining regions (CDRs) and
variant-specific mutations. We present NeutroCDR, a CDR-focused and mutation-aware framework for neutralization classification and
log10(IC50) regression. A novel step of NeutroCDR is to pair CDR-focused antibody representations with mutation-
aware antigenic features which are then integrated by a biologically informed cross-attention in the prediction. On the
SARS-CoV-2 benchmark, NeutroCDR achieved an accuracy of 0.857 and an MCC of 0.695, with an MAE of 0.430 and
an RMSE of 0.761 for regression. On an independent CoV-AbDab test set, it achieved the highest AUPRC across the
full, seen-antibody, and unseen-antibody subsets. Further attribution analysis revealed H-CDR3 and mutation-associated
antigen sites as the key contributors to the performance. This work provides an interpretable computational approach for
large-scale neutralization prediction potentially useful for antibody therapeutic development.

![The workflow of this study](https://github.com/SamHe6/NeutroCDR/blob/main/workflow.png)

# Requirements<bar>
```
numpy>=1.24
pandas>=2.0
pyyaml>=6.0
scipy>=1.10
scikit-learn>=1.3
tqdm>=4.66

torch>=2.0
transformers>=4.40

abnumber==0.4.4
anarcii==2.0.6
```

# Datasets
We provided our dataset and you can find them [neutralization.zip](https://github.com/SamHe6/NeutroCDR/blob/main/neutralization.zip),[external](https://github.com/SamHe6/NeutroCDR/tree/main/external) and [subset](https://github.com/SamHe6/NeutroCDR/tree/main/subset).

# Code
We provide the source code and you can find them [neutrocdr](https://github.com/SamHe6/NeutroCDR/tree/main/neutrocdr)

# Training
Run five-fold cross-validation.

```bash
python neutrocdr/run_cross_validation.py \
  --config neutro_cdr_formal_esm_igbert.yaml \
  --out-dir results/neutro_cdr_5fold \
  --folds 5 \
  --epochs 100 \
  --patience 12 \
  --min-delta 0.001 \
  --device cuda
```
