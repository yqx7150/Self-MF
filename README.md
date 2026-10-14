<div align="center">

<h1>Self-MF</h1>
<h3>Self-supervised Multi-contrast MRI Reconstruction<br>via Mask-guided Frequency-domain Fusion</h3>
<p><strong>Zhibo Chen</strong> · Dong Wang · Yu Guan · Xiang Ji · Qiegen Liu</p>
<p><em>Medical Physics · 2026</em></p>

<p><a href="https://doi.org/10.1002/mp.70708"><strong>Read the paper ↗</strong></a> &nbsp; · &nbsp; <a href="https://doi.org/10.1002/mp.70708">DOI: 10.1002/mp.70708</a> &nbsp; · &nbsp; <a href="#implementation"><strong>PyTorch code ↓</strong></a></p>

<p><a href="#overview"><strong>Overview</strong></a> &nbsp; · &nbsp; <a href="#method"><strong>Method</strong></a> &nbsp; · &nbsp; <a href="#experimental-results"><strong>Results</strong></a> &nbsp; · &nbsp; <a href="#implementation"><strong>Implementation</strong></a> &nbsp; · &nbsp; <a href="#citation"><strong>Citation</strong></a></p>

</div>

---

## Overview

**Self-MF** is a self-supervised reconstruction framework for **accelerated multi-contrast magnetic resonance imaging (MC-MRI)**. It exploits shared anatomical information across image contrasts while learning directly from acquired, under-sampled target k-space—without requiring fully sampled **target-contrast** training labels.

Rather than treating each contrast independently, Self-MF constructs complementary frequency-domain conditions from an auxiliary contrast and integrates them into a **score-based generative reconstruction** process. Target-specific information is maintained by self-supervision in the measurement domain and by data consistency during reconstruction.

<table>
<tr>
<td width="33%" valign="top">
<strong>01 · Self-supervised learning</strong><br><br>
Partition acquired target k-space into input and held-out measurement subsets to establish a target-specific learning signal.
</td>
<td width="33%" valign="top">
<strong>02 · Frequency-domain fusion</strong><br><br>
Use complementary frequency masks to extract structural guidance from an auxiliary contrast.
</td>
<td width="33%" valign="top">
<strong>03 · Generative reconstruction</strong><br><br>
Combine conditional score-based sampling with data consistency (DCL) and intensity calibration (ICL).
</td>
</tr>
</table>

## Method

### 1. Mask-guided frequency-domain fusion

A fully sampled auxiliary contrast is transformed into complementary, normalized frequency-domain components. Together with the acquired target k-space measurements, these components form the conditional representation used by the score network. The target contrast remains the reconstruction objective; the auxiliary contrast provides anatomical guidance.

**[View Figure 1 — Frequency-domain fusion (PDF)](figures/1.pdf)**

### 2. Contrast-aware self-supervised learning

The acquired target sampling set is randomly partitioned into disjoint input and supervision subsets. The first subset supplies target measurements for conditioning, while the second is held out to define the self-supervised learning objective. The article evaluates several split ratios and reports a favorable input fraction of **0.4**.

**[View Figure 2 — Self-supervised training framework (PDF)](figures/2.pdf)**

### 3. Conditional diffusion-based reconstruction

Self-MF uses a conditional score model and an iterative predictor–corrector sampler to recover missing target information. A **data consistency layer (DCL)** preserves the acquired target k-space information, while an **intensity calibration layer (ICL)** mitigates cross-contrast intensity inconsistencies.

**[View Figure 3 — Reconstruction pipeline (PDF)](figures/3.pdf)**

## Experimental Results

The published study evaluates Self-MF on **SIAT**, **IXI**, and **fastMRI brain** data under different undersampling patterns and acceleration factors. Selected results from the paper are shown below.

| Dataset | Sampling | Target contrast | PSNR ↑ (dB) | SSIM ↑ |
| :--- | :--- | :---: | ---: | ---: |
| SIAT | 2D random, R = 4 | T1WI | **35.16** | **0.8870** |
| SIAT | 2D random, R = 4 | T2WI | **33.46** | **0.9054** |
| SIAT | 2D random, R = 4 | PDWI | **35.31** | **0.9187** |
| fastMRI brain | 2D random, R = 4 | T2WI | **41.15** | **0.9637** |
| fastMRI brain | 2D random, R = 8 | T2WI | **35.02** | **0.8814** |

### Main reconstruction comparison

The primary qualitative comparison illustrates **SIAT multi-contrast reconstruction** at **R = 4** using 2D random undersampling. It presents T1WI, T2WI, and PDWI reconstructions alongside representative baselines, with enlarged details and error maps.

<div align="center">
<p><a href="figures/4.pdf"><strong>VIEW FIGURE 4 — MAIN RECONSTRUCTION COMPARISON (PDF) ↗</strong></a></p>
<p><sub><strong>Figure 4.</strong> Main visual comparison on SIAT, 2D random undersampling, R = 4. PDF format.</sub></p>
</div>

## Implementation

The repository contains the score-model components, self-supervised training routines, conditional reconstruction code, MRI operators, and experiment configurations.

```text
Self-MF/
├── README.md
├── selfmf/
│   ├── main.py                 Training entry point
│   ├── reconstruct.py          Conditional reconstruction
│   ├── sampling_conditional.py
│   ├── datasets_wd.py          SIAT data adapter
│   ├── datasets_wd_IXI.py      IXI data adapter
│   ├── models/                 Score-network modules
│   ├── configs/                Experiment configurations
│   └── op/                     CUDA operators
├── figures/
│   ├── 1.pdf                   Frequency-domain fusion
│   ├── 2.pdf                   Self-supervised learning
│   ├── 3.pdf                   Reconstruction framework
│   └── 4.pdf                   Main visual comparison
├── tests/
├── data/                       Data format guidance
├── requirements.txt
├── CITATION.cff
└── THIRD_PARTY_NOTICES.md
```

### Environment and data

The implementation is based on **PyTorch**, with scientific Python dependencies and CUDA operators. See [`requirements.txt`](requirements.txt) for the dependency list and [`data/README.md`](data/README.md) for data-format expectations.

The training and evaluation interfaces require locally prepared multi-contrast MRI data, sampling masks, and experiment-specific configuration. The **SIAT clinical dataset and pretrained checkpoints are not included**; public IXI and fastMRI data should be obtained under their respective usage terms.

### Training

After preparing the required data and runtime environment, the training entry point is:

```bash
cd selfmf
python main.py --config=configs/ve/SIAT_kdata_ncsnpp.py --workdir=exp --mode=train
```

### Reconstruction

The conditional reconstruction driver uses locally specified input data and trained weights. See the data-format guidance before running:

```bash
cd selfmf
python reconstruct.py
```

A lightweight acquired-measurement consistency test is also included:

```bash
python -m unittest discover -s tests -v
```

## Citation

If this work is helpful to your research, please cite the published article:

```bibtex
@article{chen2026selfmf,
  title   = {Self-supervised multi-contrast MRI reconstruction via mask-guided frequency-domain fusion},
  author  = {Chen, Zhibo and Wang, Dong and Guan, Yu and Ji, Xiang and Liu, Qiegen},
  journal = {Medical Physics},
  year    = {2026},
  doi     = {10.1002/mp.70708}
}
```

## Acknowledgements

The score-based generative modeling components build on [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch). Third-party components retain their original license terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the accompanying license texts.
