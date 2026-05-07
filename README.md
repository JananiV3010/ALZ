# Temporal Transformer Networks for Longitudinal Alzheimer's Progression Prediction

**Janani Vaiyapuriappan**  
Department of Biomedical Engineering, Johns Hopkins University  
*Course project — Deep Learning for Medical Imaging (EN.580.745)*  
*Supervised by Professor Ali Uneri*

---

> **Status:** Active development. Core pipeline (data loading, preprocessing, spatiotemporal transformer architecture, training loop) complete on OASIS-2. OASIS-3 and ADNI data acquisition currently underway. Foundation model baseline (BioViL / MedCLIP) planned as comparison.

---

## Overview

Alzheimer's disease (AD) is characterized by gradual cognitive decline, yet most deep learning approaches treat diagnosis as a static classification problem from a single scan. This project addresses that gap by framing AD prediction as a **longitudinal sequence modeling problem** — predicting conversion from Mild Cognitive Impairment (MCI) to AD using a subject's full history of structural MRI scans over time.

The clinical motivation is straightforward: early warning of conversion, before symptoms become pronounced, is where intervention has the most impact. A model that reads the trajectory — not just the current state — is more aligned with how clinicians actually think about disease progression.

---

## Approach

### Primary Model — Spatiotemporal Transformer

A temporal transformer that ingests a **sequence of T1-weighted MRI volumes** per subject across multiple timepoints and predicts AD conversion within a defined follow-up window.

The architecture has two components:

**Spatial Encoder (3D CNN)**
- Four convolutional blocks with BatchNorm and ReLU
- AdaptiveAvgPool3d to compress each 64×64×64 volume to a 256-dim feature vector
- Independently encodes each timepoint

**Temporal Transformer**
- Learnable positional encoding across visit sequence (up to 5 timepoints)
- 2-layer TransformerEncoder with 4 attention heads
- Classification from the first token (analogous to CLS token)
- BCEWithLogitsLoss with class-weighted positive weighting to handle converter/non-converter imbalance

MRI volumes are downsampled to **64×64×64** using `scipy.ndimage.zoom` to stay within Colab Pro memory constraints, with normalization to [0, 1].

### Secondary Task — CDR Worsening Prediction

In addition to subject-level conversion prediction, the model is also trained on a **transition-level task**: given all scans up to visit *i*, predict whether CDR score worsens at visit *i+1*. This framing produces more training samples and captures within-subject progression dynamics more granularly.

Subject-level train/val splits are used for both tasks to **prevent data leakage** across visits.

### Planned Baseline — Foundation Model

A pretrained medical vision foundation model (BioViL or MedCLIP) fine-tuned minimally on the longitudinal sequences, to provide a performance baseline and contextualize the temporal transformer results. Per supervisor recommendation, tuning will be kept minimal.

---

## Dataset

### Currently in use — OASIS-2

The **Open Access Series of Imaging Studies (OASIS-2)** longitudinal dataset:
- 150 subjects, 373 MRI sessions
- T1-weighted structural MRI (NIfTI format)
- Clinical Dementia Rating (CDR) scores at each visit
- 2–5 sessions per subject over several years
- Labels: Nondemented, Converted, Demented

**Subject breakdown:**
- Subjects with ≥2 sessions: ~140
- Converters (MCI → AD): ~14
- Non-converters: ~126

Class imbalance is handled via `WeightedRandomSampler` and `pos_weight` in the loss function.

### Data acquisition underway — OASIS-3

**OASIS-3** (1,098 subjects, 2,000+ MRI sessions, amyloid and tau PET) — data use agreement approved, download in progress via NITRC-IR (`nitrc.org/ir`). OASIS-3 adds:
- ~7× more subjects than OASIS-2
- Amyloid PET (AV-45) and tau PET (AV-1451) alongside structural MRI
- Freesurfer-processed volumetric outputs (hippocampal volume, cortical thickness)
- Richer longitudinal depth per subject

Integration with OASIS-3 will allow multimodal modeling (MRI + PET) and more robust evaluation.

### Planned — ADNI

**Alzheimer's Disease Neuroimaging Initiative (ADNI)** — the field's gold-standard longitudinal dataset — is planned as a third data source for cross-dataset generalization testing. ADNI includes MRI, amyloid PET, CSF biomarkers, genetics (APOE), and neuropsychological assessments across thousands of subjects.

---

## Current Results

*Training on OASIS-2 in progress. Results will be updated here as experiments complete.*

Figures generated from current data:

| Figure | Description |
|---|---|
| `dataset_overview.png` | CDR score distribution, group breakdown, sessions-per-subject histogram |
| `cdr_trajectories.png` | Longitudinal CDR trajectories — converters (red) vs stable nondemented (blue) |
| `mri_comparison.png` | Axial MRI slices — converters (top row) vs stable subjects (bottom row) at baseline |
| `training_batch_visualization.png` | Sample training batch — subjects × timepoints grid showing model inputs |
| `training_curves.png` | Loss and accuracy curves — Task 1 *(pending full training run)* |
| `confusion_matrix.png` | Confusion matrix — Task 1 *(pending full training run)* |

---

## Repository Structure

```
alzheimer-oasis/
├── ALZ_OASIS2_clean.ipynb     # Main pipeline notebook (Colab)
└── README.md
```

Data lives on Google Drive (Colab-mounted) and is not committed to the repository. OASIS-2 raw NIfTI files are organized as:

```
oasis-2/
├── OAS2_RAW_PART1/
│   └── OAS2_XXXX_MRX/RAW/mpr-1.nifti.hdr
├── OAS2_RAW_PART2/
│   └── ...
└── oasis_longitudinal_demographics.xlsx
```

---

## Technical Stack

| Component | Tool |
|---|---|
| MRI loading | NiBabel |
| Preprocessing | SciPy (`ndimage.zoom`), NumPy |
| Model | PyTorch (3D CNN + TransformerEncoder) |
| Training | Adam optimizer, ReduceLROnPlateau, WeightedRandomSampler |
| Evaluation | scikit-learn (classification report, ROC-AUC, confusion matrix) |
| Visualization | Matplotlib, Seaborn |
| Compute | Google Colab Pro (A100 GPU) |

---

## Future Directions

- **OASIS-3 integration** — scale training to 1,000+ subjects; add amyloid PET as a second input modality
- **ADNI validation** — test generalization across datasets
- **Foundation model baseline** — BioViL or MedCLIP fine-tuned on longitudinal sequences for comparison
- **Attention map visualization** — extract spatial and temporal attention weights to interpret which brain regions and timepoints drive predictions
- **Higher resolution** — experiment with 128×128×128 input once compute constraints are addressed
- **Omics integration** — APOE genotype and other genetic markers as auxiliary inputs

---

## References

- LaMontagne PJ, et al. OASIS-3: Longitudinal Neuroimaging, Clinical, and Cognitive Dataset for Normal Aging and Alzheimer Disease. *medRxiv* (2019).
- Marcus DS, et al. Longitudinal MRI Studies of the Brain in Normal Aging and Alzheimer Disease. *Journal of Cognitive Neuroscience* (2010). [OASIS-2]
- Bannur S, et al. Learning to Exploit Temporal Structure for Biomedical Vision-Language Processing. *CVPR* (2023). [BioViL]
- Vaswani A, et al. Attention Is All You Need. *NeurIPS* (2017).

---

## Acknowledgements

Project developed as part of EN.580.745 — Deep Learning for Medical Imaging at Johns Hopkins University. Thanks to Professor Ali Uneri for guidance on model design and dataset strategy.
