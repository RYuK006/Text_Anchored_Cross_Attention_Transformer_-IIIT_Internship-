# Towards Holistic Emotion Understanding: A Text-Anchored Cross-Attention Transformer for Multimodal Sentiment Analysis

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Benchmark-CMU--MOSEI-brightgreen.svg)](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/Aaron006/Text-Anchored-Transformer-Data)
[![Research](https://img.shields.io/badge/IIIT--Kottayam-iINT--2026-orange.svg)](https://www.iiitkottayam.ac.in/)
[![Author](https://img.shields.io/badge/Author-Aaron%20Alex%20Mathew-purple.svg)](https://www.linkedin.com/in/aaron-alex-mathew)

Official research repository for **"Towards Holistic Emotion Understanding: A Text-Anchored Cross-Attention Transformer for Multimodal Sentiment Analysis"**, developed during the **iINT-2026 Research Internship Programme** at the **Indian Institute of Information Technology (IIIT) Kottayam**, supervised by **Dr. Anu Maria Sebastian**.

---

## 📌 Abstract

Determining sentiment from conversational video is intrinsically challenging. Spoken text alone often conceals the true communicative intent—the exact same utterance can be sarcastic, empathetic, or hostile depending on vocal tone and facial expression. However, standard multimodal fusion paradigms suffer from notable limitations: early fusion concatenates heterogeneous feature spaces and propagates acoustic/visual noise, while late fusion processes streams independently, missing crucial cross-modal interactions.

This project investigates whether a **Cross-Attention Transformer anchored specifically to the Text modality** can better fuse acoustic and visual information. Across 11 systematic experiments on the **CMU-MOSEI** benchmark, our proposed architecture (with 16 attention heads and a 128-dimensional embedding space) achieves **80.03% binary accuracy** and a **weighted F1 score of 0.8687**, outperforming both Early Fusion (77.41% Acc) and Late Fusion (78.91% Acc) baselines.

---

## 🏗️ Architectural Overview

```
                      ┌──────────────────────────────────────┐
                      │    Text Modality (GloVe d=300)       │
                      └──────────────────┬───────────────────┘
                                         │  (Query: Q)
                                         ▼
                      ┌──────────────────────────────────────┐
                      │  Text-Anchored Cross-Attention Block │
                      │                                      │
                      │  Audio (K, V) ──► Cross-Attn (T->A)  │
                      │  Visual (K, V) ──► Cross-Attn (T->V) │
                      └──────────────────┬───────────────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │       Multi-Modal Gated Fusion       │
                      └──────────────────┬───────────────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │     Sentiment Prediction Head        │
                      └──────────────────────────────────────┘
```

### Key Technical Insights:
1. **Text as Anchor**: In human affective communication, language provides the primary semantic frame, while pitch, pause durations, and facial muscle movements modulate polarity. Treating text as Query ($Q$) forces audio and visual streams to answer specific semantic queries rather than drowning out linguistic cues.
2. **Representational Capacity Requirement**: Cross-attention mechanisms only outperform simple baselines when provided with adequate attention heads (16 heads vs 8 or 4 heads) and appropriate regularization ($d=0.1$ dropout).
3. **Qualitative Alignment**: Attention heatmaps and t-SNE projections demonstrate that the network successfully learns to attend to emotionally salient keyframes rather than naive word-level temporal correspondence.

---

## 📊 Experimental Results (CMU-MOSEI Benchmark)

| Exp ID | Configuration | MAE ↓ | Binary Accuracy (%) ↑ | Weighted F1 ↑ | Pearson r ↑ |
|:------:|:-------------:|:-----:|:---------------------:|:-------------:|:-----------:|
| **1A** | Early Fusion Baseline (Concatenation) | 0.6267 | 77.41% | 0.8376 | 0.6474 |
| **1B** | Late Fusion Baseline (Decision Ensemble) | 0.6078 | 78.91% | 0.8468 | 0.6783 |
| **1C** | Standard Cross-Attention (Vanilla) | 0.6318 | 76.67% | 0.8314 | 0.6567 |
| **2A** | Modality Ablation: Text Only | 0.6350 | 69.63% | 0.7564 | 0.6458 |
| **2B** | Modality Ablation: Text + Audio | 0.6678 | 73.83% | 0.8071 | 0.6084 |
| **2C** | Modality Ablation: Text + Vision | 0.6321 | 74.91% | 0.8166 | 0.6257 |
| **2D** | Tri-modal: Text + Audio + Vision | 0.6275 | 75.30% | 0.8179 | 0.6450 |
| **3A** | Attention Heads $H=8$ | 0.6220 | 76.18% | 0.8239 | 0.6470 |
| **3A** | **Attention Heads $H=16$ (Optimal Proposed)** | **0.6172** | **80.03%** | **0.8687** | **0.6585** |
| **3B** | Regularization: Dropout = 0.1 | 0.6256 | 78.79% | 0.8532 | 0.6469 |
| **3B** | Regularization: Dropout = 0.5 | 0.7371 | 77.34% | 0.8556 | 0.5723 |

---

## 📁 Repository Structure

```
├── .graphify/                  # Codebase semantic knowledge graph & entity index
├── CLAUDE.md                   # Fast agent memory-layer instructions
├── Structure_Diagrams/         # System architecture slides and diagram assets
├── data/                       # Dataset loader scripts & trained model checkpoints (*.pt)
│   ├── best_model_final.pt     # Checkpoint of the optimal 16-head cross-attention model
│   ├── early_best.pt           # Checkpoint for Early Fusion baseline
│   ├── late_best.pt            # Checkpoint for Late Fusion baseline
│   └── tf_*.pt                 # Checkpoints for ablation and hyperparameter runs
├── paper/                      # IEEE Conference publication files & formatting pipeline
│   ├── main.tex                # Complete LaTeX paper source (IEEEtran conference template)
│   ├── references.bib          # Bibliography references
│   ├── Aaron_Alex_Mathew_Research_Paper_IEEE_TAC.pdf # Compiled IEEE paper PDF
│   └── Paper Template.docx     # Conference formatting template reference
├── src/                        # Core PyTorch source code
│   ├── data/
│   │   ├── dataloader.py       # PyTorch Dataset and DataLoader with batch padding
│   │   └── download_data.py    # Automated dataset fetching and verification routines
│   ├── models/
│   │   ├── baseline_models.py  # Unimodal, Early Fusion, and Late Fusion implementations
│   │   └── transformer_fusion.py # Text-Anchored Cross-Attention Transformer implementation
│   ├── training/
│   │   └── train.py            # Training routines, early stopping, and metric calculations
│   └── utils/
├── visualizations/             # High-resolution research visualization figures
│   ├── attention_heatmaps.png  # Multi-head attention heatmaps across modalities
│   ├── cross_modal_alignment.png # Cross-modal temporal alignment matrices
│   ├── embedding_space_tsne.png  # t-SNE latent representation clusters
│   └── latency_vs_accuracy.png   # Inference latency vs. accuracy trade-off curves
├── Aaron Alex Mathew Report.docx # Comprehensive IIIT Kottayam internship technical report
├── experiment_results.csv      # Logged metrics for all 11 experimental configurations
├── requirements.txt            # Python dependencies
├── run_experiments.py          # Master script executing all benchmark experiments
├── run_best.py                 # Evaluator / runner for the optimal checkpoint
├── run_ensemble.py             # Multi-model ensemble inference pipeline
├── run_pipeline.py             # End-to-end data-to-prediction pipeline
├── viz_attention_heatmaps.py   # Script to generate cross-modal attention maps
├── viz_alignment_maps.py       # Script to generate temporal alignment plots
├── viz_embedding_space.py      # Script to generate t-SNE latent space projections
└── viz_latency_accuracy.py     # Script to benchmark latency vs accuracy
```

---

## 🚀 Quick Start & Reproduction

Clone the repository and automatically download the dataset in one go:

```bash
git clone https://github.com/RYuK006/Text_Anchored_Cross_Attention_Transformer_-IIIT_Internship-.git
cd Text_Anchored_Cross_Attention_Transformer_-IIIT_Internship-
python src/data/download_data.py
```

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Access (Hugging Face Backup)

The aligned preprocessed datasets (`mosei_senti_data.pkl`, `mosei_senti_data_noalign.pkl`, `iemocap_data.pkl`) are hosted on Hugging Face:
🔗 **[Aaron006/Text-Anchored-Transformer-Data](https://huggingface.co/datasets/Aaron006/Text-Anchored-Transformer-Data)**

You can download individual files via Python:
```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="Aaron006/Text-Anchored-Transformer-Data",
    filename="mosei_senti_data.pkl",
    local_dir="data",
    repo_type="dataset"
)
```

*(Alternatively, `python src/data/download_data.py` pulls from Hugging Face automatically with fallback to the CMU Dropbox archive).*

### 3. Running Models & Visualizations

To evaluate the best trained 16-head model ($80.03\%$ Acc, $0.8687$ F1):
```bash
python run_best.py
```

To run the full 11-experiment ablation and parameter sweep:
```bash
python run_experiments.py
```

To generate publication figures:
```bash
python viz_attention_heatmaps.py
python viz_alignment_maps.py
python viz_embedding_space.py
python viz_latency_accuracy.py
```

---

## 👥 Authors & Acknowledgments

* **Aaron Alex Mathew** — Department of Computer Science & Engineering (Artificial Intelligence), Providence College of Engineering, Chengannur, Kerala, India (`aaron.prc24ca001@student.providence.edu.in`)
* **Dr. Anu Maria Sebastian** (Mentor) — Department of Computer Science & Engineering, Indian Institute of Information Technology (IIIT) Kottayam, Kerala, India (`anumaria@iiitkottayam.ac.in`)

Developed under the **iINT-2026 Internship Programme** at **IIIT Kottayam**.
