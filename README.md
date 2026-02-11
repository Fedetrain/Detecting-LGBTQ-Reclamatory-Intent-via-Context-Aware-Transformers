# Detecting-LGBTQ-Reclamatory-Intent-via-Context-Aware-Transformers


# 🏳️‍🌈 GRUPPETTOZZO @ MultiPRIDE 2026
## Detecting LGBTQ+ Reclamatory Intent via Context-Aware Transformers

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
![Task](https://img.shields.io/badge/Task-EVALITA%202026-red)

> **Official Repository for the "GRUPPETTOZZO" system at EVALITA 2026 MultiPRIDE Challenge.**
> *Authors: Federico Traina, Alessandro Santoro, Gabriele Greco.*
> *University of Palermo, Department of Engineering.*

### 📄 [Read the Full Report (PDF)](./EVALITA_2026_gruppettozzo.pdf)

---

## 📌 Overview

Language within the LGBTQ+ context is characterized by complex phenomena such as **reclamation**, whereby terms originally intended as slurs are repurposed by community members for positive identity-affirming purposes.

This project implements a **Context-Aware Transformer Architecture** to address the **MultiPRIDE 2026** challenge. The goal is to distinguish between the denigratory usage of slurs (Hate Speech) and their reclaimed usage, operating in a multilingual setting (🇮🇹 Italian, 🇬🇧 English, 🇪🇸 Spanish).

We tackle two specific tasks:
* **Task A (Text-Only):** Classification based solely on the tweet text.
* **Task B (Context-Aware):** Classification enriched by the **User Biography**, which provides crucial pragmatic signals about the author's identity.

---

## 🧠 Methodology & Architecture

Our system leverages fine-tuned Encoder-only Language Models (LMs) enhanced by a custom neural head and advanced data augmentation strategies.

### 1. Data Augmentation (Back-Translation)
To address class imbalance and data scarcity, we implemented a Back-Translation pipeline (Source $\rightarrow$ Pivot $\rightarrow$ Source) using **MarianMT**.
* **Pivot Languages:** English (for IT/ES) and Spanish (for EN).
* **Strategies:**
    * *Full Augmentation:* Doubles the dataset size to increase linguistic variability.
    * *Balanced Augmentation:* Augments only the minority class (positive/reclaimed) to fix imbalance.

### 2. Context-Aware Architecture
The model processes input as `[CLS] TWEET [SEP] CONTEXT: BIO [SEP]`. We employ a **Hybrid Pooling Strategy** to maximize feature extraction:

```mermaid
graph TD
    subgraph Input Processing
    T[Tweet Text] --> Clean[Demojization & Cleaning]
    B[User Bio] --> Clean
    Clean --> Tok[Tokenizer]
    end

    subgraph Transformer Backbone
    Tok --> LM[Pre-trained LM<br/>(UmBERTo / BERT / XLM-R)]
    LM --> Emb[Last Hidden States]
    end

    subgraph "Custom Hybrid Pooling"
    Emb --> P1[CLS Token Embedding]
    Emb --> P2[Mean Pooling]
    Emb --> P3[Max Pooling]
    P1 --> Concat[Concatenation]
    P2 --> Concat
    P3 --> Concat
    end

    subgraph "Classification Head"
    Concat --> D1[Dropout]
    D1 --> MLP[Dense Layer + ReLU]
    MLP --> D2[Dropout]
    D2 --> Out[Final Prediction (Logits)]
    end
    
    style Backbone fill:#e1f5fe,stroke:#01579b
    style "Custom Hybrid Pooling" fill:#e8f5e9,stroke:#2e7d32
    style "Classification Head" fill:#fff3e0,stroke:#ef6c00
