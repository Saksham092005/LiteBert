# LiteBert

## 📖 Project Overview

LiteBERT implements a simplified BERT-style transformer to tackle the MOSAIC-25 PS1 masked language modeling challenge: predicting masked tokens in sentences using contextual understanding. Our goal was to achieve high prediction accuracy on a held-out test set of sentences containing exactly one `<MASKED>` token each :contentReference[oaicite:0]{index=0}.

---

## 🚀 Features

- **Custom Tokenization & Vocabulary**  
  - Lowercasing, punctuation-aware regex splitting  
  - Special tokens: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`  
- **Dynamic Masking Strategy**  
  - 15% mask probability per token  
  - 80% → `[MASK]`, 10% → random token, 10% unchanged  
- **Scaled-Down Transformer Encoder**  
  - 6 layers, 8 attention heads  
  - Embedding dim: 512; FFN dim: 1024; dropout: 0.1  
- **Training & Inference Pipeline**  
  - PyTorch implementation with custom `Dataset` classes  
  - Training with Adam optimizer, cross-entropy loss  
  - Inference script to generate submission CSV  

---
