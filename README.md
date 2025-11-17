# Question Answering System with BiDAF, BERT, and RAG

**Date:** April 23, 2025  
**Contributors:**  
- Amina Alisheva  
- Ariana Kenbayeva  
- Jafar Isbarov  

---

## Overview
This project implements a complete Question Answering (QA) system using both extractive and generative approaches. The work is divided into two main components:

1. **Reading Comprehension (Closed-Domain QA):**  
   Implementation of the BiDAF architecture enhanced with multilingual BERT embeddings to extract answer spans from a given context.

2. **Open-Domain QA with Retrieval-Augmented Generation (RAG):**  
   Construction of a retrieval system using TF-IDF and Sentence-BERT, followed by integration with extractive and generative QA models to answer questions across a larger corpus.

The project uses the **squad-azerbaijani-reindex-translation** dataset from Hugging Face and follows the requirements provided in the course instructions.

---

## Dataset
- **Dataset Name:** `hajili/squad-azerbaijani-reindex-translation`  
- **Purpose:** Extractive QA (context–question–answer triplets)  
- **Description:** Azerbaijani translation and re-indexing of SQuAD  
- **Use Case:** Evaluating closed-domain QA models in low-resource languages  

This dataset is suitable for span-based QA systems such as BiDAF and BERT-based architectures.

---

## Part 1: Reading Comprehension System

### BiDAF Architecture
The Bidirectional Attention Flow (BiDAF) architecture was implemented using PyTorch. The model computes:

- Context-to-query attention  
- Query-to-context attention  
- A combined similarity representation for answer span prediction  

To improve contextual understanding, the system uses BERT embeddings generated as follows:
[CLS] question [SEP] context [SEP]

using the `bert-base-multilingual-cased` tokenizer.

### Training Configuration
The following parameters were used in the training pipeline:  
`MODEL_NAME = "bert-base-multilingual-cased"`  
`DATASET_NAME = "hajili/squad-azerbaijani-reindex-translation"`  
`MAX_CONTEXT_LENGTH = 384`  
`MAX_QUESTION_LENGTH = 64`  
`BATCH_SIZE = 8`  
`LEARNING_RATE = 1e-4`  
`EPOCHS = 1`  
`DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`

These values reflect typical constraints for QA models trained on low-resource datasets.

### Results
The extractive QA component was evaluated using standard SQuAD metrics:

- **Exact Match (EM): 2.13%**  
- **F1 Score: 1.97%**

Although the training and validation losses decreased consistently, the final accuracy remained low. Typical issues included:

- Off-by-several-word span predictions  
- Difficulty handling long contexts (~300+ tokens)  
- Re-indexing inconsistencies within the dataset  
- Misalignment between tokenized and raw text answer boundaries  

These findings align with the observations presented in the related project materials and example outputs.

---

## Part 2: Open-Domain QA System (RAG)

### Retrieval Methods
Three retrieval approaches were implemented and evaluated:

1. Bag-of-Words (BoW)  
2. TF-IDF Sparse Retrieval  
3. Sentence-BERT Dense Retrieval  

Dense retrieval methods significantly outperformed sparse retrieval. Experimental results:

| Metric / Method | BoW | TF-IDF | Dense Retriever |
|-----------------|-----|--------|-----------------|
| P@5             | 0.29 | 0.36 | 0.45 |
| R@5             | 0.35 | 0.48 | 0.69 |

Sentence-BERT achieved the strongest results, retrieving more relevant passages and improving downstream QA performance.

### RAG Pipeline
The RAG pipeline consists of the following stages:

1. **Question Encoding** using Sentence-BERT  
2. **Document Retrieval** using TF-IDF or Sentence-BERT  
3. **Context Augmentation** by concatenating the top retrieved documents  
4. **Answer Generation or Extraction**:
   - Extractive: BiDAF applied to retrieved contexts  
   - Generative: mT5 or BART used to produce free-form answers  

### Observed Trade-offs
**Extractive model:**
- More precise on short, clean inputs  
- Highly sensitive to context quality and span alignment  
- Performs poorly with noisy or imperfect retrieval  

**Generative model:**
- More robust to partially relevant or noisy contexts  
- Produces fluent sentences  
- May generate confident but incorrect answers  

A small manual evaluation of 20 queries found:

- Generative RAG approach produced acceptable answers for **~65%** of cases  
- Extractive BiDAF achieved acceptable answers in **~30%** of cases  

---

## Discussion
This project highlights the strengths and limitations of different QA approaches, especially in the context of Azerbaijani text, which is a low-resource language.

Key observations include:

- BERT embeddings provide improved representation but do not fully resolve span alignment issues.  
- Dense retrieval plays a significant role in improving the accuracy of both extractive and generative systems.  
- Generative QA performs more robustly under noisy input, while extractive models require highly accurate retrieval.  
- Span prediction challenges become more severe with longer or more complex passages.  

Overall, retrieval quality strongly influences system performance, sometimes more than the answering model itself.

---

## Contributions
**Amina Alisheva:**  
Integrated BERT with the QA pipeline, developed preprocessing steps, designed the generative QA component.

**Ariana Kenbayeva:**  
Designed the TF-IDF and Sentence-BERT retrieval, implemented full Retrieval-Augmented Generation (RAG) pipeline.


**Jafar Isbarov:**  
Implemented BiDAF from scratch using PyTorch, trained models with GloVe and BERT embeddings, analyzed training results.

---

## Future Improvements
- Improve span decoding and handling of offset mismatches  
- Combine extractive and generative answers in a hybrid approach  
- Use transformers supporting long contexts (e.g., Longformer, LED)  
- Fine-tune multilingual T5 specifically for Azerbaijani QA  
- Experiment with additional retrieval techniques such as BM25 or ColBERT  
- Train models on larger Azerbaijani corpora  

---

## References
- Hajili, M. (2024). *squad-azerbaijani-reindex-translation*. Hugging Face.  
- Project Report for Course Project 5.  
- Course Instructions for QA Implementation.  
- Jurafsky, D., & Martin, J. (2025). *Speech and Language Processing (3rd ed.)*.  
