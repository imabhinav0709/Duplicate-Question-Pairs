# 📖 Methodology: Quora Duplicate Question Pairs Detection

## Table of Contents

- [1. Problem Definition](#1-problem-definition)
- [2. Dataset](#2-dataset)
- [3. Text Preprocessing Pipeline](#3-text-preprocessing-pipeline)
- [4. Feature Engineering](#4-feature-engineering)
- [5. Model Architecture](#5-model-architecture)
- [6. Inference Pipeline](#6-inference-pipeline)
- [7. Post-Processing & Entity Guard](#7-post-processing--entity-guard)

---

## 1. Problem Definition

Given a pair of questions `(Q1, Q2)`, the goal is to predict a binary label:

- **1 (Duplicate):** Both questions have the same intent and would receive the same answer.
- **0 (Not Duplicate):** The questions ask different things.

This is framed as a **supervised binary classification** problem.

---

## 2. Dataset

The model was trained on the [Quora Question Pairs dataset](https://www.kaggle.com/c/quora-question-pairs), which contains:

- **~400,000** question pairs
- **~37%** positive (duplicate) class distribution
- Real-world Quora questions with natural language variations

---

## 3. Text Preprocessing Pipeline

Before feature extraction, both questions go through a robust preprocessing pipeline implemented in `helper.py → preprocess()`:

### Step-by-step:

```
Raw Question
    │
    ▼
1. Lowercase + Strip whitespace
    │
    ▼
2. Replace special characters (%, $, ₹, €, @) with word equivalents
    │
    ▼
3. Remove [math] tags
    │
    ▼
4. Normalize large numbers (e.g., 1,000,000 → 1m)
    │
    ▼
5. Expand contractions (e.g., "can't" → "can not", "I've" → "I have")
    │
    ▼
6. Remove HTML tags (via BeautifulSoup or regex fallback)
    │
    ▼
7. Remove punctuation (keep only word characters and spaces)
    │
    ▼
Cleaned Question
```

### Contraction Expansion

A comprehensive dictionary of **70+ English contractions** is used to expand shortened forms. This ensures consistency between questions like:

> "What's the best way?" → "what is the best way"

---

## 4. Feature Engineering

The model uses **222 features** per question pair, split into 5 categories:

### 4.1 Basic Features (7 features)

| # | Feature | Description |
|---|---|---|
| 1 | `q1_len` | Character length of Q1 |
| 2 | `q2_len` | Character length of Q2 |
| 3 | `q1_words` | Word count of Q1 |
| 4 | `q2_words` | Word count of Q2 |
| 5 | `common_words` | Number of shared words between Q1 and Q2 |
| 6 | `total_words` | Total unique words across Q1 and Q2 |
| 7 | `word_share_ratio` | `common_words / total_words` |

### 4.2 Token Features (8 features)

These features analyze the overlap of **content words** (non-stopwords), **stopwords**, and **all tokens**:

| # | Feature | Description |
|---|---|---|
| 1 | `cwc_min` | Common non-stopword count / min(Q1 non-stopwords, Q2 non-stopwords) |
| 2 | `cwc_max` | Common non-stopword count / max(Q1 non-stopwords, Q2 non-stopwords) |
| 3 | `csc_min` | Common stopword count / min(Q1 stopwords, Q2 stopwords) |
| 4 | `csc_max` | Common stopword count / max(Q1 stopwords, Q2 stopwords) |
| 5 | `ctc_min` | Common token count / min(Q1 tokens, Q2 tokens) |
| 6 | `ctc_max` | Common token count / max(Q1 tokens, Q2 tokens) |
| 7 | `last_word_eq` | 1 if last words match, 0 otherwise |
| 8 | `first_word_eq` | 1 if first words match, 0 otherwise |

> **Why first/last word?** Question words (What, How, Why) appear first, and the subject/object often appears last. Matching these is a strong signal.

### 4.3 Length-Based Features (3 features)

| # | Feature | Description |
|---|---|---|
| 1 | `abs_len_diff` | Absolute difference in word count |
| 2 | `avg_token_len` | Average word count across both questions |
| 3 | `lcs_ratio` | Longest Common Substring length / min(len(Q1), len(Q2)) |

> **Longest Common Substring (LCS):** Captures shared phrasing. Computed using the `distance` library (or `SequenceMatcher` fallback).

### 4.4 Fuzzy String Matching Features (4 features)

Using the **FuzzyWuzzy** library for edit-distance-based similarity:

| # | Feature | Description |
|---|---|---|
| 1 | `fuzz_ratio` | Simple character-level similarity (Levenshtein) |
| 2 | `fuzz_partial_ratio` | Best partial match (handles substring containment) |
| 3 | `token_sort_ratio` | Sort tokens alphabetically, then compare |
| 4 | `token_set_ratio` | Compare based on set intersection/difference of tokens |

> **Why multiple fuzzy scores?**
> - `fuzz_ratio` catches near-identical questions
> - `partial_ratio` catches when one question contains the other
> - `token_sort_ratio` handles word reordering ("best Python book" vs "Python book best")
> - `token_set_ratio` handles extra words ("What is Python?" vs "What exactly is Python programming?")

### 4.5 Bag-of-Words Features (200 features)

- A `CountVectorizer` (fitted during training) transforms each question into a sparse vector.
- Only the **top 100 features** (by vocabulary index) are used per question.
- Q1's 100 BoW features and Q2's 100 BoW features are concatenated → **200 BoW features**.

```
Q1 → CountVectorizer → [top 100 values] ─┐
                                          ├─→ [200 BoW features]
Q2 → CountVectorizer → [top 100 values] ─┘
```

---

## 5. Model Architecture

### Random Forest Classifier

- **Algorithm:** Random Forest (ensemble of decision trees)
- **Library:** scikit-learn
- **Input:** 222-dimensional feature vector
- **Output:** Binary prediction (0 = Not Duplicate, 1 = Duplicate) + probability score

### Why Random Forest?

| Advantage | Description |
|---|---|
| **Handles mixed features** | Works well with both continuous (fuzzy scores) and discrete (word counts) features |
| **Feature importance** | Built-in feature importance ranking for interpretability |
| **Robust to overfitting** | Ensemble averaging reduces variance |
| **No feature scaling needed** | Tree-based models don't require normalization |
| **Fast inference** | Predictions are fast, suitable for real-time web apps |

---

## 6. Inference Pipeline

```
User Input: (Q1, Q2)
        │
        ▼
┌─────────────────────┐
│   Text Preprocessing │  ← preprocess(q1), preprocess(q2)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Feature Extraction  │  ← 22 statistical features
│  (Basic + Token +    │
│   Length + Fuzzy)     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   BoW Vectorization  │  ← CountVectorizer → top 100 × 2 = 200 features
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Feature Concatenation│  ← [22 stats] + [100 bow_q1] + [100 bow_q2] = 222
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Random Forest       │  ← model.predict_proba()
│  Prediction          │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Post-Processing     │  ← Entity Guard + Threshold
└─────────┬───────────┘
          │
          ▼
   Final Prediction
   (Duplicate / Not Duplicate)
```

---

## 7. Post-Processing & Entity Guard

### Threshold-Based Classification

The model outputs a **probability score** (0.0 to 1.0). A configurable threshold (default: 0.45) determines the final prediction:

- `P(duplicate) ≥ threshold` → **Duplicate**
- `P(duplicate) < threshold` → **Not Duplicate**

### Entity & Keyword Conflict Guard

A secondary keyword analysis layer (`analyze_keywords()`) catches cases where the ML model may be fooled by **template similarity**:

**Example Problem:**
> Q1: "What is the capital of India?"
> Q2: "What is the capital of USA?"

These questions share identical structure but ask about **different entities**. The entity guard:

1. Extracts non-stopword keywords from both questions
2. Computes set intersection and differences
3. Flags a **subject conflict** when unique keywords outnumber common keywords
4. Overrides the model prediction to **Not Duplicate** when a conflict is detected

### Jaccard Similarity

Additionally, a **Jaccard similarity score** is computed on the keyword sets:

```
Jaccard = |Q1_keywords ∩ Q2_keywords| / |Q1_keywords ∪ Q2_keywords|
```

This provides an interpretable measure of keyword overlap shown in the UI.

---

## References

- [Quora Question Pairs - Kaggle Competition](https://www.kaggle.com/c/quora-question-pairs)
- [FuzzyWuzzy Documentation](https://github.com/seatgeek/fuzzywuzzy)
- [scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [NLTK Stopwords](https://www.nltk.org/search.html?q=stopwords)
