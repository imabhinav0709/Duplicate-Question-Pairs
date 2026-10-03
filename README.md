# ❓ Quora Duplicate Question Pairs Detector

## Live Demo
🚀 Deployed App: https://duplicate-question-pairs-fiit.onrender.com/

A machine learning-powered web application that predicts whether two Quora questions are duplicates — i.e., whether they ask the same thing or share the same intent.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.54-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Problem Statement

Quora receives millions of questions daily, many of which are semantically identical despite being worded differently. Identifying duplicate questions helps:

- **Reduce redundancy** — consolidate answers in one place
- **Improve search** — users find answers faster
- **Enhance UX** — avoid answering the same question twice

This project builds a binary classifier that takes two questions as input and predicts whether they are duplicates.

---

## 🏗️ Project Structure

```
Duplicate-Question-Pairs/
├── app.py                # Streamlit web application
├── helper.py             # Feature engineering & preprocessing
├── model.pkl             # Trained Random Forest classifier
├── cv.pkl                # Fitted CountVectorizer (BoW)
├── stopwords.pkl         # NLTK English stopwords (pickled)
├── requirements.txt      # Python dependencies
├── render.yaml           # Render deployment config
├── docs/
│   └── methodology.md    # Detailed methodology & approach
└── README.md
```

---

## 🔬 Methodology Overview

The pipeline uses a **Random Forest Classifier** trained on 22 hand-crafted features + 200 Bag-of-Words features (100 per question), totaling **222 features**.

### Feature Categories

| Category | Count | Description |
|---|---|---|
| **Basic Features** | 7 | Character lengths, word counts, common/total word ratio |
| **Token Features** | 8 | Common word/stopword/token ratios, first/last word match |
| **Length Features** | 3 | Absolute length diff, avg token length, longest common substring ratio |
| **Fuzzy Features** | 4 | QRatio, Partial Ratio, Token Sort Ratio, Token Set Ratio |
| **BoW Features** | 200 | Top-100 CountVectorizer features per question |

> 📖 For a deep dive into the methodology, see [`docs/methodology.md`](docs/methodology.md)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/imabhinav0709/Duplicate-Question-Pairs.git
cd Duplicate-Question-Pairs

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

---

## 🖥️ Application Features

- **Interactive UI** — Enter any two questions and get instant duplicate detection
- **Confidence Score** — Probability-based prediction with visual progress bar
- **Entity Conflict Guard** — Prevents false positives when distinct subjects are detected (e.g., "India" vs "USA")
- **Adjustable Threshold** — Fine-tune the similarity threshold via sidebar slider
- **Keyword Breakdown** — Expandable section showing common terms, unique terms, and Jaccard similarity

---

## 📊 Example Predictions

| Question 1 | Question 2 | Prediction |
|---|---|---|
| How can I be a good programmer? | What should I do to become a good programmer? | ✅ Duplicate |
| What is the capital of India? | What is the capital of USA? | ❌ Not Duplicate |
| How do I learn Python? | What is the best way to learn Python? | ✅ Duplicate |

---

## 🌐 Deployment

This app is configured for deployment on **Render**:

1. Push the repo to GitHub
2. Create a new **Web Service** on [Render](https://render.com)
3. Connect the GitHub repo — Render auto-detects `render.yaml`
4. Deploy on the **Free** tier

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **ML Model:** Random Forest Classifier (scikit-learn)
- **NLP:** CountVectorizer (Bag of Words), FuzzyWuzzy, NLTK Stopwords
- **Text Processing:** BeautifulSoup, Regex, Contraction Expansion
- **Deployment:** Render

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

## 👤 Author

**Abhinav Maddipati** — [@imabhinav0709](https://github.com/imabhinav0709)
