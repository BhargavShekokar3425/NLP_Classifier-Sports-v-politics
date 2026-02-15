# Sport vs Politics — NLP Text Classifier

A modular Python pipeline that classifies sentences as **SPORT** or **POLITICS** using three ML algorithms and three feature representations — all implemented from scratch (no scikit-learn).

## Directory Structure

```
NLP_Classifier/
├── main.py            # Entry point (interactive menu or --auto mode)
├── interactive.py     # Live sentence classifier with logging
├── load_data.py       # Dataset loading & train/test splitting
├── preprocess.py      # Tokenisation, stopwords (NLTK), feature extractors
├── train_models.py    # Naive Bayes, Logistic Regression, KNN classifiers
├── evaluate.py        # Accuracy, Precision, Recall, F1, confusion matrix
├── visualize.py       # Matplotlib charts (bars, heatmap, confusion matrices)
├── sport.txt          # Dataset — one sport sentence per line
├── politics.txt       # Dataset — one politics sentence per line
├── README.md
└── output/            # Generated charts (created on first run)
```

## Features

| Feature Extraction | ML Classifier          |
|--------------------|------------------------|
| Bag-of-Words       | Multinomial Naive Bayes|
| TF-IDF             | Logistic Regression    |
| Word Bigrams       | K-Nearest Neighbours   |

All 9 combinations (3 × 3) are trained and evaluated automatically.

## Quick Start

```bash
# Install dependencies
pip install nltk matplotlib numpy

# Run in auto mode (train + evaluate + generate charts)
python main.py --auto

# Or use the interactive menu
python main.py
```

### Interactive Menu Options

| Option | Description |
|--------|-------------|
| 1 | Dataset statistics (class distribution, avg tokens) |
| 2 | Train & evaluate all 9 model combinations |
| 3 | Print comparison table (accuracy, P/R/F1) |
| 4 | Print confusion matrices |
| 5 | Predict a custom sentence |
| 6 | Generate visualization charts → `output/` |
| 7 | Reload dataset from disk |

## Interactive Classifier (`interactive.py`)

A dedicated live-prediction tool that trains all 9 models on startup and lets you classify sentences one after another.

```bash
python3 interactive.py
```

**What it does:**

1. Trains all 9 model combinations (3 features × 3 classifiers) automatically on startup.
2. Prompts you to enter a sentence.
3. Runs the sentence through **every model** and displays a ranked table showing:
   - Prediction (SPORT / POLITICS)
   - Confidence score
   - Per-class probabilities
   - Each model's test accuracy for reference
4. Shows the **majority vote** across all 9 models and the **most confident** model.
5. Logs every prediction with full details to `interactive.log`.

**In-session commands:**

| Command | Description |
|---------|-------------|
| *(any sentence)* | Classify it with all 9 models |
| `models` | Show trained model summary with test accuracies |
| `history` | Show all predictions made this session |
| `clear` | Clear prediction history |
| `quit` | Exit (session summary is logged) |

**Example session:**
```
Enter sentence → Virat Kohli scored a century against England

 #   Feature    Classifier             Prediction   Confidence   POLITICS     SPORT   TestAcc
─────────────────────────────────────────────────────────────────────────────────────────────
 ★1  TF-IDF     KNN                    SPORT            1.0000     0.0000    1.0000    89.29%
  2  BoW        NaiveBayes             SPORT            0.9998     0.0002    0.9998    82.14%
  ...  

  Majority vote  : SPORT  (9/9 models)
  Most confident : TF-IDF + KNN  →  SPORT  (conf 1.0000, test acc 89.29%)
```

## Visualizations

Running option **6** (or `--auto` mode) saves the following charts to `output/`:

| Chart | File |
|-------|------|
| Test accuracy bar chart | `accuracy_bars.png` |
| Accuracy heatmap (features × classifiers) | `accuracy_heatmap.png` |
| Per-class F1 scores | `f1_scores.png` |
| Confusion matrices grid | `confusion_matrices.png` |
| Train vs Test accuracy | `train_vs_test.png` |
| Class distribution pie chart | `class_distribution.png` |

## Dataset

The dataset consists of **70 sport** and **70 politics** sentences (one per line) focused on **Indian** sport and politics. The sentences were **hand-curated** covering topics such as:

- **Sport:** Cricket (IPL, ICC, Test), Olympics, football (ISL, I-League), badminton, boxing, hockey, athletics, wrestling
- **Politics:** Parliament (Lok Sabha, Rajya Sabha), elections, party politics (BJP, Congress, AAP, TMC, DMK), Supreme Court, government schemes, state politics

### Data References & Adaptation

The dataset sentences draw factual context from publicly available Indian news coverage. Key reference sources include:

- **ESPN Cricinfo** (https://www.espncricinfo.com/) — cricket statistics and match reports
- **Olympics.com** (https://olympics.com/) — Indian athletes' Olympic achievements
- **NDTV** (https://www.ndtv.com/) — Indian political news coverage
- **The Hindu** (https://www.thehindu.com/) — parliamentary proceedings and election reporting
- **Indian Express** (https://indianexpress.com/) — state politics and policy coverage
- **Press Information Bureau, Govt. of India** (https://pib.gov.in/) — government scheme announcements




## Dependencies

- **Python 3.8+**
- `nltk` — English stopword list
- `matplotlib` — chart generation
- `numpy` — used only in visualization

The ML algorithms and feature extractors are implemented **from scratch** without scikit-learn.

## License

Academic assignment — B23CS1008.
