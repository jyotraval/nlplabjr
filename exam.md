# NLP Lab Exam Playbook

Nov 17, 2026 · @Jyot

## How to use this in the exam

Every exam task will be one of your seven experiments with a twist, so the method is: **find the experiment, copy its cell, change one thing.** Section 3 has the twists most likely to appear, each as paste-ready code that was run before it went in here.

1. Match the question's keyword in section 2 to a repo file and cell number.
2. Copy that cell and change only what the question changes (dataset, model, N, a metric).
3. If the question joins two experiments, use the matching recipe in section 3.

### Repo map

All files are in `github.com/jyotraval/nlplabjr`. Cell numbers count from 0 in the notebook.

| Exp | Topic | Your file | Main libraries | Lab dataset | Prof source |
| --- | --- | --- | --- | --- | --- |
| 1 | ML classifiers on text | `exp1.ipynb` | sklearn: TfidfVectorizer, LogisticRegression, SVC, DecisionTree, metrics | 10 movie reviews, labels 1/0 | NLPLAB\_Exp1.pptx |
| 2 | Text files and PDFs | `exp2.ipynb` (+ `sample.txt`, `pdf1.pdf`, `pdf2.pdf`, `merge_pdfs*.pdf`) | built-in `open`, pypdf | sample.txt, two PDFs | NLP\_EXP2.pdf |
| 3 | Tokenize, stem, lemmatize | `exp3.ipynb` | split, re, pandas, nltk (Porter, Lancaster, Snowball, WordNet) | "The striped bats are hanging on their feet for best" | Experiment-3.pdf + 3 notebooks |
| 4 | RNN next-word prediction | `exp4ab.ipynb` (context pairs, Top-5) and `Experiment_4_RNN.ipynb` (1-word input, `generate_seq`) | TensorFlow/Keras: Tokenizer, Embedding, SimpleRNN | Part A: 10 custom sentences. Part B: Tiny Shakespeare | Experiment 4.pptx + Experiment\_4\_RNN.ipynb |
| 5A | spaCy preprocessing + TF-IDF | `exp5ab.ipynb` cells 1-8 | spaCy `en_core_web_sm`, sklearn TfidfVectorizer | 14 mixed-topic sentences | NLPLAB\_Exp5.pptx |
| 5B | Pre-trained sentiment | `exp5ab.ipynb` cells 9-17 | TextBlob, NLTK VADER, sklearn metrics, seaborn | 12 labelled sentences | NLPLAB\_Exp5.pptx |
| 6 | N-gram language models | `exp6.ipynb` (MLE + Laplace) and `exp6v2.ipynb` (padded perplexity, averages) | `nltk.lm`: MLE, Laplace, padded\_everygram\_pipeline | one 20-token "the cat sat..." string | NLPLAB\_Exp6.pptx |
| 7 | BERT text classification | `Experiment_7_Text_Classification_BERT.ipynb` | transformers, datasets, evaluate | IMDB (2000 train, 500 test) | same notebook |

Three things in the prof's material are **not in your repo**, so check them before the exam: the Exp 2 mini-exercises (rotate PDF, watermark PDF, Tkinter/menu GUI), `Stemming.ipynb` and `Lemmatization.ipynb` as separate files (their code is merged into your `exp3.ipynb`), and the Shakespeare code in `Experiment_4_RNN.ipynb` (your copy has only the markdown heading; the working Shakespeare code is in `exp4ab.ipynb` Part B). Section 3 covers the Exp 2 gaps.

## Find it fast: keyword to file and cell

If the question says it, open the file on the right. Cell numbers start at 0.

| Question mentions | Open | Cell(s) |
| --- | --- | --- |
| TF-IDF, vocabulary size, feature matrix | `exp1.ipynb` / `exp5ab.ipynb` | exp1 cell 1; exp5ab cell 7 |
| stop words in TF-IDF | `exp1.ipynb` | cell 1 (`stop_words='english'`) |
| train/test split | `exp1.ipynb` | cell 2 (`test_size=0.3`, `random_state=42`) |
| Logistic Regression, SVM, Decision Tree | `exp1.ipynb` | cell 2 (`classifiers` dict) |
| confusion matrix, accuracy, precision, recall, F1 | `exp1.ipynb`, `exp5ab.ipynb` | exp1 cell 2; exp5ab cell 16 (`evaluate_model`, works for any label names) |
| read, write, append a text file | `exp2.ipynb` | cells 1, 2, 3 |
| count lines, words, characters | `exp2.ipynb` | cell 4 |
| read, merge, split a PDF | `exp2.ipynb` | cells 5, 6-7, 8 (rotate, watermark: not in repo, see recipe R7) |
| split(), regex, pandas tokenizing | `exp3.ipynb` | cell 0 |
| sentence tokenizing, paragraph tokenizing | `exp3.ipynb` | cells 1, 2 |
| Porter, Lancaster, Snowball stemmer | `exp3.ipynb` | cell 3 |
| WordNet lemmatizer, POS-aware lemma | `exp3.ipynb` | cell 5 (output pasted in cell 4) |
| RNN, Embedding, SimpleRNN, softmax | `exp4ab.ipynb` | Part A cell 11; Part B cell 25 |
| context-target pairs, padding, `pad_sequences` | `exp4ab.ipynb` | cell 9 (A), cell 23 (B) |
| Top-1 / Top-5 accuracy, loss curves | `exp4ab.ipynb` | compile cell 11, plots cell 14 (A) and 26 (B) |
| predict next word, seed phrase | `exp4ab.ipynb` | cell 16 (A), cell 28 (B) |
| generate a sentence from one word | `Experiment_4_RNN.ipynb` | cells 27-29 (`generate_seq`) |
| one-hot labels, `categorical_crossentropy` | `Experiment_4_RNN.ipynb` | cells 23, 25 |
| Shakespeare, bigger corpus, `NUM_LINES` | `exp4ab.ipynb` | cells 19-28 |
| spaCy tokens, stop word flag, lemma, POS | `exp5ab.ipynb` | cell 5 |
| named entities, `spacy.explain` | `exp5ab.ipynb` | cell 5 |
| TextBlob polarity, subjectivity | `exp5ab.ipynb` | cell 12 |
| VADER compound score, thresholds | `exp5ab.ipynb` | cell 14 |
| bigram, trigram, `padded_everygram_pipeline`, MLE | `exp6.ipynb` | cells 2, 4 |
| next-token probability for a context | `exp6.ipynb` | cell 6 (`model.score(word, context)`) |
| perplexity, zero probability, Laplace smoothing | `exp6.ipynb` | cell 8 (best: shows MLE `inf` vs Laplace) |
| average perplexity over test sentences | `exp6v2.ipynb` | cell 9 |
| generate text from an n-gram model | `exp6.ipynb` | cell 6 (`model.generate`) |
| DistilBERT, tokenizer, `max_length` | `Experiment_7_Text_Classification_BERT.ipynb` | cells 1, 4, 5 |
| `Trainer`, `TrainingArguments`, epochs, learning rate | same | cells 7-14 |
| accuracy metric with `evaluate` | same | cell 11 |
| predict sentiment of a new sentence | same | cell 15 |

## Likely combinations and dataset swaps

A small-change exam means: a different dataset, two experiments joined, or one component swapped. The recipes below name the cell to start from and the lines to change. Unless a recipe says otherwise, I ran it on your lab data or a toy corpus and the "expect" lines are what came out.

| Chain | What it joins |
| --- | --- |
| Exp 2 → Exp 3 → Exp 5A → Exp 1 | read a file or PDF, tokenize and stem, spaCy lemmas + TF-IDF, then classify |
| Exp 1 ↔ Exp 5B ↔ Exp 7 | one labelled set, three sentiment methods: trained classic model, rule-based, transformer |
| Exp 3 / Exp 5A → Exp 6 | tokenized text becomes the n-gram training corpus |
| Exp 6 ↔ Exp 4 | both predict the next word: compare n-gram probabilities with RNN top-5 |

### R1. Preprocess first, then classify (Exp 3 or 5A into Exp 1)

The Exp 1 slide lists "Lemmatize" as a step, but the code only uses `stop_words='english'`, so this is a natural twist. Start from `exp1.ipynb` cell 1 and replace the `TfidfVectorizer` line. A callable `analyzer` replaces sklearn's own tokenizing and stop-word removal, so do both inside it.

```python
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
ps = PorterStemmer()
def prep_stem(text):
    toks = re.findall(r"\b\w+\b", text.lower())
    return [ps.stem(t) for t in toks if t not in ENGLISH_STOP_WORDS]
tfidf_vectorizer = TfidfVectorizer(analyzer=prep_stem)   # was: TfidfVectorizer(stop_words='english')

# spaCy lemma version (Exp 5A)
import spacy; nlp = spacy.load("en_core_web_sm")
def prep_lemma(text):
    return [t.lemma_.lower() for t in nlp(text) if t.is_alpha and not t.is_stop]
tfidf_vectorizer = TfidfVectorizer(analyzer=prep_lemma)
```

Expect on your 10 reviews: vocabulary 48 originally, 47 with stemming, 48 with spaCy lemmas. With a 3-review test set, accuracy can only be 0, 33, 67 or 100%. All three classifiers scored 33% in my run, so do not read that as a bug.

### R2. New dataset: a CSV, or more than two classes

Only the loading lines and the metric averaging change. Checked on a 3-class toy set (sport, business, health): accuracy 1.0.

```python
df = pd.read_csv("data.csv")                       # columns: text, label
X = tfidf_vectorizer.fit_transform(df["text"].astype(str)); y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
"Logistic Regression": LogisticRegression(max_iter=1000)      # in the classifiers dict
precision_score(y_test, y_pred, average="macro", zero_division=0)   # same for recall_score, f1_score
print(classification_report(y_test, y_pred))        # per-class table for the write-up
```

- Binary labels 0/1 work with the default metrics. Three or more classes need `average="macro"` (or `"weighted"`), otherwise sklearn raises a ValueError.
- String labels in a binary task need `pos_label="positive"`, as `exp5ab.ipynb` cell 16 already does.
- `stratify=y` needs at least 2 rows per class.

### R3. Trained model vs rule-based vs transformer on the same sentences (Exp 1, 5B, 7)

Reuse `evaluate_model(name, y_true, y_pred)` from `exp5ab.ipynb` cell 16: it takes any list of "positive"/"negative" predictions, so one function scores every method.

```python
Xtr, Xte, ytr, yte = train_test_split(texts, y_true, test_size=0.34, random_state=42, stratify=y_true)
vec = TfidfVectorizer(stop_words="english")
lr = LogisticRegression().fit(vec.fit_transform(Xtr), ytr)
evaluate_model("LogReg", yte, lr.predict(vec.transform(Xte)))
evaluate_model("TextBlob", yte, ["positive" if TextBlob(t).sentiment.polarity >= 0 else "negative" for t in Xte])
# VADER with the usual three-way thresholds instead of >= 0
c = sia.polarity_scores(t)["compound"]
label = "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral"
```

Expect on your 12 sentences, all of them: TextBlob accuracy 0.8333 and F1 0.857 (it calls two negatives positive because their polarity is 0.00 and 0.13), VADER accuracy 1.0. These match your saved notebook output. On a 5-sentence test split the trained logistic regression scored 0.4 against TextBlob's 0.8, because 7 training sentences are too few to learn from while lexicon rules need no training. For the transformer leg, use `predict()` from R9.

### R4. RNN variants: other layer, other corpus, generate several words (Exp 4)

Start from `exp4ab.ipynb`. All four layer types trained on a 13-sentence corpus; prediction and generation were run on the last one, BiLSTM.

```python
# (1) other recurrent layer: change ONE line in the model (cell 11 / cell 25)
from tensorflow.keras.layers import LSTM, GRU, Bidirectional
SimpleRNN(128, activation="tanh")   ->   LSTM(128)   or   GRU(128)   or   Bidirectional(LSTM(128))

# (2) other corpus: only the list of lines changes (cell 5 for Part A, cell 21 for Part B)
lines = df["headline"].tolist()                  # news headlines, college notices, IMDB review text ...
lines = [clean_text(l) for l in lines if len(l.split()) >= 2]
train_lines, val_lines = train_test_split(lines, test_size=0.2, random_state=42)
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")      # cap the vocabulary for a big corpus

# (3) generate N words: feed each prediction back in (uses predict_top_k_words from cell 16)
def generate(seed, n_words=5):
    text = clean_text(seed)
    for _ in range(n_words):
        text += " " + predict_top_k_words(text, top_k=1)[0][0]
    return text
```

- Longer or shorter context: change `MAX_CONTEXT_WORDS` (and `CONTEXT_LENGTHS`). The model `Input(shape=...)` and the predict function both read it.
- Expect on small data: SimpleRNN, LSTM, GRU and BiLSTM all reached 80-90% training top-1 but only about 25% validation top-5. Swapping the layer does not fix too little data.
- `generate("natural language", 4)` returned "natural language processing includes text classification" on that corpus.

### R5. N-gram variants: other N, other smoothing, held-out perplexity, corpus from a file (Exp 6)

Start from `exp6.ipynb` cells 2-8. The pattern for any N is: `padded_everygram_pipeline(n, ...)`, a model built with the same `n`, and a context of the last `n-1` tokens.

```python
# (1) other N and other smoothing: change n and the model class
data, vocab = padded_everygram_pipeline(4, tokenized_train)
lm = Laplace(4)            # MLE(4) | Laplace(4) | Lidstone(0.1, 4) | KneserNeyInterpolated(4)
lm.fit(data, vocab)
lm.score("sat", tuple(tokens[-3:]))       # context = last N-1 tokens

# (2) perplexity on HELD-OUT sentences: pad, take n-grams, flatten
from itertools import chain
test_ngrams = list(chain.from_iterable(ngrams(pad_both_ends(s, n), n) for s in tokenized_test))
lm.perplexity(test_ngrams)

# (3) corpus from a text file (Exp 2 + Exp 3 + Exp 6)
raw = open("sample.txt", encoding="utf-8").read()
tokenized_train = [word_tokenize(s.lower()) for s in sent_tokenize(raw)]

# (4) readable generated text: drop the padding symbols
" ".join(w for w in lm.generate(8, text_seed=["the", "cat"]) if w not in ("<s>", "</s>"))
```

On an 8-sentence cat/dog/bird corpus with 6 training and 2 test sentences, I got: MLE perplexity `inf` for every N (a test n-gram was never seen, probability 0). Laplace gave 13.4, 9.0, 10.0, 10.4 for N = 1 to 4, and Lidstone(0.1) gave 19.6, 6.3, 6.4, 6.5. The pattern to quote: MLE breaks on unseen n-grams, smoothing makes perplexity finite, and from N = 2 up, a small add-m (Lidstone) gave lower perplexity than add-one (Laplace) on this sparse data. KneserNey also returned `inf` on a corpus this small, so use it only with real text.

### R6. spaCy and TF-IDF variants (Exp 5A)

Start from `exp5ab.ipynb` cells 5 and 7. `docs = [nlp(s) for s in sentences]` gives you one `Doc` per sentence to slice any way.

```python
[t.lemma_ for t in d if t.pos_ == "NOUN"]               # nouns only; also "VERB", "PROPN", "ADJ"
[t.text for t in d if t.is_alpha and not t.is_stop]      # no stop words, no punctuation
[(e.text, e.label_) for d in docs for e in d.ents]       # every entity; Counter(label) for a summary

# TF-IDF on lemmas without stop words, then the most similar sentence
clean = [" ".join(t.lemma_.lower() for t in d if t.is_alpha and not t.is_stop) for d in docs]
vec = TfidfVectorizer(); M = vec.fit_transform(clean)
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(M); np.fill_diagonal(sim, 0)
sim[0].argmax() + 1, sim[0].max()                        # sentence number and score
```

Expect on the 14 lab sentences: vocabulary 81 with plain `TfidfVectorizer()` as in your cell 7, 72 after lemmas and stop-word removal. Entity counts: ORG 3, GPE 2, PERSON 1, PRODUCT 1, CARDINAL 1, DATE 1. Sentence 1 is closest to sentence 2 (cosine 0.158, through the shared word "team"); 84 of the 91 sentence pairs score exactly 0 because the sentences share almost no content words.

### R7. Exp 2 extras from the prof's slides that are missing from your repo

The prof's mini exercise is merge, split, rotate and watermark, and the learning outcomes include a menu-driven application. Your `exp2.ipynb` has merge and split only. The PDF steps ran on your `merge_pdfs.pdf`; the menu is a short skeleton that I ran in a longer form.

```python
from pypdf import PdfReader, PdfWriter
r, w = PdfReader("merge_pdfs.pdf"), PdfWriter()
for p in r.pages: p.rotate(90); w.add_page(p)             # rotate: multiples of 90, clockwise
w.write("rotated.pdf")

# watermark: build a one-page stamp with reportlab (pip install reportlab), merge it onto each page
import io; from reportlab.pdfgen import canvas
buf = io.BytesIO(); c = canvas.Canvas(buf); c.setFont("Helvetica-Bold", 60); c.setFillGray(0.8)
c.translate(100, 200); c.rotate(45); c.drawString(0, 0, "CONFIDENTIAL"); c.save(); buf.seek(0)
stamp = PdfReader(buf).pages[0]; w = PdfWriter()
for p in PdfReader("merge_pdfs.pdf").pages: p.merge_page(stamp); w.add_page(p)
w.write("watermarked.pdf")

# one file per page, and PDF -> text file -> word frequency
for i, p in enumerate(PdfReader("merge_pdfs.pdf").pages, 1):
    ww = PdfWriter(); ww.add_page(p); ww.write(f"page_{i}.pdf")
open("out.txt", "w").write("\n".join(p.extract_text() for p in PdfReader("merge_pdfs.pdf").pages))
from collections import Counter; import re
Counter(re.findall(r"\b\w+\b", open("out.txt").read().lower())).most_common(5)

# menu-driven skeleton
while True:
    c = input("1 write  2 read  3 append  4 stats  5 exit: ")
    if c == "1": open("sample.txt", "w").write(input("text: "))
    elif c == "2": print(open("sample.txt").read())
    elif c == "3": open("sample.txt", "a").write("\n" + input("text: "))
    elif c == "4": d = open("sample.txt").read(); print(len(d.splitlines()), len(d.split()), len(d))
    elif c == "5": break
```

If the exam gives a different PDF, only the file names change. `extract_text()` returns an empty string for scanned (image-only) pages, so check that before blaming the code.

### R8. Exp 3 variants: stemmer table, stop words, lemma with the right POS

Start from `exp3.ipynb` cells 3 and 5.

```python
words = nltk.word_tokenize(text)
pd.DataFrame({"word": words,
              "Porter": [porter.stem(w) for w in words],
              "Lancaster": [lancaster.stem(w) for w in words],
              "Snowball": [snowball.stem(w) for w in words]})

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS          # stop words with no download
[w for w in words if w.lower() not in ENGLISH_STOP_WORDS]
# NLTK list instead: nltk.download('stopwords'); set(stopwords.words('english'))

lemmatizer.lemmatize("hanging", pos="v")      # pos: 'n' noun (the default), 'v' verb, 'a' adjective, 'r' adverb
```

Expect on "The striped bats are hanging on their feet for best": Porter and Snowball give stripe, bat, hang; Lancaster differs on two words: it keeps "striped" and cuts "are" to "ar". The WordNet lemmatizer leaves "are" and "hanging" unchanged because its default POS is noun; with `pos="v"` they become "be" and "hang", as in your cell 4 output. Stemming is rule-based chopping and can produce non-words; lemmatization uses a dictionary and returns real words.

### R9. BERT variants: faster run, another dataset, more metrics, confusion matrix, batch prediction (Exp 7)

Your saved run (2000 train, 500 test, `max_length` 128, 3 epochs) took 4614 seconds, about 77 minutes, on CPU with no accelerator. In a timed lab, shrink it first.

```python
# lines you normally change
model_name = "distilbert-base-uncased"                  # or "bert-base-uncased"
DATASET, TEXT_COL, LABEL_COL, NUM_LABELS = "rotten_tomatoes", "text", "label", 2
#   ("ag_news", "text", "label", 4)   ("emotion", "text", "label", 6)   ("stanfordnlp/imdb", "text", "label", 2)
N_TRAIN, N_TEST, MAX_LEN, EPOCHS = 400, 100, 64, 1
ds = load_dataset(DATASET)          # print(ds): the eval split may be called "validation", not "test"

# accuracy + precision + recall + F1 (replaces the evaluate.load("accuracy") cell)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary" if NUM_LABELS == 2 else "macro", zero_division=0)
    return {"accuracy": accuracy_score(labels, preds), "precision": p, "recall": r, "f1": f1}

out = trainer.predict(test_tok)                          # confusion matrix after training
confusion_matrix(out.label_ids, out.predictions.argmax(-1))

def predict(texts, id2label):                            # list of sentences -> (label, probability)
    enc = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(model.device)
    with torch.no_grad(): probs = torch.softmax(model(**enc).logits, dim=-1)
    return [(id2label[i], round(float(probs[k, i]), 3)) for k, i in enumerate(probs.argmax(-1).tolist())]
```

- `num_labels` must equal the number of classes of the new dataset, and the label column must be named `label` (use `rename_column`).
- Drop the text column before `set_format('torch')`, exactly as in your cell 6.
- If the model sits on a GPU, move the inputs with `.to(model.device)` as above; your original cell 15 would otherwise fail with a device mismatch.
- What was and was not tested: the whole recipe ran end to end on a tiny, randomly initialised DistilBERT with synthetic text, which proves the code path, not accuracy. My sandbox could not download real models or datasets, and it ran transformers 5.23 where your notebook pins 4.56.1.

## Per-experiment sheets

Each sheet gives what the code does, what to change first, the output saved in your repo, and the sentence to say if asked.

### Exp 1: ML classifiers on text (`exp1.ipynb`)

- **Does:** `TfidfVectorizer(stop_words='english')`, a 70/30 split (`random_state=42`), then LogisticRegression, SVC and DecisionTree, scored with confusion matrix, accuracy, precision, recall and F1.
- **Change first:** `test_size`; classifier arguments such as `SVC(kernel="linear")` or `DecisionTreeClassifier(max_depth=3)`; add `MultinomialNB()` to the dict; `TfidfVectorizer(ngram_range=(1, 2))` (87 features on your data) or `CountVectorizer` (48). All of these ran.
- **Saved output:** vocabulary 48, matrix shape (10, 48). All three models print confusion matrix `[[0 2] [0 1]]`, accuracy 0.333, precision 0.333, recall 1.0, F1 0.5, because each predicted "positive" for all 3 test reviews.
- **Say:** TF-IDF is term frequency times inverse document frequency, so words common to every document get a low weight. Precision = TP / (TP + FP), recall = TP / (TP + FN), F1 is their harmonic mean.

### Exp 2: text files and PDFs (`exp2.ipynb`)

- **Does:** file modes `r`, `w`, `a`, `x`; `with open` closes the file; statistics from `splitlines()`, `split()`, `len()`; `PdfReader` and `extract_text()` to read, `PdfWriter.append` to merge, a page-range loop to split.
- **Saved state:** `sample.txt` holds 3 lines, 11 words, 60 characters after the append cell.
- **Watch:** cell 5 reads `merge_pdfs1.pdf`, which cell 8 creates, so on a fresh folder run the cells in the order merge, split, read. The prof's slides say PyPDF2; you use `pypdf`, the maintained successor with the same `PdfReader` and `PdfWriter` names. The old `PdfFileReader` name fails in recent PyPDF2.
- **Say:** `'w'` overwrites, `'a'` appends, `'x'` fails if the file already exists.

### Exp 3: tokenize, stem, lemmatize (`exp3.ipynb`)

- **Does:** three word tokenizers (`split()`, regex `\b\w+\b`, pandas `.str.split()`), `sent_tokenize`, a paragraph splitter on blank lines, then Porter, Lancaster, Snowball and WordNet.
- **Saved output:** `split()` leaves punctuation attached ("tokenization."); the regex turns "Here's" into "here" and "s". Lemmatizer: bats to bat, feet to foot, and "are" to "be" only with `pos='v'`.
- **Watch:** the `nltk.download(...)` lines are commented out in cells 3 and 5 (cell 1 still downloads punkt\_tab); run them once (`punkt_tab`, `wordnet`, `omw-1.4`). `stopwords` is imported in cell 3 but never used.
- **Say:** a stem is a chopped suffix and may not be a word; a lemma is a dictionary form and needs the part of speech.

### Exp 4: RNN next-word prediction (two notebooks)

|  | `Experiment_4_RNN.ipynb` (prof's) | `exp4ab.ipynb` (slides) |
| --- | --- | --- |
| Input | the 1 previous word, `Input(shape=(1,))` | up to 6 words (Part A) or 7 (Part B), padded on the left |
| Labels, loss | one-hot, `categorical_crossentropy` | integers, `sparse_categorical_crossentropy` |
| Layers | Embedding 10, SimpleRNN 50, softmax | Embedding 64, SimpleRNN 128, Dense 64, Dropout, softmax |
| Metrics | accuracy | top-1 and top-5 accuracy |
| Predict | `generate_seq(model, tokenizer, "natural", 3)` chains argmax words | `predict_top_k_words("natural language")` returns 5 words with probabilities |

- **Saved output, Part A:** vocabulary 33, `X_train` (132, 6), 132 pairs; validation top-1 0.263, top-5 0.316, so it overfits 10 sentences.
- **Saved output, Part B:** 12,750 training and 2,250 validation lines, vocabulary 5,000; by epoch 10 validation top-1 is about 0.11 and top-5 about 0.26. "the king" gives of (0.278), is, o', richard, that; "i will" gives not (0.141), be, you, i, have.
- **Why these choices:** `mask_zero=True` makes the RNN ignore padding zeros; sparse labels avoid a one-hot array that the prof's notebook says would be about 18.7 GB for full Shakespeare; top-5 is reported because many next words are reasonable. The slides add: if every seed gives nearly the same words, the model has learned only word frequency.

### Exp 5A and 5B: spaCy + TF-IDF, then pre-trained sentiment (`exp5ab.ipynb`)

- **5A does:** per token `text`, `is_stop`, `lemma_`, `pos_`; per sentence `doc.ents` with `ent.label_` and `spacy.explain(label)`; then `TfidfVectorizer()` on the raw sentences. I re-ran it: vocabulary 81, matrix (14, 81). "The Gujarat Technical University" is ORG and "Gandhinagar" is GPE, as in the slide.
- **5B does:** TextBlob `polarity` (-1 to 1) and `subjectivity` (0 to 1), VADER `compound` (-1 to 1), each thresholded at 0, then `evaluate_model`.
- **Saved output, 5B:** TextBlob accuracy 0.8333, precision 0.75, recall 1.0, F1 0.857; VADER 1.0 on all four. TextBlob's two errors are negative sentences with polarity 0.00 and 0.13.
- **Say:** TextBlob and VADER are lexicon-based, so they need no training data. VADER also reacts to capital letters, punctuation ("good!") and negation ("not bad").

### Exp 6: n-gram language models (`exp6.ipynb`, `exp6v2.ipynb`)

- **Does:** `padded_everygram_pipeline(n, tokens)` returns padded n-grams plus a vocabulary; `MLE(n)` or `Laplace(n)` is fitted on them; `.score(word, context)` gives P(word | context), `.generate()` samples text, `.perplexity(ngrams)` scores held-out text.
- **Saved output:** vocabulary 17 tokens (14 words plus `<s>`, `</s>`, `<UNK>`). After "cat" the bigram model gives sat, ran and "." 0.333 each; after "the cat" the trigram model gives sat 0.5, "." 0.5, ran 0. Perplexity for "the cat sat": bigram MLE 2.4495, trigram MLE 2.0, Laplace 8.3666 and 9.5. For "the cat ran": trigram MLE `inf`, Laplace trigram 19.0.
- **Say:** longer context sharpens predictions but needs more data; MLE gives probability 0 to anything unseen, so perplexity becomes infinite, and smoothing moves some probability to unseen n-grams.

```latex
PP(W) = P(w_1 w_2 \dots w_N)^{-1/N} = 2^{-\frac{1}{N}\sum_{i=1}^{N}\log_2 P(w_i \mid w_{i-n+1}^{i-1})}
```

### Exp 7: BERT text classification (`Experiment_7_Text_Classification_BERT.ipynb`)

- **Does:** `distilbert-base-uncased` fine-tuned on 2,000 IMDB reviews and tested on 500; tokenizer with `padding=True, truncation=True, max_length=128`; `Trainer` with learning rate 2e-5, 3 epochs, batch size 8, weight decay 0.01; accuracy by `evaluate`.
- **Saved output:** 750 training steps, training loss 0.3223, runtime 4,614 s. The saved copy has no output for the evaluation cell (14) or the prediction cell (15), so the final accuracy is not recorded anywhere in the repo.
- **Say:** BERT is a pre-trained transformer; fine-tuning adds a classification head and trains all weights on labelled data. DistilBERT is a smaller, faster distilled version. `num_labels` is the number of classes.

## Traps in your notebooks, and a pre-exam checklist

The first row matters most: a conclusion in your repo disagrees with its own output.

| Where | What goes wrong | Fix |
| --- | --- | --- |
| `exp6v2.ipynb` cells 9 and 12 | The conclusion says trigram perplexity is lower, but cell 9 prints `inf` for both models on "the cat sat on the mat ." and for the trigram on "the dog barked at the cat ."; the averages are `inf`. Cause: the training text is one long token list, so `<s>` only precedes the first "the cat", and the test n-grams ("mat", ".") and (`<s>`, "the", "dog") were never seen. | Use `exp6.ipynb` cell 8 (Laplace), or call `sent_tokenize` on the training text first. I checked that fix: "the dog barked at the cat ." then gives bigram 1.4877 and trigram 1.1665, which supports the conclusion. The "mat ." sentence stays `inf` under MLE because that bigram never occurs. |
| `exp1.ipynb` cell 2 | All three classifiers print identical scores. | Not a bug: 10 reviews give a 3-review test set and every model predicts "positive". Say this in the conclusion. |
| `Experiment_7_...BERT.ipynb` cell 1 | Your saved output shows `RemoteEntryNotFoundError` when loading the tokenizer on your Windows machine ("additional\_chat\_templates does not exist"). | The commented pip line in cell 0 pins `transformers==4.56.1` and `tokenizers==0.22.0`, which looks like your workaround; I could not reproduce or confirm it. The training run itself came from a Linux environment such as Colab. Test cell 1 on the exam machine early. |
| `Experiment_7_...BERT.ipynb` cells 14, 15 | No saved output, so no final accuracy or sample prediction is recorded. | Re-run, or at least know the shape of the result: `{'eval_loss': ..., 'eval_accuracy': ...}`. |
| `exp5ab.ipynb` cells 1 and 7 | `!pip install`, `!python -m spacy download` and `display(...)` only work inside Jupyter. | In a plain script, delete the `!` lines and use `print(df.head())`. |
| `exp5ab.ipynb` cells 12, 14 | The threshold `>= 0` labels a score of exactly 0 as positive, which causes two of TextBlob's errors. | Mention it; the usual VADER cut-offs are +0.05 and -0.05 (R3). |
| `exp2.ipynb` cells 5 and 8 | Cell 5 reads `merge_pdfs1.pdf`, which cell 8 creates. | Run merge, then split, then read. |
| `Experiment_4_RNN.ipynb` cell 37 | It contains the Shakespeare instructions but no code. | The working Shakespeare code is in `exp4ab.ipynb` Part B. |

### Before the exam

1. Run each notebook once, top to bottom, on the machine you will use, so models and data are cached.
2. Download the NLTK data once: `punkt`, `punkt_tab`, `wordnet`, `omw-1.4`, `vader_lexicon`, `stopwords`. In Python: `import nltk; [nltk.download(r) for r in ["punkt", "punkt_tab", "wordnet", "omw-1.4", "vader_lexicon", "stopwords"]]`.
3. Install the spaCy model: `python -m spacy download en_core_web_sm`.
4. Cache the big downloads: `tf.keras.utils.get_file` for Shakespeare, `AutoTokenizer` and `AutoModelForSequenceClassification` for distilbert, `load_dataset("stanfordnlp/imdb")`.
5. Print this doc together with the repo, and put the recipe files from R1-R9 in the repo before you print it, so the code is on paper next to your notebooks.
