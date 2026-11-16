#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 NLP LAB EXAM MASTER  --  ONE-FILE MAP OF github.com/jyotraval/nlplabjr
===============================================================================
 Purpose   : Open-book exam companion. Everything from all 7 experiments,
             broken into sequential steps, in the order you should do them.
             Every code block is tagged [SOURCE: <notebook>, cell N] so you
             can flip to the exact place in your printed repo hard-copy.
 Usage     :
     python nlp_lab_exam_master.py            -> prints this index + master map
     python nlp_lab_exam_master.py --exp1     -> runs Experiment 1 (etc.)
     python nlp_lab_exam_master.py --exp1 --exp6  -> runs several
     (import heavy libraries is inside each experiment function, so listing
      the index works even without tensorflow/spacy installed)
 Reading   : Just scroll. Sections are ordered EXACTLY as the course ran:
             EXP1 -> EXP7, then combos, swaps, traps, one-liners.
===============================================================================
 THE 30-SECOND EXAM TRIAGE (do this before writing anything)
===============================================================================
   Q1. What TYPE is the task?
         classification      -> EXP1 (classic ML)  or EXP7 (BERT)
         next-word / LM      -> EXP6 (counting)    or EXP4 (RNN)
         preprocess only     -> EXP3 (NLTK)        or EXP5A (spaCy)
         sentiment           -> EXP5B (TextBlob/VADER)
         files / PDFs        -> EXP2
   Q2. Which notebook is the base?            -> see MASTER MAP below
   Q3. What did the exam CHANGE?
         dataset / corpus    -> SECTION 9 swap recipes
         combination of exps -> SECTION 8 combo playbook (glue line included)
         N / metric / model  -> the "KNOBS" note inside each step
   Q4. Standard deliverable (same in every prof deck):
         Aim (handwritten) -> Code + results/graphs (print) -> Conclusion
         Four standard outputs: dataset/vocab sizes, model summary or
         frequency table, curves or probability table, predictions/metrics.
===============================================================================
 MASTER MAP: EXAM TASK -> YOUR CODE
===============================================================================
 | TASK KEYWORD                            | OPEN (repo file)    | SECTIONS |
 |-----------------------------------------|---------------------|----------|
 | TF-IDF + LogReg/SVM/DecisionTree + metr | exp1.ipynb (3 cells)| EXP1     |
 | write/read/append text, PDF merge/split | exp2.ipynb (8 cells)| EXP2     |
 |   rotate/watermark (deck mini-exercise) | exp2 extensions     | EXP2.5   |
 | tokenize (word/sent/para), stem, lemma  | exp3.ipynb (6 cells)| EXP3     |
 | RNN next-word (custom OR real corpus)   | exp4ab.ipynb        | EXP4     |
 |   teaching version, 1-word context      | Experiment_4_RNN    | EXP4-T   |
 | spaCy pipeline + TF-IDF full display    | exp5ab.ipynb  A     | EXP5A    |
 | TextBlob + VADER sentiment + evaluation | exp5ab.ipynb  B     | EXP5B    |
 | n-gram LM bigram/trigram, PPL, Laplace  | exp6.ipynb          | EXP6     |
 |   vocab-wide P table + avg perplexity   | exp6v2.ipynb        | EXP6-V2  |
 | DistilBERT fine-tune on IMDB            | Experiment_7_...   | EXP7     |
 | combining two experiments               | --                  | SECTION8 |
 | changed dataset / model / metric        | --                  | SECTION9 |
 | crash, inf, or weird output             | --                  | SECTION10|
===============================================================================
 NOTE ON REPRODUCIBILITY: np.random.seed(42) / random_state=42 / shuffle
 seed=42 appear everywhere in the notebooks -- keep them so numbers match.
===============================================================================
"""

import sys

# =============================================================================
# SECTION 0 -- SHARED SETUP (do once, at the top of any exam notebook)
# =============================================================================
# [SOURCE: exp3.ipynb cell 3-5, exp6.ipynb cell 2, exp5ab.ipynb cell 10]
# The single most common crash is a missing NLTK resource. Run this block
# FIRST in every exam; it is idempotent (already-downloaded = no-op).

NLTK_SETUP = """
import nltk
for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4', 'vader_lexicon']:
    nltk.download(pkg, quiet=True)
"""


def setup_nltk():
    """SECTION 0.1 -- run me before EXP3 / EXP5B / EXP6."""
    print("[setup] downloading NLTK resources (punkt, wordnet, vader, ...)")
    exec(NLTK_SETUP, globals())


# =============================================================================
# SECTION 1 -- EXPERIMENT 1: ML CLASSIFIERS ON TEXT (TF-IDF)
# [SOURCE: exp1.ipynb -- complete, 3 code cells]
# Prof deck: NLPLAB_Exp1.pptx (10-review mock sentiment dataset)
# Sequence: data -> TF-IDF -> split -> train 3 models -> metrics
# =============================================================================
def experiment_1():
    """Full pipeline. Any new dataset only replaces the `data` dict."""
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import (confusion_matrix, accuracy_score,
                                 precision_score, recall_score, f1_score)

    # ---- STEP 1: dataset ------------------------------------------------
    # [SOURCE: exp1.ipynb cell 0; prof deck slide 3]
    # KNOB: exam gives CSV -> df = pd.read_csv('exam.csv') then rename
    #       columns to review/sentiment, and map string labels to 0/1.
    data = {
        'review': [
            "I absolutely loved this movie, it was fantastic and brilliant!",
            "Horrible acting, absolute waste of time and money.",
            "The plot was gripping and the performances were spectacular.",
            "Tragic plot, terrible direction, and awful execution.",
            "An excellent masterpiece, highly recommended to everyone.",
            "Boring, slow paced, and predictable story line.",
            "Amazing cinematography with a beautiful musical score.",
            "Disappointed with the ending, it was completely ruined.",
            "The characters were well-developed and the script was witty.",
            "Avoid this film at all costs, it was a total disaster."
        ],
        'sentiment': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=Positive, 0=Negative
    }
    df = pd.DataFrame(data)
    print("Dataset successfully loaded. Total records:", len(df))

    # ---- STEP 2: TF-IDF vectorization -----------------------------------
    # [SOURCE: exp1.ipynb cell 1; prof deck slide 5]
    # stop_words='english' handles lowercasing+tokenization+stopword removal.
    # KNOB: ngram_range=(1,2) for bigrams, max_features=500 to cap vocab.
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    X = tfidf_vectorizer.fit_transform(df['review'])
    y = df['sentiment']
    print("Vocabulary Size:", len(tfidf_vectorizer.get_feature_names_out()))
    print("Feature Matrix Shape:", X.shape)

    # ---- STEP 3: split + train three classifiers ------------------------
    # [SOURCE: exp1.ipynb cell 2; prof deck slides 6-7]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    classifiers = {
        "Logistic Regression": LogisticRegression(),
        "SVM": SVC(),
        "Decision Tree": DecisionTreeClassifier()
    }
    # KNOB: add more models the same way, e.g.
    #   from sklearn.naive_bayes import MultinomialNB      -> "Naive Bayes"
    #   from sklearn.ensemble import RandomForestClassifier-> "Random Forest"
    for name, model in classifiers.items():
        model.fit(X_train, y_train)

    # ---- STEP 4: evaluate every model -----------------------------------
    for name, model in classifiers.items():
        y_pred = model.predict(X_test)
        print(f"\n----- {name} -----")
        print("Confusion Matrix:", *confusion_matrix(y_test, y_pred), sep='\n')
        print("Accuracy :", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred, zero_division=0))
        print("Recall   :", recall_score(y_test, y_pred, zero_division=0))
        print("F1-score :", f1_score(y_test, y_pred, zero_division=0))


# =============================================================================
# SECTION 2 -- EXPERIMENT 2: TEXT + PDF FILE HANDLING
# [SOURCE: exp2.ipynb -- 8 code cells]
# Prof deck: NLP_EXP2.pdf (mini-exercises: merge, split, rotate, watermark)
# Sequence: write -> read -> append -> stats -> PDF read -> merge -> split
# =============================================================================
def experiment_2():
    """Plain-text handling plus pypdf read/merge/split."""
    # ---- STEP 1: write / read / append a text file ----------------------
    # [SOURCE: exp2.ipynb cells 1-3; deck pages 4-6]
    with open("sample.txt", "w") as file:
        file.write("Hello, Python!\n")
        file.write("Welcome to file handling.")

    with open("sample.txt", "r") as file:          # read whole file
        content = file.read()
        print(content)
    with open("sample.txt", "r") as file:          # read line by line
        for line in file:
            print(line.strip())

    with open("sample.txt", "a") as file:          # append keeps old content
        file.write("\nThis is a new line.")

    # ---- STEP 2: text statistics -----------------------------------------
    # [SOURCE: exp2.ipynb cell 4; deck page 7]
    data = open('sample.txt').read()
    print(len(data.splitlines()))   # lines
    print(len(data.split()))        # words
    print(len(data))                # characters

    # ---- STEP 3: read a PDF ----------------------------------------------
    # [SOURCE: exp2.ipynb cell 5; deck page 8]  (pip install pypdf)
    from pypdf import PdfReader
    reader = PdfReader("merge_pdfs1.pdf")
    print("Number of pages:", len(reader.pages))
    for i, page in enumerate(reader.pages, start=1):
        print(f"\n--- Page {i} ---")
        print(page.extract_text())

    # ---- STEP 4: merge PDFs -----------------------------------------------
    # [SOURCE: exp2.ipynb cell 6; deck page 9]
    from pypdf import PdfWriter
    def PDFmerge(pdfs, output):
        pdfWriter = PdfWriter()
        for pdf in pdfs:
            pdfWriter.append(pdf)
        with open(output, 'wb') as f:
            pdfWriter.write(f)
    PDFmerge(['pdf1.pdf', 'pdf2.pdf'], 'merge_pdfs.pdf')

    # ---- STEP 5: split a PDF by page indices ------------------------------
    # [SOURCE: exp2.ipynb cell 8; deck page 10]
    # splits=[1] on a 2-page pdf -> page 0 in <name>0.pdf, page 1 in <name>1.pdf
    def PDFsplit(pdf, splits):
        reader = PdfReader(pdf)
        start, end = 0, splits[0]
        for i in range(len(splits) + 1):
            writer = PdfWriter()
            outputpdf = pdf.split('.pdf')[0] + str(i) + '.pdf'
            for page in range(start, end):
                writer.add_page(reader.pages[page])
            with open(outputpdf, "wb") as f:
                writer.write(f)
            start = end
            try:
                end = splits[i + 1]
            except IndexError:
                end = len(reader.pages)
    PDFsplit('merge_pdfs.pdf', [1])


# ---- SECTION 2.5: deck mini-exercises (rotate + watermark) -----------------
# [SOURCE: NLP_EXP2.pdf page 13 -- NOT in exp2.ipynb; these are the answers]
def experiment_2b():
    """Rotate a PDF 90 degrees and stamp a watermark (deck mini-exercises)."""
    from pypdf import PdfReader, PdfWriter

    # STEP 1: rotate every page (degrees must be a multiple of 90)
    reader = PdfReader("merge_pdfs.pdf")
    writer = PdfWriter()
    for page in reader.pages:
        page.rotate(90)
        writer.add_page(page)
    with open("rotated.pdf", "wb") as f:
        writer.write(f)

    # STEP 2: watermark -- stamp.pdf is any one-page PDF with the stamp/text
    reader = PdfReader("merge_pdfs.pdf")
    stamp = PdfReader("stamp.pdf").pages[0]
    writer = PdfWriter()
    for page in reader.pages:
        page.merge_page(stamp)          # overlays stamp onto the page
        writer.add_page(page)
    with open("watermarked.pdf", "wb") as f:
        writer.write(f)


# =============================================================================
# SECTION 3 -- EXPERIMENT 3: TOKENIZATION, STEMMING, LEMMATIZATION
# [SOURCE: exp3.ipynb -- 6 code cells]
# Prof deck: Experiment-3.pdf (+ prof's own Tokenization/Stemming/
# Lemmatization .ipynb files in the zip)
# Sequence: 3 tokenize methods -> NLTK sentence/paragraph -> 3 stemmers
#           -> lemmatizer with/without POS
# =============================================================================
def experiment_3():
    """Every preprocessing primitive from the course, in notebook order."""
    setup_nltk()

    # ---- STEP 1: three quick tokenization methods -------------------------
    # [SOURCE: exp3.ipynb cell 0; deck page 3]
    text = "This is a simple sentence for tokenization."
    tokens = text.split()                       # Method 1: str.split()
    print(tokens)

    import re
    text = "This is a sentence. Here's another one, with numbers 123."
    tokens = re.findall(r'\b\w+\b', text.lower())   # Method 2: regex
    print(tokens)

    import pandas as pd
    df = pd.DataFrame({"text": ["Ayush and Smrita are beautiful couple"]})
    df['tokens'] = df['text'].str.split()       # Method 3: pandas
    print(df['tokens'][0])

    # ---- STEP 2: NLTK sentence tokenization -------------------------------
    # [SOURCE: exp3.ipynb cell 1; deck page 4]
    import nltk
    sentence_data = ("The First sentence is about Python. The Second: about "
                     "Django. You can learn Python,Django and Data Analysis here. ")
    print(nltk.sent_tokenize(sentence_data))

    # ---- STEP 3: paragraph tokenization (regex) ---------------------------
    # [SOURCE: exp3.ipynb cell 2; deck page 5]
    def paragraph_tokenize(text):
        """A paragraph = block separated by one or more blank lines."""
        paragraphs = re.split(r'\n\s*\n', text.strip())
        return [p.strip() for p in paragraphs if p.strip()]

    input_text = """
    This is the first paragraph.
    It has two lines.

    This is the second paragraph.

    This is the third one, which
    also has two lines.
    """
    for i, para in enumerate(paragraph_tokenize(input_text), start=1):
        print(f"Paragraph {i}:", para)

    # ---- STEP 4: stemming with 3 stemmers side by side --------------------
    # [SOURCE: exp3.ipynb cell 3; deck page 6]
    from nltk.stem import (PorterStemmer, LancasterStemmer,
                           SnowballStemmer, WordNetLemmatizer)
    text = "The striped bats are hanging on their feet for best"
    words = nltk.word_tokenize(text)

    porter, lancaster = PorterStemmer(), LancasterStemmer()
    snowball = SnowballStemmer("english")
    for w in words:
        print(w, "->", porter.stem(w), lancaster.stem(w), snowball.stem(w))
    # NOTE for conclusion: Lancaster is the aggressive one ("hanging" ->
    # "hang" in Porter but "hang"/"hent" style over-stems elsewhere).

    # ---- STEP 5: lemmatization, default vs POS-tagged ---------------------
    # [SOURCE: exp3.ipynb cells 4-5; deck page 6-7]
    lemmatizer = WordNetLemmatizer()
    for w in words:
        print(w, "->", lemmatizer.lemmatize(w))                # noun default
    for w in words:
        print(w, "-> Noun:", lemmatizer.lemmatize(w, pos='n'),
              "Verb:", lemmatizer.lemmatize(w, pos='v'))
    # KEY EXAMPLE to quote: bats->bat, feet->foot (noun);
    # are->are (noun) but are->be (verb). Lemma depends on POS.


# =============================================================================
# SECTION 4 -- EXPERIMENT 4: RNN NEXT-WORD PREDICTION (PARTS A AND B)
# [SOURCE: exp4ab.ipynb -- THE journal version, use this one]
# Prof deck: Experiment 4.pptx (21 slides, Dr. Parita Shah)
# Sequence: imports -> A1 data -> A2 clean+vocab -> A3 pairs -> A4 model
#           -> A5 train+curves -> A6 predict -> B1..B5 same on Shakespeare
# =============================================================================
def experiment_4():
    """SimpleRNN next-word prediction, custom sentences then Shakespeare."""
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Input, Embedding, SimpleRNN,
                                         Dense, Dropout)
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.model_selection import train_test_split

    np.random.seed(42)                    # [cell 2] reproducibility
    tf.random.set_seed(42)

    # ======================================================================
    # PART A -- CUSTOM SENTENCES  (10 training / 3 validation)
    # ======================================================================

    # ---- STEP A1: define sentences ---------------------------------------
    # [SOURCE: exp4ab.ipynb cell 5; prof deck slide 6]
    training_sentences = [
        "students learn natural language processing",
        "students study natural language processing",
        "natural language processing includes text classification",
        "natural language processing includes next word prediction",
        "recurrent neural network predicts the next word",
        "the model predicts the next word",
        "the model learns word order",
        "students write python code",
        "the teacher demonstrates next word prediction",
        "the tokenizer converts words into token identifiers"
    ]
    validation_sentences = [
        "students test next word prediction",
        "natural language processing uses neural networks",
        "the model learns from input sequences"
    ]

    # ---- STEP A2: clean text + build vocabulary ---------------------------
    # [SOURCE: exp4ab.ipynb cell 7; prof deck slide 7]
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"[^a-z' ]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    training_sentences = [clean_text(s) for s in training_sentences]
    validation_sentences = [clean_text(s) for s in validation_sentences]

    tokenizer = Tokenizer(oov_token="<OOV>", lower=True)
    tokenizer.fit_on_texts(training_sentences)     # fit on TRAIN only!
    vocab_size = len(tokenizer.word_index) + 1     # +1: index 0 = padding
    index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}
    print("Vocabulary size:", vocab_size)

    # ---- STEP A3: context-target pairs (variable context lengths) ---------
    # [SOURCE: exp4ab.ipynb cell 9]
    MAX_CONTEXT_WORDS = 6
    MAX_SEQUENCE_LEN = MAX_CONTEXT_WORDS + 1
    CONTEXT_LENGTHS = (1, 2, 3, MAX_CONTEXT_WORDS)   # B uses (1, 2, 4, 7)

    def create_next_word_examples(sentences):
        sequences = []
        for sentence in sentences:
            token_ids = tokenizer.texts_to_sequences([sentence])[0]
            if len(token_ids) < 2:
                continue
            for target_pos in range(1, len(token_ids)):
                lengths = {min(c, target_pos) for c in CONTEXT_LENGTHS}
                for L in lengths:
                    sequences.append(token_ids[target_pos - L: target_pos + 1])
        padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LEN,
                               padding="pre", truncating="pre")
        X = padded[:, :-1]          # all but last = context
        y = padded[:, -1]           # last = target word id
        return X.astype("int32"), y.astype("int32")

    X_train, y_train = create_next_word_examples(training_sentences)
    X_val, y_val = create_next_word_examples(validation_sentences)

    # ---- STEP A4: build + compile SimpleRNN -------------------------------
    # [SOURCE: exp4ab.ipynb cell 11; prof deck slides 12-14]
    # mask_zero=True -> RNN ignores pre-padded zeros (viva favorite).
    model = Sequential([
        Input(shape=(MAX_CONTEXT_WORDS,), name="input_context"),
        Embedding(input_dim=vocab_size, output_dim=64,
                  mask_zero=True, name="word_embedding"),
        SimpleRNN(128, activation="tanh", dropout=0.10, name="simple_rnn"),
        Dense(64, activation="relu"),
        Dropout(0.20),
        Dense(vocab_size, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",    # y is an INDEX not one-hot
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="top1_accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy")
        ])
    model.summary()

    # KNOB: LSTM/GRU variant -> from tensorflow.keras.layers import LSTM
    #       and swap SimpleRNN(128) for LSTM(128). Shapes unchanged.

    # ---- STEP A5: train with callbacks + curves ---------------------------
    # [SOURCE: exp4ab.ipynb cells 13-14]
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)
    ]
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=100, batch_size=16,
                        callbacks=callbacks, verbose=1)
    print(model.evaluate(X_val, y_val, return_dict=True))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Part A: Model Loss'); plt.xlabel('Epochs'); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['top1_accuracy'], label='Train Top-1 Acc')
    plt.plot(history.history['val_top1_accuracy'], label='Val Top-1 Acc')
    plt.plot(history.history['top5_accuracy'], label='Train Top-5', linestyle='--')
    plt.plot(history.history['val_top5_accuracy'], label='Val Top-5', linestyle='--')
    plt.title('Part A: Accuracy'); plt.xlabel('Epochs'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.show()

    # ---- STEP A6: predict top-5 next words --------------------------------
    # [SOURCE: exp4ab.ipynb cell 16; prof deck slide 15]
    def predict_top_k_words(seed_text, top_k=5):
        cleaned_seed = clean_text(seed_text)
        token_ids = tokenizer.texts_to_sequences([cleaned_seed])[0]
        if not token_ids:
            raise ValueError("Seed phrase has no known words.")
        token_ids = token_ids[-MAX_CONTEXT_WORDS:]
        model_input = pad_sequences([token_ids], maxlen=MAX_CONTEXT_WORDS,
                                    padding="pre", truncating="pre")
        probs = model.predict(model_input, verbose=0)[0]
        ranked = np.argsort(probs)[::-1]
        predictions = []
        for idx in ranked:
            word = index_to_word.get(int(idx))
            if idx == 0 or word is None or word == "<OOV>":
                continue                     # skip pad + OOV
            predictions.append((word, float(probs[idx])))
            if len(predictions) == top_k:
                break
        return predictions

    for seed in ["natural language", "recurrent neural",
                 "next word", "the model"]:
        print(f"\nSeed: '{seed}'")
        for word, prob in predict_top_k_words(seed, top_k=5):
            print(f"  --> {word:15s} (prob: {prob:.4f})")

    # ======================================================================
    # PART B -- SHAKESPEARE (same pipeline, real corpus)
    # [SOURCE: exp4ab.ipynb cells 19-28; prof deck slide 16+]
    # ======================================================================

    # ---- STEP B1: download corpus -----------------------------------------
    url = ("https://storage.googleapis.com/download.tensorflow.org/"
           "data/shakespeare.txt")
    path_to_file = tf.keras.utils.get_file("shakespeare.txt", url)
    with open(path_to_file, "r", encoding="utf-8") as file:
        raw_text = file.read()

    # ---- STEP B2: clean lines + capped vocabulary -------------------------
    def clean_line(line):
        line = line.lower()
        line = re.sub(r"[^a-z' ]+", " ", line)
        line = re.sub(r"\s+", " ", line).strip()
        return line

    cleaned_lines = [clean_line(line) for line in raw_text.splitlines()]
    cleaned_lines = [ln for ln in cleaned_lines if len(ln.split()) >= 2]
    NUM_LINES = 15000          # KNOB: None = full corpus (slow!)
    selected_lines = cleaned_lines if NUM_LINES is None else cleaned_lines[:NUM_LINES]
    train_lines, val_lines = train_test_split(selected_lines,
                                              test_size=0.15, random_state=42)

    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")  # vocab cap
    tokenizer.fit_on_texts(train_lines)
    vocab_size = min(5000, len(tokenizer.word_index) + 1)

    # ---- STEP B3: pairs (same trick, extra id filter) ---------------------
    MAX_CONTEXT_WORDS = 7
    MAX_SEQUENCE_LEN = MAX_CONTEXT_WORDS + 1
    CONTEXT_LENGTHS = (1, 2, 4, MAX_CONTEXT_WORDS)

    def make_examples(lines):
        sequences = []
        for line in lines:
            ids = tokenizer.texts_to_sequences([line])[0]
            ids = [i for i in ids if 0 < i < vocab_size]   # drop pad/OOV
            if len(ids) < 2:
                continue
            for target_pos in range(1, len(ids)):
                lengths = {min(c, target_pos) for c in CONTEXT_LENGTHS}
                for L in lengths:
                    sequences.append(ids[target_pos - L: target_pos + 1])
        padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LEN,
                               padding="pre", truncating="pre")
        X = padded[:, :-1]
        y = padded[:, -1]
        return X.astype("int32"), y.astype("int32")

    X_train, y_train = make_examples(train_lines)
    X_val, y_val = make_examples(val_lines)

    # ---- STEP B4: train (smaller head, bigger batch) ----------------------
    model = Sequential([
        Input(shape=(MAX_CONTEXT_WORDS,)),
        Embedding(input_dim=vocab_size, output_dim=64, mask_zero=True),
        SimpleRNN(128, activation="tanh"),
        Dropout(0.25),
        Dense(vocab_size, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=[
                      tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                      tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy")])
    early_stop = EarlyStopping(monitor="val_loss", patience=5,
                               restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=30, batch_size=512, callbacks=[early_stop])

    # ---- STEP B5: predict (same as A6, seeds from Shakespeare) ------------
    index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}

    def predict_next_word(seed_text, top_k=5):
        cleaned_seed = clean_line(seed_text)
        token_list = tokenizer.texts_to_sequences([cleaned_seed])[0]
        token_list = [i for i in token_list if 0 < i < vocab_size]
        if not token_list:
            raise ValueError("Seed phrase contains no known vocabulary words.")
        token_list = token_list[-MAX_CONTEXT_WORDS:]
        token_list = pad_sequences([token_list], maxlen=MAX_CONTEXT_WORDS,
                                   padding="pre", truncating="pre")
        probs = model.predict(token_list, verbose=0)[0]
        ranked = np.argsort(probs)[::-1]
        predictions = []
        for idx in ranked:
            word = index_to_word.get(int(idx))
            if idx == 0 or word is None or word == "<OOV>":
                continue
            predictions.append((word, float(probs[idx])))
            if len(predictions) == top_k:
                break
        return predictions

    for seed in ["to be or not", "my lord", "the king", "i will"]:
        print(f"\nSeed: '{seed}'")
        for word, prob in predict_next_word(seed, top_k=5):
            print(f"  --> {word:15s} (prob: {prob:.4f})")


# =============================================================================
# SECTION 4T -- EXPERIMENT 4, TEACHING VERSION (one-word context)
# [SOURCE: Experiment_4_RNN.ipynb -- prof's classroom version]
# Differences vs exp4ab: ONE word in, ONE word out; one-hot targets;
# categorical_crossentropy; generate_seq() loop for word-by-word generation.
# Open this if the exam asks to GENERATE A SEQUENCE, not top-5.
# =============================================================================
def experiment_4_teaching():
    from numpy import array
    import numpy as np
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Dense, SimpleRNN, Embedding

    # ---- STEP 1: source text (long multiline string of simple sentences) --
    # [SOURCE: Experiment_4_RNN.ipynb cell 12 -- ~90 "students/nlp/model"
    # sentences; paste your copy from the notebook]
    data = """
    students learn natural language processing
    students study natural language processing
    recurrent neural network predicts the next word
    the model predicts the next word
    ...
    """

    # ---- STEP 2: integer encode + vocab -----------------------------------
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([data])
    encoded_data = tokenizer.texts_to_sequences([data])[0]
    vocab_size = len(tokenizer.word_index) + 1      # +1 for padding!

    # ---- STEP 3: word -> next-word pairs ----------------------------------
    sequences = list()
    for i in range(1, len(encoded_data)):
        sequences.append(encoded_data[i - 1: i + 1])   # [input, output]
    sequences = array(sequences)
    X, y = sequences[:, 0], sequences[:, 1]

    # ---- STEP 4: one-hot targets (THIS version's signature move) ----------
    y = to_categorical(y, num_classes=vocab_size)

    # ---- STEP 5: model + compile ------------------------------------------
    model = Sequential()
    model.add(Input(shape=(1,)))                 # ONE word as input
    model.add(Embedding(vocab_size, 10))         # 10-dim embedding
    model.add(SimpleRNN(50))                     # 50 hidden units
    model.add(Dense(vocab_size, activation='softmax'))
    # NOTE the loss difference vs exp4ab:
    model.compile(loss='categorical_crossentropy',  # one-hot targets
                  optimizer='adam', metrics=['accuracy'])

    # ---- STEP 6: fit --------------------------------------------------------
    history = model.fit(X, y, epochs=100, validation_split=0.20, verbose=1)

    # ---- STEP 7: generate a word SEQUENCE (loop) ---------------------------
    # [SOURCE: cell 27-29] this is the answer when the exam says "generate"
    def generate_seq(model, tokenizer, enter_text, n_pred):
        in_text, result = enter_text, enter_text
        for _ in range(n_pred):
            encoded = tokenizer.texts_to_sequences([in_text])[0]
            encoded = array(encoded).reshape(1, 1)   # model eats (1,1)
            prediction = model.predict(encoded, verbose=0)
            yhat = np.argmax(prediction, axis=1)[0]
            out_word = tokenizer.index_word.get(yhat, '')
            if out_word == '':
                break
            in_text = out_word                # predicted word feeds back
            result = result + ' ' + out_word
        return result

    print(generate_seq(model, tokenizer, 'natural', 3))
    print(generate_seq(model, tokenizer, 'recurrent', 3))


# =============================================================================
# SECTION 5A -- EXPERIMENT 5 (A): SPACY PIPELINE + TF-IDF FULL DISPLAY
# [SOURCE: exp5ab.ipynb Part A -- cells 3, 5, 7]
# Prof deck: NLPLAB_Exp5.pptx (14-sentence domain dataset: sports/finance/
# health/education -- THE professor's own data, likely reused in the exam)
# Sequence: load data -> spaCy tokens/POS/NER -> TF-IDF vocabulary+matrix
# =============================================================================
def experiment_5a():
    """spaCy preprocessing then sklearn TF-IDF, per the lab manual."""
    # ---- STEP 1: the 14-sentence dataset ----------------------------------
    # [SOURCE: exp5ab.ipynb cell 3; prof deck slide 2]
    sentences = [
        "The football team won the championship match",
        "The cricket team prepared for the final tournament",
        "The coach trained the players for the important game",
        "The stock market gained after strong economic news",
        "The company reported higher profit and revenue",
        "Investors bought technology shares during the market rally",
        "The doctor prescribed medicine for the patient",
        "The hospital opened a new health care center",
        "The nurse checked the patient's temperature and blood pressure",
        "The university introduced a machine learning course for students",
        "The Gujarat Technical University opened a laboratory in Gandhinagar",
        "Dr Mehta presented the project at the Indian Institute of Technology Delhi",
        "BlueGrid received funding of 5 million rupees in August 2026",
        "The Commonwealth Games project started in Ahmedabad"
    ]

    # ---- STEP 2+3: spaCy pipeline (tokens, stop, lemma, POS, NER) ---------
    # [SOURCE: exp5ab.ipynb cell 5; prof deck slides 3-4]
    # pip install spacy && python -m spacy download en_core_web_sm
    import spacy
    nlp = spacy.load("en_core_web_sm")

    for idx, text in enumerate(sentences, 1):
        doc = nlp(text)
        print(f"\n{'='*75}")
        print(f"Sentence {idx}: {text}")
        print(f"{'Token':<16} | {'is_stop_word':<13} | {'lemma':<16} | {'POS':<6}")
        print("-" * 60)
        for token in doc:
            print(f"{token.text:<16} | {str(token.is_stop):<13} | "
                  f"{token.lemma_:<16} | {token.pos_:<6}")
        entities = [ent.text for ent in doc.ents]
        print(f"\nNamed Entities:\n{tuple(entities)}")
        print("\nNamed Entities with Tags:")
        if doc.ents:
            for ent in doc.ents:
                print(f"{ent.text:<40} | {ent.label_:<6} | {spacy.explain(ent.label_)}")
        else:
            print("None")
    # Deck's sample output to reproduce: sentence 11 ->
    #   Gujarat Technical University = ORG (Companies, agencies, institutions)
    #   Gandhinagar = GPE (Countries, cities, states)

    # ---- STEP 4+5: TF-IDF vocabulary, size, full matrix -------------------
    # [SOURCE: exp5ab.ipynb cell 7; prof deck slides 5]
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()          # NOTE: no stop_words here (5A)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()

    print("=== Complete Feature Set (Vocabulary) ===")
    print(feature_names)
    print(f"\nVocabulary Size: {len(feature_names)}")

    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(),
                            columns=feature_names,
                            index=[f"Sentence {i+1}" for i in range(len(sentences))])
    print("\n=== Full Matrix (Preview - first 5 rows / 10 features) ===")
    print(df_tfidf.iloc[:5, :10].to_string())

    print("\n=== Non-Zero TF-IDF Weights Per Sentence ===")
    for i, sent in enumerate(sentences):
        row = df_tfidf.iloc[i]
        non_zero = row[row > 0]
        print(f"\nSentence {i+1}: '{sent}'")
        for word, score in non_zero.items():
            print(f"   {word:<15} : {score:.4f}")


# =============================================================================
# SECTION 5B -- EXPERIMENT 5 (B): TEXTBLOB + VADER SENTIMENT (PRE-TRAINED)
# [SOURCE: exp5ab.ipynb Part B -- cells 10, 12, 14, 16]
# Prof deck: NLPLAB_Exp5.pptx slides 6-8
# Sequence: labeled data -> TextBlob rule -> VADER rule -> metrics+heatmaps
# =============================================================================
def experiment_5b():
    """Two lexicon models scored with the Experiment-1 metric kit."""
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from textblob import TextBlob
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, confusion_matrix,
                                 classification_report)
    import matplotlib.pyplot as plt
    import seaborn as sns

    nltk.download('vader_lexicon', quiet=True)

    # ---- STEP 1: labeled dataset (12 sentences: 6 pos / 6 neg) ------------
    # [SOURCE: exp5ab.ipynb cell 10]
    dataset = [
        {"text": "The football team won the championship match brilliantly!", "label": "positive"},
        {"text": "The company reported higher profit and outstanding revenue growth.", "label": "positive"},
        {"text": "Investors bought technology shares during the booming market rally.", "label": "positive"},
        {"text": "The hospital opened an excellent and modern health care center.", "label": "positive"},
        {"text": "The university introduced an amazing machine learning course.", "label": "positive"},
        {"text": "The coach trained the players very well for the important game.", "label": "positive"},
        {"text": "The stock market crashed severely after terrible economic news.", "label": "negative"},
        {"text": "The patient suffered from severe pain and critical illness.", "label": "negative"},
        {"text": "The company suffered huge financial loss and massive layoffs.", "label": "negative"},
        {"text": "The project failed miserably due to poor management and lack of funds.", "label": "negative"},
        {"text": "The match was completely ruined by terrible weather and bad refereeing.", "label": "negative"},
        {"text": "The medicine had terrible side effects and harmed the patient.", "label": "negative"}
    ]
    texts = [item["text"] for item in dataset]
    y_true = [item["label"] for item in dataset]

    # ---- STEP 2: TextBlob rule-based analysis -----------------------------
    # [SOURCE: exp5ab.ipynb cell 12]
    # polarity in [-1, 1]; subjectivity in [0, 1]
    textblob_preds = []
    for text in texts:
        polarity = TextBlob(text).sentiment.polarity
        pred = "positive" if polarity >= 0 else "negative"   # threshold rule
        textblob_preds.append(pred)

    # ---- STEP 3: VADER lexicon-based analysis ------------------------------
    # [SOURCE: exp5ab.ipynb cell 14; deck slide 8]
    # VADER weighs intensity ("GREAT" vs "great"), punctuation ("good!"),
    # and negation ("not bad"). compound in [-1, 1].
    sia = SentimentIntensityAnalyzer()
    vader_preds = []
    for text in texts:
        scores = sia.polarity_scores(text)     # pos/neu/neg/compound
        pred = "positive" if scores['compound'] >= 0 else "negative"
        vader_preds.append(pred)

    # ---- STEP 4: evaluation (same metrics as Experiment 1) -----------------
    # [SOURCE: exp5ab.ipynb cell 16]
    labels = ["positive", "negative"]

    def evaluate_model(name, y_true, y_pred):
        print(f"\n==================== {name} Evaluation ====================")
        print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred, pos_label='positive'):.4f}")
        print(f"Recall   : {recall_score(y_true, y_pred, pos_label='positive'):.4f}")
        print(f"F1-Score : {f1_score(y_true, y_pred, pos_label='positive'):.4f}")
        print(classification_report(y_true, y_pred, labels=labels))
        return confusion_matrix(y_true, y_pred, labels=labels)

    cm_textblob = evaluate_model("TextBlob", y_true, textblob_preds)
    cm_vader = evaluate_model("NLTK VADER", y_true, vader_preds)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_textblob, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title("TextBlob Confusion Matrix")
    sns.heatmap(cm_vader, annot=True, fmt='d', cmap='Greens',
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title("NLTK VADER Confusion Matrix")
    plt.tight_layout(); plt.show()


# =============================================================================
# SECTION 6 -- EXPERIMENT 6: N-GRAM LANGUAGE MODELS (BIGRAM vs TRIGRAM)
# [SOURCE: exp6.ipynb -- MLE + Laplace + candidate table + 2 bar charts]
# Prof deck: NLPLAB_Exp6.pptx (train -> test "the cat" -> perplexity)
# Sequence: tokenize -> train bi/tri MLE -> test contexts -> Laplace ->
#           perplexity -> 2 graphs
# =============================================================================
def experiment_6():
    """The better exam base: has Laplace + perplexity table + graphs."""
    import nltk
    from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
    from nltk.lm import MLE, Laplace
    from nltk.tokenize import word_tokenize
    from nltk.util import ngrams
    import pandas as pd
    import matplotlib.pyplot as plt

    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    # ---- STEP 1: tokenize the corpus ---------------------------------------
    # [SOURCE: exp6.ipynb cell 2; prof deck slide 4]
    # KNOB: exam corpus goes here; keep the list-of-lists shape.
    train_text = ["the cat sat on the mat and then cat ran away quickly. "
                  "the dog barked at the cat."]
    tokenized_train = [word_tokenize(sent.lower()) for sent in train_text]

    # ---- STEP 2: train bigram + trigram MLE models -------------------------
    # [SOURCE: exp6.ipynb cell 4]
    # padded_everygram_pipeline adds <s>/</s> markers and yields ALL n-grams
    # up to n. WARNING: generators are single-use -- fresh pair per model!
    n_bigram = 2
    train_data_2, padded_vocab_2 = padded_everygram_pipeline(n_bigram, tokenized_train)
    model_bigram = MLE(n_bigram)
    model_bigram.fit(train_data_2, padded_vocab_2)

    n_trigram = 3
    train_data_3, padded_vocab_3 = padded_everygram_pipeline(n_trigram, tokenized_train)
    model_trigram = MLE(n_trigram)
    model_trigram.fit(train_data_3, padded_vocab_3)

    # frequency tables (explicit, per the deck)
    print("Bigram context counts ('cat'):",
          dict(model_bigram.context_counts(('cat',))))
    print("Trigram context counts ('the','cat'):",
          dict(model_trigram.context_counts(('the', 'cat'))))

    # ---- STEP 3: test both models on a prompt -------------------------------
    # [SOURCE: exp6.ipynb cell 6; prof deck slide 5]
    # Bigram context = last 1 token; trigram context = last 2 tokens.
    prompt = "the cat"
    tokens = word_tokenize(prompt.lower())
    ctx_bi = (tokens[-1],)           # ('cat',)
    ctx_tri = tuple(tokens[-2:])     # ('the', 'cat')

    candidates = ['sat', 'ran', '.', 'on', 'away']
    table = []
    for w in candidates:
        table.append({'Candidate': w,
                      "P(w|'cat') [Bigram]": model_bigram.score(w, ctx_bi),
                      "P(w|'the cat') [Trigram]": model_trigram.score(w, ctx_tri)})
    df_res = pd.DataFrame(table)
    print(df_res.to_string(index=False))

    # random generation (nltk samples from the model's distribution)
    print("\nGenerated Bigram sentence:",
          ' '.join(model_bigram.generate(6, text_seed=['cat'])))
    print("Generated Trigram sentence:",
          ' '.join(model_trigram.generate(6, text_seed=['the', 'cat'])))

    # ---- STEP 4: Laplace models + perplexity --------------------------------
    # [SOURCE: exp6.ipynb cell 8]
    # MLE gives inf perplexity on unseen n-grams; Laplace (add-1) fixes it.
    td2, pv2 = padded_everygram_pipeline(2, tokenized_train)
    lap_bigram = Laplace(2); lap_bigram.fit(td2, pv2)
    td3, pv3 = padded_everygram_pipeline(3, tokenized_train)
    lap_trigram = Laplace(3); lap_trigram.fit(td3, pv3)

    test_phrases = {
        'the cat sat': ['the', 'cat', 'sat'],       # seen in both models
        'the cat ran': ['the', 'cat', 'ran']        # UNSEEN trigram -> inf
    }
    ppl_data = []
    for phrase, seq in test_phrases.items():
        bg = list(ngrams(seq, 2))
        tg = list(ngrams(seq, 3))
        ppl_data.append({
            'Sequence': phrase,
            'Bigram MLE PPL': round(model_bigram.perplexity(bg), 4),
            'Trigram MLE PPL': (round(model_trigram.perplexity(tg), 4)
                                if model_trigram.perplexity(tg) != float('inf')
                                else 'inf'),
            'Bigram Laplace PPL': round(lap_bigram.perplexity(bg), 4),
            'Trigram Laplace PPL': round(lap_trigram.perplexity(tg), 4)
        })
    df_ppl = pd.DataFrame(ppl_data)
    print(df_ppl.to_string(index=False))

    # ---- STEP 5: the two graphs ----------------------------------------------
    # [SOURCE: exp6.ipynb cell 10]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width, idx = 0.35, range(len(candidates))
    ax.bar([i - width/2 for i in idx], df_res["P(w|'cat') [Bigram]"],
           width=width, label="Bigram (N=2)", color='#1f77b4')
    ax.bar([i + width/2 for i in idx], df_res["P(w|'the cat') [Trigram]"],
           width=width, label="Trigram (N=3)", color='#ff7f0e')
    ax.set_title("Next Token Probability under Context Constraints",
                 fontweight='bold')
    ax.set_xlabel("Candidate Words"); ax.set_ylabel("Probability")
    ax.set_xticks(list(idx)); ax.set_xticklabels(candidates)
    ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 4))
    x_labels = list(test_phrases.keys())
    idx_ppl = range(len(x_labels))
    ax.bar([i - width/2 for i in idx_ppl], df_ppl['Bigram Laplace PPL'],
           width=width, label='Bigram Laplace', color='#2ca02c')
    ax.bar([i + width/2 for i in idx_ppl], df_ppl['Trigram Laplace PPL'],
           width=width, label='Trigram Laplace', color='#d62728')
    ax.set_title("Perplexity Evaluation (Laplace Smoothed)", fontweight='bold')
    ax.set_xlabel("Test Sequence"); ax.set_ylabel("Perplexity")
    ax.set_xticks(list(idx_ppl)); ax.set_xticklabels(x_labels)
    ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()


# =============================================================================
# SECTION 6V2 -- EXPERIMENT 6, VARIANT 2 (vocabulary-wide P + avg perplexity)
# [SOURCE: exp6v2.ipynb]
# Differences vs exp6: probability table across the WHOLE vocabulary, and
# perplexity on full test SENTENCES (pad_both_ends) with averages.
# Open this if the exam asks for a full P table or average perplexity.
# =============================================================================
def experiment_6_v2():
    import nltk
    from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
    from nltk.lm import MLE
    from nltk.tokenize import word_tokenize
    from nltk.util import ngrams
    import pandas as pd
    import numpy as np

    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    # ---- STEP 1: same corpus, same training (see SECTION 6 STEP 2) --------
    # [SOURCE: exp6v2.ipynb cell 3]
    train_text = ["the cat sat on the mat and then cat ran away quickly. "
                  "the dog barked at the cat."]
    tokenized_train = [word_tokenize(sent.lower()) for sent in train_text]

    train_data_bi, vocab_bi = padded_everygram_pipeline(2, tokenized_train)
    model_bi = MLE(2); model_bi.fit(train_data_bi, vocab_bi)

    train_data_tri, vocab_tri = padded_everygram_pipeline(3, tokenized_train)
    model_tri = MLE(3); model_tri.fit(train_data_tri, vocab_tri)

    # ---- STEP 2: vocab-wide probability table -------------------------------
    # [SOURCE: exp6v2.ipynb cell 5]
    prompt = "the cat"
    prompt_tokens = word_tokenize(prompt.lower())
    context_bi = [prompt_tokens[-1]]      # ['cat']
    context_tri = prompt_tokens[-2:]      # ['the', 'cat']

    results = []
    for word in sorted(model_bi.vocab):
        p_bi = model_bi.score(word, context_bi)
        p_tri = model_tri.score(word, context_tri)
        if p_bi > 0 or p_tri > 0:
            results.append({"Next Token": word,
                            "Bigram P(w | cat)": p_bi,
                            "Trigram P(w | the, cat)": p_tri})
    df_probs = pd.DataFrame(results).sort_values(
        by="Trigram P(w | the, cat)", ascending=False).reset_index(drop=True)
    print("=== Next-Token Prediction Probabilities ===")
    print(df_probs.to_string(index=False))

    # ---- STEP 3: perplexity on full sentences + average ---------------------
    # [SOURCE: exp6v2.ipynb cell 9]
    # pad_both_ends wraps each test sentence with <s>/</s> so boundary
    # n-grams are counted too -- fairer evaluation.
    test_sentences = ["the cat sat on the mat .",
                      "the dog barked at the cat ."]
    tokenized_test = [word_tokenize(s.lower()) for s in test_sentences]

    ppl_bi, ppl_tri = [], []
    for sent in tokenized_test:
        test_bi = list(ngrams(pad_both_ends(sent, n=2), 2))
        test_tri = list(ngrams(pad_both_ends(sent, n=3), 3))
        ppl_bi.append(model_bi.perplexity(test_bi))
        ppl_tri.append(model_tri.perplexity(test_tri))

    print("Average Bigram Perplexity :", np.mean(ppl_bi))
    print("Average Trigram Perplexity:", np.mean(ppl_tri))
    # Numbers your notebook concluded: bigram spreads P=1/3 over
    # sat/ran/. after 'cat'; trigram sharpens to P=0.5 sat, 0.5 '.'
    #   -> richer context = less uncertainty.
    # VERIFIED GOTCHA (ran 2026-11-16): on the test sentence
    # "the cat sat on the mat ." the bigram ('mat', '.') NEVER occurs in
    # training (corpus has "the mat and ..."), so P=0 -> MLE average PPL
    # is inf for BOTH models. That is not a bug -- it is the zero-frequency
    # problem. Write exactly that, or evaluate with Laplace models for
    # finite numbers. (One unseen n-gram poisons the whole average.)


# =============================================================================
# SECTION 7 -- EXPERIMENT 7: BERT TEXT CLASSIFICATION (DISTILBERT + IMDB)
# [SOURCE: Experiment_7_Text_Classification_BERT.ipynb -- 16 cells, complete]
# Sequence: load dataset -> subsample -> tokenize -> remove text col ->
#           model head -> TrainingArguments -> Trainer + metric -> predict
# =============================================================================
def experiment_7():
    """HuggingFace fine-tuning skeleton, exactly as in your notebook."""
    # pip install "transformers==4.56.1" "tokenizers==0.22.0" datasets evaluate
    # (version pin from the notebook's first cell -- keep it)
    from datasets import load_dataset
    from transformers import (AutoTokenizer,
                              AutoModelForSequenceClassification,
                              TrainingArguments, Trainer)
    import evaluate
    import numpy as np

    # ---- STEP 1: model + tokenizer (same name for both) --------------------
    # [SOURCE: cell 1]
    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ---- STEP 2: dataset + subsample ----------------------------------------
    # [SOURCE: cell 2]  KNOB: ag_news -> num_labels=4;
    #                   ('tweet_eval', 'sentiment') -> 3 classes
    dataset = load_dataset('stanfordnlp/imdb')
    train_ds = dataset['train'].shuffle(seed=42).select(range(2000))
    test_ds = dataset['test'].shuffle(seed=42).select(range(500))

    # ---- STEP 3: tokenize in batches ----------------------------------------
    # [SOURCE: cells 4-5]
    def tokenize_batch(batch):
        return tokenizer(batch['text'], padding=True, truncation=True,
                         max_length=128)

    train_tok = train_ds.map(tokenize_batch, batched=True, batch_size=None)
    test_tok = test_ds.map(tokenize_batch, batched=True, batch_size=None)

    # ---- STEP 4: drop text col, torch format --------------------------------
    # [SOURCE: cell 6]  Trainer only wants input_ids/attention_mask/label
    train_tok = train_tok.remove_columns(['text'])
    test_tok = test_tok.remove_columns(['text'])
    train_tok.set_format('torch')
    test_tok.set_format('torch')

    # ---- STEP 5: classification head ------------------------------------------
    # [SOURCE: cell 8]
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2)

    # ---- STEP 6: training arguments -------------------------------------------
    # [SOURCE: cell 9]  lr 2e-5 and 3 epochs are the standard BERT numbers.
    args = TrainingArguments(
        output_dir='./bert_imdb_results',
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        report_to='none')

    # ---- STEP 7: metric + Trainer + train --------------------------------------
    # [SOURCE: cells 11-14]
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)   # logits -> class
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(model=model, args=args,
                      train_dataset=train_tok, eval_dataset=test_tok,
                      compute_metrics=compute_metrics)
    trainer.train()
    print(trainer.evaluate())

    # ---- STEP 8: predict on a fresh sentence ------------------------------------
    # [SOURCE: cell 15]  1 = Positive, 0 = Negative for IMDB
    sample_text = 'The film is meaningful, emotional beautifully performed.'
    inputs = tokenizer(sample_text, padding=True, truncation=True,
                       return_tensors='pt')
    outputs = model(**inputs)
    predicted_label = outputs.logits.argmax(dim=-1).item()
    print('Predicted Sentiment: ',
          'Positive' if predicted_label == 1 else 'Negative')


# =============================================================================
# SECTION 8 -- COMBINATION PLAYBOOK (exam = two experiments glued together)
# =============================================================================
# A combined question is two sections above glued at ONE data object.
# "GLUE" below is the exact line that connects them.
#
# COMBO              | WHAT IT LOOKS LIKE                  | GLUE LINE
# --------------------+--------------------------------------+------------------------
# EXP3 -> EXP1/5A    | preprocess then TF-IDF+classifier    | df['clean'] = df['review'].apply(clean_and_lemmatize)
#                    |   (visible effect: vocabulary        |   then vectorize df['clean'] instead of df['review'];
#                    |    shrinks -- print both sizes)      |   clean = tokenize->stem/lemmatize->' '.join
# EXP3 -> EXP5A      | NLTK stemming vs spaCy lemmas        |   loop nlp(text); compare token.lemma_ column
#                    |                                      |   with Porter stems in one table
# EXP5A -> EXP5B     | spaCy preprocess then sentiment      |   sia.polarity_scores(' '.join(t.lemma_ for t in doc
#                    |   (lemmas/stopword removal change    |        if not t.is_stop))  vs raw-text scores
#                    |    polarity -- good conclusion)      |
# EXP1 vs EXP7       | classic ML vs BERT, same task        |   keep same split; report both accuracy tables;
#                    |                                      |   in exam shrink BERT (range(500), 1 epoch)
# EXP6 vs EXP4       | n-gram LM vs RNN, same corpus        |   same train/val split; trigram top continuations
#                    |                                      |   next to predict_top_k_words() for same seeds
# EXP2 -> anything   | file handling as data loading        |   raw = open('corpus.txt').read() or
#                    |                                      |   PdfReader('x.pdf').pages[i].extract_text()
# EXP5B -> EXP1      | score sentiment rules like a model   |   y_pred lists go straight into accuracy_score /
#                    |                                      |   confusion_matrix (already done in 5B STEP 4)
# EXP3 -> EXP4       | proper preprocessing before RNN      |   exp4's clean_text IS exp3-lite; swap in
#                    |                                      |   stem/lemma for a deeper variant
# -----------------------------------------------------------------------------
# CROSS-EXPERIMENT THREADS (one sentence each, for any conclusion):
#   * TF-IDF appears twice: EXP1 (stop_words='english') vs EXP5A (raw).
#   * The metric kit is identical in EXP1 and EXP5B.
#   * Next-word prediction appears 3x with rising sophistication:
#       counting (EXP6) -> recurrence (EXP4) -> attention (EXP7 context).
#   * Preprocessing deepens: regex (EXP2/4) -> NLTK (EXP3) -> spaCy (EXP5).

# =============================================================================
# SECTION 9 -- SWAP RECIPES (what to edit when the exam changes one thing)
# =============================================================================
# MUTATION                          | BASE            | EDIT
# ----------------------------------+-----------------+---------------------------
# Different sentiment dataset (CSV/ | EXP1 STEP 1     | df = pd.read_csv('exam.csv');
#   dict/HF dataset)                |                 |   map string labels to 0/1
#                                   |                 |   (df['label'].map({...}))
# Add/swap classifier               | EXP1 STEP 3     | one entry in the classifiers
#                                   |                 |   dict (NB, RandomForest, KNN)
# N-gram with N=4 / new corpus /    | EXP6 STEP 1-2   | change n and train_text; fresh
#   Laplace required                |                 |   pipeline generators per model
# RNN on different corpus           | EXP4 B1-B2      | only the loader changes:
#   (Gutenberg/headlines/notices/   |                 |   open().read().splitlines() ->
#   IMDB text)                      |                 |   clean lines -> >=2 words ->
#                                   |                 |   train_test_split(0.15, seed=42)
# LSTM / GRU instead of SimpleRNN   | EXP4 A4         | swap the layer; shapes unchanged;
#                                   |                 |   report new param count
# Different seeds / top-k           | EXP4 A6         | edit seed list / top_k argument
# 3-class VADER (add neutral)       | EXP5B STEP 3    | compound >= 0.05 pos, <= -0.05 neg,
#                                   |                 |   else neutral; labels list + macro avg
# BERT different dataset/metric     | EXP7 STEP 2/7   | load_dataset swap + num_labels;
#                                   |                 |   evaluate.load('f1') with average='macro'
# Rotate / watermark PDF            | SECTION 2.5     | ready-made functions there
# =============================================================================

# =============================================================================
# SECTION 10 -- TRAPS AND FIXES (read in the last 10 minutes)
# =============================================================================
# TRAP                                     | FIX
# -----------------------------------------+--------------------------------
# nltk ResourceLookupError                 | SECTION 0 setup_nltk() first
# nltk.lm generator exhausted after fit    | fresh padded_everygram_pipeline
#                                          |   per model (exp6v2 comments it)
# MLE perplexity = inf                     | that IS the MLE limitation --
#                                          |   say it, or switch to Laplace(n)
# wrong loss for target shape              | int targets -> sparse_categorical_
#                                          |   crossentropy; one-hot (to_categorical)
#                                          |   -> categorical_crossentropy
# tokenizer fit on val/test data           | fit on TRAIN only (EXP4 B2)
# vocab_size off by one                    | len(word_index) + 1  (0 = padding)
# <OOV>/padding returned as top word       | skip idx 0 and '<OOV>' in ranking
#                                          |   loop (both predict fns do)
# metrics on string labels                 | pos_label='positive', or map to 0/1;
#                                          |   zero_division=0 for tiny data
# spaCy model missing                      | python -m spacy download en_core_web_sm
# transformers import fails                | pip install transformers==4.56.1
#                                          |   tokenizers==0.22.0 (notebook pin)
# Shakespeare too slow                     | cut NUM_LINES / vocab cap / epochs,
#                                          |   raise batch (deck's own advice)
# every seed predicts same word            | check mask_zero=True, variable
#                                          |   contexts, more data/epochs
# one-char-per-line outputs in notebooks   | Colab stdout artifact; harmless in
#                                          |   printout, restart kernel if rerun
# =============================================================================

# =============================================================================
# SECTION 11 -- API ONE-LINERS (the 25 lines you will retype most)
# =============================================================================
# nltk
#   nltk.word_tokenize(s) / nltk.sent_tokenize(s)
#   PorterStemmer().stem(w) / LancasterStemmer() / SnowballStemmer('english')
#   WordNetLemmatizer().lemmatize(w, pos='v')
#   from nltk.corpus import stopwords; stopwords.words('english')
#   padded_everygram_pipeline(2, tokenized) -> MLE(2).fit(train_data, vocab)
#   model.score(w, ('ctx',)) / model.perplexity(ngrams(seq, 2))
#   model.generate(6, text_seed=['the'])
# pypdf
#   PdfReader(f).pages[i].extract_text() / PdfWriter().append(f)
#   writer.add_page(reader.pages[i]) / page.rotate(90)
# sklearn
#   TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=500)
#   .get_feature_names_out() / X.shape / X.toarray()
#   train_test_split(X, y, test_size=0.3, random_state=42)
#   confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# textblob / vader
#   TextBlob(s).sentiment  -> (polarity, subjectivity)
#   SentimentIntensityAnalyzer().polarity_scores(s) -> compound
# spacy
#   nlp = spacy.load('en_core_web_sm'); doc = nlp(s)
#   token.text / token.is_stop / token.lemma_ / token.pos_ / token.dep_
#   [(ent.text, ent.label_, spacy.explain(ent.label_)) for ent in doc.ents]
# keras
#   Tokenizer(num_words=5000, oov_token='<OOV>'); fit_on_texts; texts_to_sequences
#   pad_sequences(seqs, maxlen=L, padding='pre', truncating='pre')
#   Embedding(vocab, 64, mask_zero=True) -> SimpleRNN(128) -> Dense(vocab, softmax)
#   loss='sparse_categorical_crossentropy', metrics=[SparseTopKCategoricalAccuracy(k=5)]
#   EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# huggingface
#   load_dataset('stanfordnlp/imdb'); AutoTokenizer.from_pretrained(name)
#   AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)
#   Trainer(model, args, train_dataset, eval_dataset, compute_metrics)
#   model(**tokenizer(s, return_tensors='pt')).logits.argmax(dim=-1).item()
# =============================================================================


# =============================================================================
# CLI RUNNER -- python nlp_lab_exam_master.py [--exp1 .. --exp7] [--all]
# =============================================================================
EXPERIMENTS = {
    '--exp1': ("EXP1  ML classifiers + TF-IDF", experiment_1),
    '--exp2': ("EXP2  text + PDF file handling", experiment_2),
    '--exp2b': ("EXP2.5 rotate + watermark", experiment_2b),
    '--exp3': ("EXP3  tokenize / stem / lemmatize", experiment_3),
    '--exp4': ("EXP4  RNN next-word (A+B)", experiment_4),
    '--exp4t': ("EXP4T teaching version + generate_seq", experiment_4_teaching),
    '--exp5a': ("EXP5A spaCy + TF-IDF", experiment_5a),
    '--exp5b': ("EXP5B TextBlob + VADER", experiment_5b),
    '--exp6': ("EXP6  n-gram LM (MLE+Laplace)", experiment_6),
    '--exp6v2': ("EXP6V2 vocab table + avg PPL", experiment_6_v2),
    '--exp7': ("EXP7  DistilBERT fine-tune", experiment_7),
    '--setup': ("shared NLTK setup only", setup_nltk),
}


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)                     # the index + master map above
        print("Runnable sections:")
        for flag, (title, _) in EXPERIMENTS.items():
            print(f"   python nlp_lab_exam_master.py {flag:8s}  # {title}")
        print("\n(Heavy libs are imported inside each function, so this "
              "index works even without tensorflow/spacy installed.)")
        return 0
    for flag in args:
        if flag not in EXPERIMENTS:
            print(f"unknown flag {flag!r} -- run with no args for the index")
            return 1
        title, fn = EXPERIMENTS[flag]
        banner = f" RUNNING {flag} : {title} "
        print("\n" + "=" * 79)
        print(banner.center(79, "="))
        print("=" * 79 + "\n")
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
