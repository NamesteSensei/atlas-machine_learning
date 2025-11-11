# 🧠 Word Embeddings in Natural Language Processing (NLP)

This project introduces **word embeddings**, one of the most important
concepts in Natural Language Processing (NLP). Word embeddings are how
computers convert words into numerical form so they can analyze and
understand human language.

You will explore several embedding techniques — from basic to advanced —
to understand how words can be represented as vectors in a meaningful way.

---

## 📚 Learning Objectives

By the end of this project, you should be able to explain:

- What is **Natural Language Processing (NLP)**?
- What is a **word embedding**?
- What is **Bag of Words (BoW)**?
- What is **TF-IDF** and how it differs from BoW?
- What are **CBOW** and **Skip-Gram**?
- What is an **n-gram**?
- What is **negative sampling**?
- What are **Word2Vec**, **GloVe**, **FastText**, and **ELMo**?

---

## 🧰 Technologies Used

- Python 3.9 (Ubuntu 20.04 LTS)
- NumPy 1.25.2
- TensorFlow 2.15.0
- Keras 2.15.0
- Gensim 4.3.3
- pycodestyle 2.11.1

All scripts are executable and follow **PEP8 (pycodestyle)** standards with
a maximum line length of **78 characters**.

---

## 📦 Installation

Install the required dependencies:

```bash
pip install --user numpy==1.25.2
pip install --user tensorflow==2.15
pip install --user gensim==4.3.3
pip install --user pycodestyle==2.11.1

Project Stucture

atlas-machine_learning/
└── supervised_learning/
    └── word_embeddings/
        ├── 0-bag_of_words.py
        ├── 0-main.py
        ├── 1-tf_idf.py
        ├── 2-word2vec.py
        ├── 3-gensim_to_keras.py
        ├── 4-fasttext.py
        ├── 5-elmo
        ├── README.md

