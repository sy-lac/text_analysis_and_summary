import streamlit as st
from collections import Counter
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing import text,sequence
from tensorflow.keras.preprocessing.text import Tokenizer

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from textblob import TextBlob, Blobber
from textblob_fr import PatternTagger, PatternAnalyzer

import spacy.cli
spacy.cli.download("fr_core_news_md")

import torch
import sentencepiece as spm
from transformers import CamembertTokenizer, CamembertModel
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity


# nombre de mots et de mots uniques
def number_words(text):
  word = text.split()
  return 'Nombre de mots : {} - Nombre de mots uniques : {}'.format(len(word), len(Counter(word)))
    
# polarité
def polarity(text):
  tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
  if tb(text).sentiment[0] < 0:
      return f'Polarité de {tb(text).sentiment[0]} : ce texte est plus négatif que positif'
  elif tb(text).sentiment[0] > 0:
      return f'Polarité de {tb(text).sentiment[0]} : ce texte est plus positif que négatif'
  else :
      return f'Polarité de {tb(text).sentiment[0]} : ce texte est neutre, pas plus négatif que positif'


# subjectivité
def subjectivity(text):
  tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
  if tb(text).sentiment[1] < 0.5:
    return f'Subjectivité de {tb(text).sentiment[1]} : ce texte est plus subjectif que factuel'
  elif tb(text).sentiment[1] > 0.5:
    return f'Subjectivité de {tb(text).sentiment[1]} : ce texte est plus subjectif que factuel'
  else :
    return f'Subjectivité de {tb(text).sentiment[1]} : ce texte est neutre, pas plus subjectif que factuel'


# mots clés
def keywords(text):
  nlp = spacy.load("fr_core_news_md")
  text2 = nlp(text)
  text_keywords = [token.text for token in text2 if token.pos_== 'NOUN' or token.pos_== 'PROPN' or token.pos_== 'VERB']
  counter_words = Counter(text_keywords)
  most_freq_words = [word for word in counter_words.most_common(10)]
  most_freq_words_p = []
  for i in range(len(most_freq_words)):
    mfwp = most_freq_words[i][0]
    most_freq_words_p.append(mfwp)
  return most_freq_words_p


# summary1
def summary_1(text):
  model = CamembertModel.from_pretrained('camembert-base')
  tokenizer = CamembertTokenizer.from_pretrained('camembert-base')

  ## preprocessing
  sentences = sent_tokenize(text)
  tokenized_sentences = [tokenizer.encode(sent, add_special_tokens=True) for sent in sentences]

  ## padding, encoding
  max_len = 0
  for i in tokenized_sentences:
    if len(i) > max_len:
      max_len = len(i)

  padded_sentences = []
  for i in tokenized_sentences:
    while len(i) < max_len:
      i.append(0)
    padded_sentences.append(i)

  input_ids = torch.tensor(padded_sentences)

  ## embedding
  with torch.no_grad():
    last_hidden_states = model(input_ids)[0]

  sentence_embeddings = []
  for i in range(len(sentences)):
    sentence_embeddings.append(torch.mean(last_hidden_states[i], dim=0).numpy())

  ## summarizing
  similarity_matrix = cosine_similarity(sentence_embeddings)

  num_sentences = 3
  summary_sentences = []
  for i in range(num_sentences):
    sentence_scores = list(enumerate(similarity_matrix[i]))
    sentence_scores = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
    summary_sentences.append(sentences[sentence_scores[1][0]])

  summary = ' '.join(summary_sentences)
  return summary


# summary2
def summary_2(text):
  nlp = spacy.load("fr_core_news_md")
  text2 = nlp(text)
  text_keywords = [token.text for token in text2 if token.pos_== 'NOUN' or token.pos_== 'PROPN']

  counter_words = Counter(text_keywords)
  most_freq_words = [word for word in counter_words.most_common(3)]

  most_freq_words_p = []
  for i in range(len(most_freq_words)):
      mfwp = most_freq_words[i][0]
      most_freq_words_p.append(mfwp)

  sentences = sent_tokenize(text)
  summary = []
  for sent in sentences:
    for word in sent.split():
      if word in most_freq_words_p and sent not in summary:
        summary.append(sent)
      
  return summary


def analyze_text(text):
    nb_mots = number_words(text)
    polarite = polarity(text)
    subjectivite = subjectivity(text)
    mots_cles = keywords(text)
    resume1 = summary_1(text)
    resume2 = summary_2(text)

    return nb_mots, polarite, subjectivite, mots_cles, resume1, resume2



background_style = """
    <style>
        body {
            background-image: url('desmotsdanstouslessens.jpg');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
    </style>
"""
st.markdown(background_style, unsafe_allow_html=True)
st.image('desmotsdanstouslessens.jpg', use_column_width=True)

st.title('Text Analysis and Summary')

text = st.text_area('Enter text here:')

if st.button('Analyze'):
  if text:
    nb_mots, polarite, subjectivite, mots_cles, resume1, resume2 = analyze_text(text)

    st.write(nb_mots)
    st.write(polarite)
    st.write(subjectivite)
    st.write('Mots clés :', ', '.join(mots_cles))
    st.write(f'Résumé 1 : {resume1}')
    st.write('Résumé 2 :', ''.join(resume2))

if st.button('Clear'):
    text = ""

