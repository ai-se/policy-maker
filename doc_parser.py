from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from bs4 import BeautifulSoup
from collections import OrderedDict
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from nltk.stem.porter import PorterStemmer
from utils.lib import O
from sklearn.model_selection import train_test_split
from utils.lda_extended import log_perplexity as perplexity
import json
import requests
import re
import pickle
import lda
import numpy as np
import matplotlib.pyplot as plt
import logging
import pandas as pd

logging.getLogger('lda').setLevel(logging.ERROR)

# Data Collection
IOT_URL = "https://www.federalregister.gov/api/v1/documents.json?" \
          "conditions%5Bagencies%5D%5B%5D=commerce-department&" \
          "conditions%5Bagencies%5D%5B%5D=defense-department&" \
          "conditions%5Bagencies%5D%5B%5D=health-and-human-services-department" \
          "&conditions%5Bagencies%5D%5B%5D=national-aeronautics-and-space-administration" \
          "&conditions%5Bagencies%5D%5B%5D=transportation-department&conditions%5Bterm%5D=%22internet+of+things" \
          "%22&page=1"
SUS_DEV_URL = "https://www.federalregister.gov/api/v1/documents.json?conditions%5Bagencies%" \
              "5D%5B%5D=commerce-department&conditions%5Bagencies%5D%5B%5D=defense-department" \
              "&conditions%5Bagencies%5D%5B%5D=national-aeronautics-and-space-administration" \
              "&conditions%5Bagencies%5D%5B%5D=health-and-human-services-department" \
              "&conditions%5Bagencies%5D%5B%5D=transportation-department" \
              "&conditions%5Bterm%5D=%22sustainable+development%22&page=1"
IOT_PATH = "files/data/iot.pkl"
SUS_DEV_PATH = "files/data/sus_dev.pkl"

# LDADE
TOKEN_PATTERN = re.compile(r"(?u)\b[a-zA-Z]{2}[a-zA-Z]+\b")
ITERATIONS = 100
ALPHA = None
BETA = None
STOP_WORDS = ENGLISH_STOP_WORDS.union(['software', 'engineering'])
N_TOPICS = 10
RANDOM_STATE = 1
AGENCY_MAP = {
  'Transportation Department': 'DOT',
  'Federal Transit Administration': 'FTA',
  'Commerce Department': 'DOC',
  'International Trade Administration': 'ITA',
  'Economic Development Administration': 'EDA',
  'National Oceanic and Atmospheric Administration': 'NOAA',
  'Federal Highway Administration': 'FHWA',
  'Interior Department': 'DOI',
  'Fish and Wildlife Service': 'FWS',
  'Defense Department': 'DOD',
  'Navy Department': 'USN',
  'Health and Human Services Department': 'HHS',
  'Food and Drug Administration': 'FDA',
  'Environmental Protection Agency': 'EPA',
  'National Highway Traffic Safety Administration': 'NHTSA',
  'Children and Families Administration': 'ACF',
  'National Institute of Standards and Technology': 'NIST',
  'Housing and Urban Development Department': 'HUD',
  'Federal Railroad Administration': 'FRA',
  'Maritime Administration': 'MARAD',
  'Army Department': 'USA',
  'National Aeronautics and Space Administration': 'NASA',
  'National Telecommunications and Information Administration': 'NTIA',
  'National Technical Information Service': 'NTIS'
}
PERMITTED_AGENCIES = ['DOC', 'DOT', 'NASA', 'HHS', 'DOD']

class Document(O):
  """
  Attributes:
    id: Document ID(Auto Increment)
    abstract: Abstract of the document
    html_url: URL of the document in html
    pdf_url: URL of the document in pdf
    agencies: List of agencies associated with this document
    publication date: Date when this document was published
    content: Page content from the html url
    vector: Term Frequency from count vectorizer
    topic_count: Count of terms from each topic in the document
    topic_score: Score of each topic in the document
  """
  id = 0

  def __init__(self, **kwargs):
    O.__init__(self, **kwargs)
    self.id = Document.id
    Document.id += 1

  @staticmethod
  def from_json(json_obj):
    return Document(title=json_obj['title'],
                    abstract=json_obj['abstract'],
                    html_url=json_obj['html_url'],
                    pdf_url=json_obj['pdf_url'],
                    agencies=[agency['name'] for agency in json_obj['agencies'] if 'name' in agency],
                    publication_date=json_obj['publication_date'])

  @staticmethod
  def dump(documents, file_name):
    with open(file_name, 'wb') as f:
      pickle.dump(documents, f)

  @staticmethod
  def load(file_name):
    with open(file_name, 'rb') as f:
      return pickle.load(f)

  def get_raw(self):
    if self.content is not None:
      return self.content
    elif self.abstract is not None:
      print(2)
      return self.abstract
    else:
      print(3)
      return self.title


def get_json(src, from_internet=False):
  if from_internet:
    return json.loads(requests.get(src).text)
  else:
    with open(src) as f:
      return json.loads(f.read())


def parse(file_name, from_internet=False):
  json_obj = get_json(file_name, from_internet)
  documents = []
  for result in json_obj['results']:
    document = Document.from_json(result)
    print(document.id)
    document.content = process_html_url(document.html_url)
    documents.append(document)
  if 'next_page_url' in json_obj:
    documents += parse(json_obj['next_page_url'], from_internet=True)
  return documents


def process_html_url(url):
  response = requests.get(url)
  if response.status_code != 200:
    return None
  soup = BeautifulSoup(response.content, "lxml")
  wrapper = soup.find("div", {"id": "fulltext_content_area"})
  html_text = wrapper.get_text(separator=u" ", strip=True).encode("ascii", "ignore")
  html_text = re.sub(r'https?://.*[\r\n]*', '', html_text)
  return html_text


def write_to_file(file_name, topics, values):
  with open(file_name, "w") as f:
    for topic_count, ll in zip(topics, values):
      f.write("%d,%f\n"%(topic_count, ll))


class Graph(O):
  def __init__(self, documents):
    O.__init__(self)
    document_map = OrderedDict()
    agency_map = OrderedDict()
    for document in documents:
      document_map[document.id] = document
      for agency in document.agencies:
        a_documents = agency_map.get(agency, [])
        a_documents.append(document.id)
        agency_map[agency] = a_documents
    self.agency_map = agency_map
    self.document_map = document_map
    self.vectorizer = None

  def vectorize(self, document_map=None, stop_words=STOP_WORDS, token_pattern=TOKEN_PATTERN):
    if self.vectorizer is None:
      stemmer = PorterStemmer()

      def stem(tokens):
        return [stemmer.stem(item) for item in tokens if item not in stop_words]

      def tokenize(doc):
        return stem(token_pattern.findall(doc))
      self.vectorizer = CountVectorizer(analyzer='word', tokenizer=tokenize, lowercase=True, stop_words=None)

      if document_map is None: document_map = self.document_map
      docs = [document.get_raw() for _, document in document_map.items()]
      doc_2_vec = self.vectorizer.fit_transform(docs).toarray()
      for vector, (_, document) in zip(doc_2_vec, document_map.items()):
        document.vector = vector
    return self.vectorizer, doc_2_vec

  def lda(self, n_topics=N_TOPICS, n_iter=ITERATIONS, random_state=RANDOM_STATE, alpha=ALPHA, beta=BETA):
    vectorizer, doc_2_vec = self.vectorize()
    alpha = alpha if alpha else 50 / n_topics
    beta = beta if beta else 0.01
    model = lda.LDA(n_topics=n_topics, alpha=alpha, eta=beta, n_iter=n_iter, random_state=random_state)
    # vectors = [document.vector for _, document in self.document_map.items()]
    model.fit(doc_2_vec)
    topics = model.ndz_
    for topic, (doc_id, document) in zip(topics, self.document_map.items()):
      document.topic_count = topic
      sum_t = sum(topic)
      sum_t = sum_t if sum_t else 0.00001
      document.topics_score = [t / sum_t for t in topic]
    return model, self.vectorizer.get_feature_names()


def report_lda(model, vocabulary, n_terms=10):
  for index, topic_dist in enumerate(model.topic_word_):
    topic_words = np.array(vocabulary)[np.argsort(topic_dist)][:-(n_terms + 1):-1]
    print('Topic {}: {}'.format(index, ', '.join(topic_words)))


def optimal_topics(file_name, to_file, fig_name):
  topics = range(2, 51, 1)
  topic_scores = []
  for topic_count in topics:
    print("TOPICS : ", topic_count)
    docs = Document.load(file_name)
    train, test = train_test_split(docs, test_size=0.2, random_state=RANDOM_STATE)
    graph = Graph(train)
    model, vocabulary = graph.lda(topic_count)
    perplex = perplexity(model, graph, test)
    topic_scores.append(perplex)
    # topic_scores.append(model.loglikelihood())
  write_to_file(to_file, topics, topic_scores)
  plot_topics(topics, topic_scores, fig_name)


def plot_topics(topics, scores, fig_name):
  plt.plot(topics, scores, 'ro')
  plt.xlabel('Topics ->')
  plt.ylabel('Log Perplexity ->')
  plt.savefig(fig_name)
  plt.clf()


def make_heatmap(arr, row_labels, column_labels, fig_name):
  plt.figure(figsize=(4, 3))
  df = pd.DataFrame(arr, columns=column_labels, index=row_labels)
  cax = plt.matshow(df, interpolation='nearest', cmap='hot_r')
  plt.colorbar(cax)
  plt.xticks(np.arange(len(list(df.columns))), list(df.columns), rotation="vertical")
  plt.yticks(np.arange(len(list(df.index))), list(df.index))
  plt.title("Topics to Conference Distribution", y=1.2)
  plt.savefig(fig_name, bbox_inches='tight')
  plt.clf()


def diversity(documents, fig_name):
  graph = Graph(documents)
  model, vocabulary = graph.lda()
  report_lda(model, vocabulary)
  heatmap_map = {}
  valid_agencies = []
  for agency, documents in graph.agency_map.items():
    if agency not in AGENCY_MAP or AGENCY_MAP[agency] not in PERMITTED_AGENCIES: continue
    topics = np.array([0] * model.n_topics)
    for document in documents:
      topics = np.add(topics, graph.document_map[document].topic_count)
    if sum(topics) > 0:
      heatmap_map[agency] = topics
      valid_agencies.append(agency)
  row_labels = ["Topic %1d" % ind for ind in range(model.n_topics)]
  column_labels = [AGENCY_MAP[agency] for agency in valid_agencies]
  heatmap_arr = []
  for agency in sorted(heatmap_map.keys()):
    tot = sum(heatmap_map[agency])
    dist = [top / tot for top in heatmap_map[agency]]
    heatmap_arr.append(dist)
  make_heatmap(np.transpose(heatmap_arr), row_labels, column_labels, fig_name)


def _optimal():
  for data in ['iot', 'sus_dev']:
    print(data.upper())
    file_name = 'files/data/%s.pkl' % data
    to_file = 'files/results/%s.csv' % data
    fig_name = 'files/results/%s.png' % data
    optimal_topics(file_name, to_file, fig_name)


def _main():
  keyword = "sus_dev"
  docs = Document.load('files/data/%s.pkl' % keyword)
  diversity(docs, 'files/results/figs/%s_diversity.png' % keyword)
  # model, vocabulary = graph.lda()
  # report_lda(model, vocabulary)


if __name__ == "__main__":
  _main()
