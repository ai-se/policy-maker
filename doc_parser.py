from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from bs4 import BeautifulSoup
from collections import OrderedDict
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from nltk.stem.porter import PorterStemmer
from utils.lib import O, median, iqr
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
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram

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
    # if self.content is not None:
    #   return self.content
    # el
    if self.abstract is not None:
      return self.abstract
    else:
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
      document.topics_score = Graph.compute_topic_score(topic)
    return model, self.vectorizer.get_feature_names()

  @staticmethod
  def compute_topic_score(topic_count):
    sum_t = sum(topic_count)
    sum_t = sum_t if sum_t else 0.00001
    return [t / sum_t for t in topic_count]


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

# Settings for 10 rows and 5 columns
settings_10_5 = O(
    fig_size=(8, 8),
    col_axes=[0.3,   # col dendo left
              0.81,   # col dendo bottom
              0.36,   # col dendo width
              0.15],  # col dendo height
    row_axes=[0.0,    # row dendo left
              0.055,   # row dendo bottom
              0.23,   # row dendo width
              0.69],  # row dendo height
    plot_axes=[0.10,  # hm left
               0.05,  # hm bottom
               0.7,   # hm width
               0.7],  # hm height
)

# Settings for 10 rows and 4 columns
settings_10_4 = O(
    fig_size=(8, 8),
    col_axes=[0.325,   # col dendo left
              0.81,   # col dendo bottom
              0.29,   # col dendo width
              0.15],  # col dendo height
    row_axes=[0.0,    # row dendo left
              0.055,   # row dendo bottom
              0.23,   # row dendo width
              0.69],  # row dendo height
    plot_axes=[0.05,  # hm left
               0.05,  # hm bottom
               0.7,   # hm width
               0.7],  # hm height
)


settings_map = {
    "iot": settings_10_4,
    "sus_dev": settings_10_5
}


def make_dendo_heatmap(arr, row_labels, column_labels, figname, settings):
  df = pd.DataFrame(arr, columns=column_labels, index=row_labels)
  # Compute pairwise distances for columns
  col_clusters = linkage(pdist(df.T, metric='euclidean'), method='complete')
  # plot column dendrogram
  fig = plt.figure(figsize=settings.fig_size)
  axd2 = fig.add_axes(settings.col_axes)
  col_dendr = dendrogram(col_clusters, orientation='top',
                         color_threshold=np.inf)  # makes dendrogram black)
  axd2.set_xticks([])
  axd2.set_yticks([])
  # plot row dendrogram
  axd1 = fig.add_axes(settings.row_axes)
  row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')
  row_dendr = dendrogram(row_clusters, orientation='left',
                         count_sort='ascending',
                         color_threshold=np.inf)  # makes dendrogram black
  axd1.set_xticks([])
  axd1.set_yticks([])
  # remove axes spines from dendrogram
  for i, j in zip(axd1.spines.values(), axd2.spines.values()):
    i.set_visible(False)
    j.set_visible(False)
  # reorder columns and rows with respect to the clustering
  df_rowclust = df.ix[row_dendr['leaves'][::-1]]
  df_rowclust.columns = [df_rowclust.columns[col_dendr['leaves']]]
  # plot heatmap
  axm = fig.add_axes(settings.plot_axes)
  cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')
  fig.colorbar(cax)
  axm.set_xticks(np.arange(len(list(df_rowclust.columns))))
  axm.set_xticklabels(list(df_rowclust.columns), rotation="vertical")
  axm.set_yticks(np.arange(len(list(df_rowclust.index))))
  axm.set_yticklabels(list(df_rowclust.index))
  plt.savefig(figname, bbox_inches='tight')
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
  hm_name = 'files/results/figs/%s_diversity.png' % fig_name
  make_heatmap(np.transpose(heatmap_arr), row_labels, column_labels, hm_name)
  dendo_hm_name = 'files/results/figs/%s_dendo.png' % fig_name
  make_dendo_heatmap(np.transpose(heatmap_arr), row_labels, column_labels,
                     dendo_hm_name, settings_map[fig_name])
  return graph, model, vocabulary


def compute_document_topics(graph, lda_model):
  documents = graph.document_map.values()
  vectors = np.array([document.vector for document in documents])
  topic_counts = lda_model.transform(vectors)
  topic_deltas = []
  for topic, document in zip(topic_counts, documents):
    topic_deltas.append([round(i - j, 2) for i, j in zip(topic, document.topics_score)])
  print("#### Topic Deltas on Reconstruction")
  print("```")
  for i, topic_dist in enumerate(map(list, zip(*topic_deltas))):
    print("Topic %d ::  Min = %0.2f, Median = %0.2f, Max = %0.2f, IQR = %0.2f" %
          (i, min(topic_dist), median(topic_dist), max(topic_dist), iqr(topic_dist)))
  print("```")


def _optimal():
  for data in ['iot', 'sus_dev']:
    print(data.upper())
    file_name = 'files/data/%s.pkl' % data
    to_file = 'files/results/%s.csv' % data
    fig_name = 'files/results/figs/%s_topics.png' % data
    optimal_topics(file_name, to_file, fig_name)


def _main():
  for data in ['iot', 'sus_dev']:
    print("## %s" % data)
    docs = Document.load('files/data/%s.pkl' % data)
    graph, model, vocabulary = diversity(docs, data)
    compute_document_topics(graph, model)

  # model, vocabulary = graph.lda()
  # report_lda(model, vocabulary)


if __name__ == "__main__":
  _main()
