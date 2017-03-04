from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

import json
import requests
from utils.lib import O


JSON_PATH = 'files/data/iot.json'


class Document(O):
  id = 0

  def __init__(self, **kwargs):
    O.__init__(self, **kwargs)
    Document.id += 1

  @staticmethod
  def from_json(json_obj):
    return Document(title=json_obj['title'],
                    abstract=json_obj['abstract'],
                    html_url=json_obj['html_url'],
                    pdf_url=json_obj['pdf_url'],
                    agencies=[agency['name'] for agency in json_obj['agencies']],
                    publication_date=json_obj['publication_date'])


def get_json(src, from_internet=False):
  if from_internet:
    return json.loads(requests.get(src).text)
  else:
    with open(src) as f:
      return json.loads(f.read())


def parse(file_name, from_internet=False):
  json_obj = get_json(file_name, from_internet)
  documents = [Document.from_json(result) for result in json_obj['results']]
  if 'next_page_url' in json_obj:
    documents += parse(json_obj['next_page_url'], from_internet=True)
  return documents


print(len(parse(JSON_PATH)))