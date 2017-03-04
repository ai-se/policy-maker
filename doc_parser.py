from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

import json
import requests
from bs4 import BeautifulSoup
import re
import pickle
from utils.lib import O

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
  text = wrapper.get_text(separator=u" ", strip=True).encode("ascii", "ignore")
  text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
  return text
  # header = wrapper.find("h1", text="SUPPLEMENTARY INFORMATION:")
  # print(url)
  # if header:
  #   info_id = "p-" + header.get("id").split("-")[1]
  #   content_block = wrapper.find("p", attrs={"id": info_id})
  #   print(content_block.get_text())
  #   if content_block:
  #     text = content_block.get_text(strip=True).encode("ascii", "ignore")
  #     text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
  #     return text
  # return None


Document.dump(parse(IOT_URL, True), IOT_PATH)
Document.dump(parse(SUS_DEV_URL, True), SUS_DEV_PATH)
print(len(Document.load(IOT_PATH)))
print(len(Document.load(SUS_DEV_PATH)))
