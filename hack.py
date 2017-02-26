from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

import requests
from bs4 import BeautifulSoup
from utils.lib import O
import re

BASE_URL = "https://www.nsf.gov"
PUB_URL = BASE_URL + "/publications/index.jsp"
START_URL = PUB_URL + "?org=NSF&archived=false&pub_type=Program+Announcements+%26+Information&nsf_org=NSF&search1="
SEPERATOR = "$|$"
FILE = "temp.csv"


class Document(O):
  id = 0

  def __init__(self, title=None, html=None, pdf=None):
    O.__init__(self)
    self.id = Document.id
    self.title = title
    self.html = html
    self.pdf = pdf
    Document.id += 1


def download(url=START_URL, rec=0):
  resp = requests.get(url)
  if resp.status_code != 200:
    print("Failed to fetch url: '%s'" % url)
    return
  soup = BeautifulSoup(resp.content)
  anchors = soup.find_all('a', attrs={'href': re.compile("/pubs/.+=NSF$")})
  htmls = soup.find_all('a', text='HTML')
  pdfs = soup.find_all('a', text='PDF')
  docs = []
  for anchor, html, pdf in zip(anchors, htmls, pdfs):
    html_uri, pdf_uri = BASE_URL + html['href'], BASE_URL + pdf['href']
    document = Document(anchor.text, html_uri, pdf_uri)
    docs.append(document)
    # save("files/html/%s.html" % document.id, html_uri)
    # save("files/pdf/%s.pdf" % document.id, pdf_uri)
  next_page = soup.find('a', text="Next")
  if next_page:
    print(rec)
    docs += download(PUB_URL + next_page['href'], rec + 1)
  return docs


def save(file_name, url):
  try:
    resp = requests.get(url)
    if resp.status_code != 200:
      print("Failed to fetch url while saving: '%s'" % url)
      return
    with open(file_name, 'wb') as f:
      f.write(resp.content)
  except requests.exceptions.SSLError as e:
    print("Invalid URL: %s" % url)
    print(e)


def dump(file_name, documents):
  with open(file_name, 'wb') as f:
    header = SEPERATOR.join(["ID", "Title", "HTML", "PDF"])
    f.write(header + "\n")
    for d in documents:
      line = SEPERATOR.join([str(d.id), d.title, d.html, d.pdf]).encode('utf8')
      f.write(line + "\n")


dump(FILE, download())
