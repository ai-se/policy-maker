from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

import json
import requests
from urllib import quote
from utils.lib import O
import cPickle as pkl
import csv

__author__ = "bigfatnoob"


class Award(O):
  def __init__(self):
    O.__init__(self)

  @staticmethod
  def header():
    return ['id', 'agency', 'date', 'awardee', 'affiliation', 'title', 'fund', 'publicAccess', 'report']

  def row(self):
    i = self
    report = None
    if i.report:
      report = i.report.encode('ascii', 'ignore').replace('\n', ' ').replace('\t', ' ')
    return [i.id, i.agency, i.date, i.awardee, i.affiliation, i.title, i.fund, i.is_public_access_mandate, report]


def fetch_awards(keywords):
  awards = {}
  for keyword in keywords:
    awards_json = fetch_json_for_keyword(keyword)
    for award_json in awards_json['award']:
      award_id = award_json['id']
      if award_id in awards: continue
      award = Award()
      award.id = award_id
      award.affiliation = award_json['awardeeName']
      award.fund = int(award_json['fundsObligatedAmt'])
      award.awardee = "%s %s" % (award_json['piFirstName'], award_json['piLastName'])
      award.title = award_json['title']
      award.date = award_json['date']
      award.agency = award_json['agency']
      award.is_public_access_mandate = True if award_json['publicAccessMandate'] else False
      award['report'] = fetch_award_report(award_id)
      awards[award_id] = award
  save_data(awards, 'iot_awards')
  return awards


def fetch_award_report(award_id):
  base_uri = "http://api.nsf.gov/services/v1/awards/%s/projectoutcomes.json"
  uri = base_uri % award_id
  print("Fetching : %s" % uri)
  response = requests.get(uri)
  if response.status_code == 200:
    statuses_json = json.loads(response.text)['response']['award']
    status = ""
    for status_json in statuses_json:
      if "projectOutComesReport" in status_json:
        status += " " + status_json["projectOutComesReport"]
    if status:
      return status
  else:
    print("Failed to retrieve json. Status: %d; Message: %s" % (response.status_code, response.text))
  return None


def fetch_json_for_keyword(keyword):
  base_uri = "http://api.nsf.gov/services/v1/awards.json"
  keyword = quote(keyword, safe='')
  uri = "%s?keyword=%s" % (base_uri, keyword)
  print("Fetching : %s" % uri)
  response = requests.get(uri)
  if response.status_code == 200:
    return json.loads(response.text)['response']
  else:
    print("Failed to retrieve json. Status: %d; Message: %s" % (response.status_code, response.text))
    return None


def save_data(data, file_name):
  file_name = "data/%s.pkl" % file_name
  with open(file_name, 'wb') as f:
    pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)


def load_data(file_name):
  file_name = "data/%s.pkl" % file_name
  with open(file_name) as f:
    return pkl.load(f)


def dump_exists(file_name):
  file_name = "data/%s.pkl" % file_name
  return os.path.isfile(file_name)


def load_iot():
  if dump_exists("iot_awards"):
    return load_data("iot_awards")
  else:
    return fetch_awards(["IOT", "Internet Of Things"])


def dump_csv(read_file, write_file):
  read = open(read_file, 'rb')
  write = open(write_file, 'wb')
  awards = pkl.load(read)
  csv_writer = csv.writer(write, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
  csv_writer.writerow(Award.header())
  for award in awards.values():
    csv_writer.writerow(award.row())
  read.close()
  write.close()


def _main():
  awards = load_iot()
  print(len(awards))


if __name__ == "__main__":
  dump_csv("data/iot_awards.pkl", "data/iot_awards.csv")
