from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
import numpy as np

__author__ = "panzer"


class O:
  def __init__(self, **d):
    self.has().update(**d)

  def has(self):
    return self.__dict__

  def update(self, **d):
    self.has().update(d)
    return self

  def __repr__(self):
    show = [':%s %s' % (k, self.has()[k]) for k in sorted(self.has().keys()) if k[0] is not "_"]
    txt = ' '.join(show)
    if len(txt) > 60:
      show = map(lambda x: '\t' + x + '\n', show)
    return '{' + ' '.join(show) + '}'

  def __getitem__(self, item):
    return self.has().get(item)

  def __setitem__(self, key, value):
    self.has()[key] = value


def median(arr):
  return np.median(arr).item()


def iqr(arr):
  return np.subtract(*np.percentile(arr, [75, 25])).item()


def say(*lst):
  print(*lst, end="")
  sys.stdout.flush()
