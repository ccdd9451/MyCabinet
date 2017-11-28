#!/bin/env python
# coding: utf-8

from xilio import load
from pathlib import Path
import numpy as np

data = [load(x)["Y"] for x in Path(".").glob("*_avg")]
print(tuple(map(np.mean, data)))
print(tuple(map(lambda x:np.std(x, ddof=1), data)))
