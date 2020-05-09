#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 23:44:03 2020

@author: vefak
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('bank-additional-full.csv')
X = dataset.iloc[:20, :].values
y = dataset.iloc[:, 3].values
