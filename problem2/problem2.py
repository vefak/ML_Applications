# -*- coding: utf-8 -*-
"""
Created on Thu May  7 21:17:06 2020

@author: vmakm
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_excel('bank-additional-full.xlsx')

X = dataset.iloc[:, :20].values
y = dataset.iloc[:, -1].values