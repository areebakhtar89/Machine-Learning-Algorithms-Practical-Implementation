# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 11:27:35 2020

@author: Areeb
"""
import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.randn(5,3), index = ['a','b','c','f','h'])
df = df.reindex(['a','b','c','d','e','f','g','h'])
df = df.rename(columns={0:'one',1:'two',2:'three'})

print(df[df['one'].notnull()])
print('--'*40)
df['sum'] = df.sum(axis=1)
print(df)
print(df['one'].sum())
print(df['two'].sum())
print(df['three'].sum())
print('--'*40)
print(df.fillna(0))
print('--'*40)
print(df.fillna(method='backfill'))
print('--'*40)
print(df.fillna(method='pad'))
print('--'*40)
print(df.fillna(df.mean()))