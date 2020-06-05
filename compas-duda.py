#!/usr/bin/env python
# coding: utf-8

def load_data():
  import pandas as pd
  df = pd.read_csv('data/compas-scores-two-years.csv', index_col=0)
  df = df[(df['days_b_screening_arrest'] <= 30) & (df['days_b_screening_arrest'] >= -30)]
  df = df[(df['race'] == 'African-American') | (df['race'] == 'Caucasian')]
  return df

if __name__ == '__main__':
  load_data()
