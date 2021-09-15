import pandas as pd, numpy as np

train_data = pd.read_csv('data/train.csv')
print(train_data['tot_proteins'].value_counts())	
