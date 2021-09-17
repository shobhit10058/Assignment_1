import pandas as pd
import seaborn as sns
from IPython.display import display
from matplotlib import pyplot as plt

df = pd.read_csv("data/train.csv")
display(df.head())
display(df.info)
#display(df.columns)
display(df['is_patient'].value_counts())
display(df.nunique())	
display(df.describe())
display(df.isnull().sum())

g = sns.FacetGrid(df, col="gender", hue="is_patient", height=5, aspect=1.25, ylim=(0, 40))
g.map(sns.scatterplot, "age", "tot_bilirubin", alpha=.7)
g.add_legend()
plt.show()

sns.relplot(data=df, x="direct_bilirubin", y="tot_bilirubin", hue="is_patient")
plt.show()

sns.scatterplot(data=df, x="age", y="tot_proteins", hue="is_patient", palette="bright")
plt.show()

fig, axs2 = plt.subplots(ncols=2, figsize=(20, 6), nrows = 1)
sns.scatterplot(data=df, x="age", hue="is_patient", y="albumin", palette="deep", ax=axs2[0])
sns.scatterplot(data=df, x="age", hue="is_patient", y="ag_ratio", palette="deep", ax=axs2[1])
plt.show()

sns.set_style('whitegrid')
fig, axs = plt.subplots(ncols=3, figsize=(20, 6), nrows = 1)
sns.countplot(x = 'is_patient', data = df, ax = axs[0])
sns.countplot(x = 'gender', data = df, ax = axs[1])
sns.countplot(x = 'is_patient', hue = 'gender', data = df,ax = axs[2])
plt.show()

sns.displot(df['age'], kde = True, color ='red', bins = 30)
plt.show()