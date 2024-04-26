import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('vdjdb.csv')

epitope_column = 'antigen.epitope'
epitope_counts = data[epitope_column].value_counts()

top_10_epitopes = epitope_counts.head(10)

plt.figure(figsize=(10, 8))
patches, texts, autotexts = plt.pie(top_10_epitopes, labels=top_10_epitopes.index, autopct='%1.1f%%', startangle=140,
                                    textprops={'fontsize': 12})

plt.setp(texts, size=15)

plt.title('Distribution of Top 10 Epitopes', fontsize=15, verticalalignment='bottom')
plt.axis('equal')
plt.show()
