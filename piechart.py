import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('vdjdb.csv')

epitope_column = 'antigen.epitope'

epitope_counts = data[epitope_column].value_counts()


top_10_epitopes = epitope_counts.head(10)

others_count = epitope_counts[10:].sum()

pie_data = pd.Series([others_count], index=['Others'])._append(top_10_epitopes)

colors = ['#d3d3d3']
additional_colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0','#ffb3e6', '#c4e17f', '#76d7c4', '#f7c6c7', '#f7b7a3']
colors.extend(additional_colors)

explode = [0.1 if pie_data[index] < others_count else 0 for index in range(len(pie_data))]


plt.figure(figsize=(10, 8))
patches, texts, autotexts = plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140, colors=colors, explode=explode, textprops={'fontsize': 12})


plt.setp(autotexts, size=10, weight='bold', color='darkblue')
plt.setp(texts, size=14)
plt.title('Distribution of Epitopes', fontsize=16, verticalalignment='bottom')

plt.axis('equal')
plt.show()











# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Load the dataset
# data = pd.read_csv('vdjdb.csv')
#
# epitope_column = 'antigen.epitope'
# epitope_counts = data[epitope_column].value_counts()
#
# top_10_epitopes = epitope_counts.head(10)
#
# plt.figure(figsize=(10, 8))
# patches, texts, autotexts = plt.pie(top_10_epitopes, labels=top_10_epitopes.index, autopct='%1.1f%%', startangle=140,
#                                     textprops={'fontsize': 12})
#
# plt.setp(texts, size=15)
#
# plt.title('Distribution of Top 10 Epitopes', fontsize=15, verticalalignment='bottom')
# plt.axis('equal')
# plt.show()
