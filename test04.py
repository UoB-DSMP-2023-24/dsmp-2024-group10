import pandas as pd

data = pd.read_csv("vdjdb.txt", sep="\t")
data = data.drop(columns=["reference.id", "method", "meta", 
                          "cdr3fix", "web.method", "web.method.seq", 
                          "web.cdr3fix.nc", "web.cdr3fix.unmp"])
data = data[data["vdjdb.score"] > -1]
data = data.dropna()
data = data.drop_duplicates()
alpha = data[data["gene"] == "TRB"]
print(len(alpha))
letters=set()
for index, row in data.iterrows():
    for letter in row["cdr3"]:
        letters.add(letter)

print("letter: ", letters)

human_alpha = data[(data["species"] == "HomoSapiens") & (data["gene"] == "TRA")]
human_beta = data[(data["species"] == "HomoSapiens") & (data["gene"] == "TRB")]
mouse_alpha = data[(data["species"] == "MusMusculus") & (data["gene"] == "TRA")]
mouse_beta = data[(data["species"] == "MusMusculus") & (data["gene"] == "TRB")]

alpha_beta = data.loc[(data['complex.id']!=0)]
print(alpha_beta)
print(len(alpha_beta))
# df_a_b=df_a_b.rename(columns={'antigen.epitope': 'epitope', 'v.segm': 'v_a_gene','j.segm': 'j_a_gene','cdr3': 'cdr3_a_aa','species':'subject'})

# print(merged_rows)

print("len(human_alpha): ", len(human_alpha))
print("len(human_beta): ", len(human_beta))
print("len(mouse_alpha): ", len(mouse_alpha))
print("len(mouse_beta)", len(mouse_beta))