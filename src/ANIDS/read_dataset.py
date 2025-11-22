import pandas as pd 

df_2017 = pd.read_csv("./rawdata/2017.csv")
df_2018 = pd.read_csv("./rawdata/2018.csv")

print(df_2017[' Label'].value_counts())
print(df_2018["Label"].value_counts())