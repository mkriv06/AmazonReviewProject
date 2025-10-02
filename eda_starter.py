import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("fake-reviews.csv")

print(df.head())

print(df.isnull().sum())

df["char_length"] = df["text_"].apply(len)

df["word_count"] = df["text_"].str.split().apply(len)

sns.boxplot(x="char_length", y="label", data=df)

plt.title("Character Length of Reviews by Label")

plt.xlabel("Character length")

plt.ylabel("Review label")

plt.show()