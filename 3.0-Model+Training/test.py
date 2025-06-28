import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("3.0-Model+Training\Algerian_forest_fires_dataset_UPDATE.csv")
print(df.head())

print(df.info())

print(df.describe())
