# from datasets import load_dataset
# dataset = load_dataset("flwrlabs/femnist")
import pandas as pd

# 加载数据
df = pd.read_parquet("train-00000-of-00001.parquet")
print(df.head())  # 查看数据的前几行
