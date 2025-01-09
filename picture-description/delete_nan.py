import pandas as pd

# 讀取 CSV 檔案
df = pd.read_csv('train_1764_0520_merged.csv')

# 計算總資料筆數
total_rows = len(df)

# 刪除有缺失值的資料
df_cleaned = df.dropna(subset=['delivery', 'relevance', 'language'])

# 計算被刪除的資料筆數
deleted_rows = total_rows - len(df_cleaned)

# 輸出結果
print(f"被刪除的資料筆數: {deleted_rows}")
print(f"總資料筆數: {total_rows}")

# 如果需要，可以將清理後的資料另存成新的 CSV 檔案
df_cleaned.to_csv('cleaned_train_1764_0520_merged.csv', index=False)