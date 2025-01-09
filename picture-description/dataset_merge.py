import pandas as pd

# 讀取 IS-itemized_train_base.csv
itemized_train_df = pd.read_csv('IS-itemized_train_base.csv')

# 讀取 IS-itemized_dev_base.csv
itemized_dev_df = pd.read_csv('IS-itemized_dev_base.csv')

# 合併資料
itemized_df = pd.concat([itemized_train_df, itemized_dev_df])

# 去除 speaker_id 中的 '_300'
itemized_df['speaker_id'] = itemized_df['speaker_id'].str.replace('_300', '', regex=False)

# 選取需要的欄位
itemized_df = itemized_df[['speaker_id', 'delivery', 'relevance', 'language']]

# 讀取 train_1764_0520.csv
train_df = pd.read_csv('train_1764_0520.csv')

train_df['speaker_id'] = train_df['speaker_id'].astype(str)
itemized_df['speaker_id'] = itemized_df['speaker_id'].astype(str)

# 合併資料，根據 speaker_id
merged_df = pd.merge(train_df, itemized_df, on='speaker_id', how='left')

# 將合併後的資料保存為新的 CSV 檔案
merged_df.to_csv('train_1764_0520_merged.csv', index=False)