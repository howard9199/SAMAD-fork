from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login
import pandas as pd

# 讀取 CSV 文件
df = pd.read_csv("/datas/store163/howard/samad/SAMAD/fulltest_1964_0520.csv")

# 將 DataFrame 轉換為 Dataset
fulltest_dataset = Dataset.from_pandas(df)

# 從 hub 加載現有的數據集
existing_dataset = load_dataset("ntnu-smil/Unseen_1964")

# 創建新的 DatasetDict
new_dataset = DatasetDict({
    'train': existing_dataset['train'],
    'validation': existing_dataset['validation'],
    'test': existing_dataset['test'],
    'fulltest': fulltest_dataset
})

# 登入 Hugging Face (需要先設定 HUGGING_FACE_TOKEN 環境變量)
#login()

# 推送到 hub
new_dataset.push_to_hub("ntnu-smil/Unseen_1964", private=False)
