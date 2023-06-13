# AICUP-2023-Evidence
## AI CUP 2023 春季賽「真相只有一個: 事實文字檢索與查核競賽」

### 運行環境
-  執行時使用 python版本 3.8 的 Jupyter notebook
- 請將 https://drive.google.com/drive/folders/1ivvE6Nkw4j9zDvSmj7QucrKPO41rAuGn?usp=sharing 內的檔案下載下來
	- `model.50.pt` 下載下來放至 `checkpoints\claim_verification\e30_bs32_5e-05_top5`的資料夾內
	- `val_acc=0.5266094_model.6500.pt` 下載下來放至 `checkpoints\sent_retrieval\e1_bs64_2e-05_neg0.03_top5`的資料夾內

### 資料
-  `checkpoints\sent_retrieval` 資料夾內會放置第二步驟 **Sentence Retrieval** 訓練模型的權重檔，而該資料夾內的命名方式，會根據這步驟的 Hyperparameter 來進行命名。
-  `checkpoints\claim_verification` 資料夾內會放置第三步驟 **Claim Verification** 訓練模型的權重檔，而該資料夾內的命名方式，會根據這步驟的 Hyperparameter 來進行命名。

### 訓練
提供 `AI_CUP0531.ipynb` 檔以實現本組 private score 最高之結果，只要將該檔案和下載下來的檔案們，置於同一個資料夾內，即可直接執行。
如根據前面的步驟自行訓練而在第三步驟 **Claim Verification** 有不一樣的結果的話，可以根據在 `checkpoints\sent_retrieval\{你的訓練參數}` 內找尋 val_acc 和 val_loss 最為理想的權重檔，並將下方程式碼 `ckpt_name` 改成該權重檔名。
```python
ckpt_name = "val_acc=0.5090_model.1150.pt"  #@param {type:"string"}
model = load_model(model, ckpt_name, CKPT_DIR)
predicted_label = run_predict(model, test_dataloader, device)
```

### 結果比較
- LR = Learning Rate
- 本組 val_acc 不是取分數最高的，而是評估過 val_loss 後取最適合的
- 「搜尋優化」是指本組改寫的 `get_pred_page` 部分，反之「未搜尋優化」則是指原大會所提供的程式碼。
- 「原資料」是指僅使用大會第一次提供的程式碼，「資料擴充」是指將大會兩次提供的train dataset合併 
- 「數據平衡」是指更改 `pair_with_wiki_sentences` 的 negative ratio 使 positive 和 negative 的數據更為平衡。

| model  | LR  | epoch  | val_acc  | Public score  | Private score  |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------  |
| 原資料+未搜尋優化+sijunhe/nezha-base-wwm | 7e-5  | 20  | 0.5340  | 0.3994  | X  |
| 原資料+未搜尋優化+sijunhe/nezha-base-wwm | 5e-5  | 20  | **0.5378**  | 0.4135  | X  |
| 原資料+搜尋優化+bert-base-chinese | 7e-5  | 20  | 0.4889  | 0.3802  | X  |
| 原資料+搜尋優化+sijunhe/nezha-base-wwm | 5e-5  | 20  | 0.5353  | 0.4125  | X  |
| 資料擴充+搜尋優化+sijunhe/nezha-base-wwm | 5e-5  | 30  | 0.5090  | **0.4395**  | 0.3791  |
| 資料擴充+搜尋優化+sijunhe/nezha-base-wwm | 5e-5  | 30  | 0.5004  | 0.4267 | 0.3811  |
| 資料擴充+搜尋優化+sijunhe/nezha-base-wwm+ 數據平衡 | 6e-5  | 30  | 0.5266  | 0.4186  | **0.3887**  |

### 小記
- 為節省儲存空間，在進行實驗時我們會定期清理一些權重，本組僅留下 private score 最佳之權重檔案
- 如要更改參數，請記得注意在 `checkpoints` 資料夾內的資料夾是否和 Hyperparameter 有相對應
