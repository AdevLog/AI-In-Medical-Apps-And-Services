# 使用NER model進行名詞標註
* 流程： \
a. 將範例 sample_data.txt 處理成 BIO 格式。 \
b. 將資料分成 Training 及 Testing。 \
c. 將每個字轉成 word vector 並給予標註。 \
d. 使用 CRF 預測並輸出 f1 score。 \
e. 輸出預測結果至 output.csv。

* 改進 f1 score 可以有以下幾種方式： \
a. 抓取不同特徵，如 POS-tag，word_length，word_position，盡量描述資料分佈狀態使模型有更好的預測。 \
b. 使用不同 NER model：神經網路模型去學習特徵，取代人工調整參數。 \
c. 使用品質更好的 word2vec 特徵。


本文不修改優化方式，使用 Chinese Word Vectors 中文詞向量(https://github.com/Embedding/Chinese-Word-Vectors) 製作的word2vec 當作輸入特徵依據；另外算出整個 data 的 f1 score。因此，原本使用的 cna.cbow.cwe_p.tar_g.512d.0.txt 改成
sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5_2
