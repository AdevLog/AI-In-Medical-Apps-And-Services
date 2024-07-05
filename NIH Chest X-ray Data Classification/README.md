# 訓練 AI 模型分辨 NIH Chest X-rays 資料集，並分析訓練結果。
訓練模型: retrain.py \
小型資料集: nih_labels.csv \
預測模型: nih.py

選擇醫療 AI 模型 DenseNet121 做訓練，因其特點是加強特徵的傳遞還有減輕梯度消失問題，很適合用做醫療影像學習，模型評量指標選用AUC，用 sklearn.metrics 實現。
