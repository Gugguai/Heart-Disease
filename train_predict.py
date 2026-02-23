import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# 1. 加载数据
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# 2. 数据预处理
# 转换目标变量
# 'Presence' -> 1, 'Absence' -> 0
if 'Heart Disease' in train_df.columns:
    train_df['Heart Disease'] = train_df['Heart Disease'].map({'Presence': 1, 'Absence': 0})
    print("Mapped 'Heart Disease' to binary values.")

# 分离特征和标签
X = train_df.drop(['id', 'Heart Disease'], axis=1)
y = train_df['Heart Disease']

# 测试集特征（保留 id 用于提交，但在预测时不需要）
X_test = test_df.drop(['id'], axis=1)
test_ids = test_df['id']

# 检查缺失值
print("\nChecking for missing values in training data:")
print(X.isnull().sum())
print("\nChecking for missing values in test data:")
print(X_test.isnull().sum())

# 3. 模型训练
print("\nTraining XGBoost model...")
# 划分验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化模型
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# 训练
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# 验证集评估
y_pred_val = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred_val)
print(f"\nValidation Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_val, y_pred_val))

# 4. 预测与提交
print("\nGenerating predictions for test set...")
test_predictions = model.predict(X_test)

# 创建提交 DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'Heart Disease': test_predictions
})

# 保存提交文件
submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'.")

# 预览提交文件
print("\nSubmission preview:")
print(submission.head())
