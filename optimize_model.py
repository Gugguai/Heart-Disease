import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
import time

# 1. 加载数据
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. 数据预处理
# 转换目标变量 'Presence' -> 1, 'Absence' -> 0
if 'Heart Disease' in train_df.columns:
    train_df['Heart Disease'] = train_df['Heart Disease'].map({'Presence': 1, 'Absence': 0})
    print("Mapped 'Heart Disease' to binary values.")

# 分离特征和标签
X = train_df.drop(['id', 'Heart Disease'], axis=1)
y = train_df['Heart Disease']

# 测试集特征
X_test = test_df.drop(['id'], axis=1)
test_ids = test_df['id']

# 3. 超参数搜索空间
# 我们将使用 RandomizedSearchCV 进行搜索，这比网格搜索更有效率
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'max_depth': [3, 4, 5, 6, 8, 10],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05],
    'reg_lambda': [0, 0.001, 0.005, 0.01, 0.05]
}

# 初始化模型
xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1  # 使用所有核心
)

# 4. 执行随机搜索
print("Starting RandomizedSearchCV...")
# 使用 StratifiedKFold 确保每个折叠中类别的比例一致
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=50,  # 尝试 50 种组合
    scoring='accuracy',
    cv=skf,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

start_time = time.time()
random_search.fit(X, y)
end_time = time.time()

print(f"Optimization finished in {end_time - start_time:.2f} seconds.")
print(f"Best parameters found: {random_search.best_params_}")
print(f"Best cross-validation accuracy: {random_search.best_score_:.4f}")

# 5. 使用最佳模型进行最终预测
best_model = random_search.best_estimator_

print("\nGenerating predictions with the best model...")
test_predictions = best_model.predict(X_test)

# 创建提交 DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'Heart Disease': test_predictions
})

# 保存提交文件
submission.to_csv('submission_optimized.csv', index=False)
print("Optimized submission file saved as 'submission_optimized.csv'.")
