import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import time

# 1. 加载数据
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. 数据预处理
if 'Heart Disease' in train_df.columns:
    train_df['Heart Disease'] = train_df['Heart Disease'].map({'Presence': 1, 'Absence': 0})
    print("Mapped 'Heart Disease' to binary values.")

X = train_df.drop(['id', 'Heart Disease'], axis=1)
y = train_df['Heart Disease']
X_test = test_df.drop(['id'], axis=1)
test_ids = test_df['id']

# 3. 定义优化后的参数
# 移除了 early_stopping_rounds，因为它在 fit 中可能不受支持，在构造函数中传入
params = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 4,
    'min_child_weight': 3,
    'gamma': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 50  # 移至构造函数
}

# use_label_encoder 在新版本中已移除，这里去掉

print("\nUsing optimized parameters:")
for k, v in params.items():
    print(f"  {k}: {v}")

# 4. 交叉验证评估
print("\nStarting 5-Fold Cross-Validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBClassifier(**params)
    
    # fit 中移除 early_stopping_rounds，因为它已经在 params 中了
    model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        verbose=False
    )
    
    val_pred = model.predict(X_val_fold)
    score = accuracy_score(y_val_fold, val_pred)
    cv_scores.append(score)
    models.append(model)
    
    # 注意：best_iteration 在新版本中可能需要通过 get_booster() 获取，或者直接在 model 上
    best_iter = getattr(model, 'best_iteration', -1)
    print(f"Fold {fold+1} Accuracy: {score:.4f} (Best Iteration: {best_iter})")

mean_score = np.mean(cv_scores)
print(f"\nMean CV Accuracy: {mean_score:.4f}")

# 5. 生成最终预测 (使用所有折叠模型的平均预测)
print("\nGenerating predictions using ensemble of CV models...")
test_preds_prob = np.zeros(len(X_test))

for model in models:
    test_preds_prob += model.predict_proba(X_test)[:, 1]

# 取平均
test_preds_prob /= len(models)
# 转换为类别
test_predictions = (test_preds_prob > 0.5).astype(int)

# 6. 保存提交文件
submission = pd.DataFrame({
    'id': test_ids,
    'Heart Disease': test_predictions
})

submission.to_csv('submission_optimized_cv.csv', index=False)
print("Optimized submission file saved as 'submission_optimized_cv.csv'.")
