import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import os
import matplotlib.font_manager as fm

# 配置 Matplotlib 中文支持
plt.rcParams['axes.unicode_minus'] = False
# 尝试加载 SimHei.ttf
font_path = 'SimHei.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['SimHei']
else:
    # 回退方案
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']

# 1. 加载数据
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
submission_df = pd.read_csv('submission_optimized_cv.csv')

# 2. 数据预处理 (与 app.py 保持一致)
processed_df = train_df.copy()
if processed_df['Heart Disease'].dtype == 'object':
     processed_df['Heart Disease'] = processed_df['Heart Disease'].map({'Presence': 1, 'Absence': 0})

if processed_df['Heart Disease'].isnull().any():
    processed_df = processed_df.dropna(subset=['Heart Disease'])
    
X = processed_df.drop(['id', 'Heart Disease'], axis=1)
y = processed_df['Heart Disease']

# 划分数据集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 训练模型
print("Training model...")
model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=3,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# 4. 生成图表并保存
output_dir = 'images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Generating plots...")

# --- 特征重要性 ---
fig6, ax6 = plt.subplots(figsize=(10, 8))
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns
sns.barplot(x=importances[indices], y=features[indices], ax=ax6, palette="viridis")
ax6.set_title("XGBoost 特征重要性")
ax6.set_xlabel("重要性分数")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
plt.close(fig6)
print("Saved feature_importance.png")

# --- 混淆矩阵 ---
y_pred = model.predict(X_val)
cm = confusion_matrix(y_val, y_pred)
fig7, ax7 = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax7)
ax7.set_title("混淆矩阵")
ax7.set_xlabel("预测值")
ax7.set_ylabel("真实值")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close(fig7)
print("Saved confusion_matrix.png")

# --- ROC 曲线 ---
y_prob = model.predict_proba(X_val)[:, 1]
fpr, tpr, thresholds = roc_curve(y_val, y_prob)
roc_auc = auc(fpr, tpr)
fig8, ax8 = plt.subplots(figsize=(10, 8))
ax8.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
ax8.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax8.set_xlim([0.0, 1.0])
ax8.set_ylim([0.0, 1.05])
ax8.set_xlabel('False Positive Rate')
ax8.set_ylabel('True Positive Rate')
ax8.set_title('Receiver Operating Characteristic (ROC)')
ax8.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
plt.close(fig8)
print("Saved roc_curve.png")

# --- 预测结果分布 ---
pred_counts = submission_df['Heart Disease'].value_counts()
fig9, ax9 = plt.subplots(figsize=(8, 6))
sns.barplot(x=pred_counts.index, y=pred_counts.values, ax=ax9, palette="pastel")
ax9.set_title("测试集预测结果分布 (0: Absence, 1: Presence)")
ax9.set_ylabel("数量")
ax9.set_xlabel("预测类别")
for i, v in enumerate(pred_counts.values):
    ax9.text(i, v + 50, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'))
plt.close(fig9)
print("Saved prediction_distribution.png")

print("All plots generated successfully!")
