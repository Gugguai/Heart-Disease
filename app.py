import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
import io

# 1. 配置页面
st.set_page_config(
    page_title="心脏病预测分析系统",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 配置 Matplotlib 中文支持
# 尝试使用 SimHei 字体，如果不可用则回退到系统默认
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 3. 数据加载与缓存
@st.cache_data
def load_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    submission_df = pd.read_csv('submission_optimized_cv.csv')
    return train_df, test_df, submission_df

train_df, test_df, submission_df = load_data()

# 4. 侧边栏导航
st.sidebar.title("导航栏")
page = st.sidebar.radio("选择页面", ["项目介绍", "数据概览", "探索性分析 (EDA)", "特征工程", "模型可视化", "预测结果"])

# --- 页面内容 ---

# A. 项目介绍
if page == "项目介绍":
    st.title("❤️ 心脏病预测分析系统")
    st.markdown("""
    本系统旨在通过机器学习技术预测患者是否患有心脏病。
    
    ### 项目目标
    利用 XGBoost 模型对提供的临床数据进行分析，识别心脏病的高风险因素，并对测试集进行预测。
    
    ### 数据集来源
    - **train.csv**: 包含训练数据和目标标签 `Heart Disease`。
    - **test.csv**: 需要进行预测的测试数据。
    
    ### 技术栈
    - **数据处理**: Pandas, NumPy
    - **可视化**: Matplotlib, Seaborn
    - **建模**: XGBoost
    - **展示**: Streamlit
    """)
    
    st.image("https://img.freepik.com/free-vector/human-heart-anatomy-diagram_1308-125345.jpg?w=826&t=st=1708680000~exp=1708680600~hmac=...", caption="心脏结构示意图 (仅作装饰)", use_column_width=False, width=400)


# B. 数据概览
elif page == "数据概览":
    st.title("📊 数据概览")
    
    st.subheader("1. 数据集预览")
    st.write("训练集前 5 行：")
    st.dataframe(train_df.head())
    
    st.subheader("2. 数据统计描述")
    st.write(train_df.describe())
    
    st.subheader("3. 数据形状")
    st.write(f"训练集形状: {train_df.shape}")
    st.write(f"测试集形状: {test_df.shape}")
    
    st.subheader("4. 缺失值检查")
    missing_values = train_df.isnull().sum()
    if missing_values.sum() == 0:
        st.success("🎉 数据集中没有缺失值！")
    else:
        st.warning("⚠️ 数据集中存在缺失值：")
        st.write(missing_values[missing_values > 0])
    
    # 可视化缺失值 (即使是0也可以展示)
    fig, ax = plt.subplots(figsize=(10, 5))
    missing_values.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title("各特征缺失值数量")
    ax.set_ylabel("缺失数量")
    ax.set_xlabel("特征")
    st.pyplot(fig)


# C. 探索性分析 (EDA)
elif page == "探索性分析 (EDA)":
    st.title("🔍 探索性数据分析 (EDA)")
    
    # 目标变量分布
    st.subheader("1. 目标变量分布 (Heart Disease)")
    target_counts = train_df['Heart Disease'].value_counts()
    
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title("心脏病患病比例")
    st.pyplot(fig1)
    
    # 数值特征分布
    st.subheader("2. 数值特征分布")
    numerical_features = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
    selected_num_feature = st.selectbox("选择数值特征进行查看", numerical_features)
    
    # 直方图
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.histplot(data=train_df, x=selected_num_feature, hue='Heart Disease', kde=True, element="step", ax=ax2)
    ax2.set_title(f"{selected_num_feature} 分布 (按是否患病)")
    st.pyplot(fig2)
    
    # 箱线图
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=train_df, x='Heart Disease', y=selected_num_feature, ax=ax3)
    ax3.set_title(f"{selected_num_feature} 箱线图 (按是否患病)")
    st.pyplot(fig3)
    
    # 类别特征分布
    st.subheader("3. 类别特征分布")
    categorical_features = ['Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium']
    selected_cat_feature = st.selectbox("选择类别特征进行查看", categorical_features)
    
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.countplot(data=train_df, x=selected_cat_feature, hue='Heart Disease', ax=ax4)
    ax4.set_title(f"{selected_cat_feature} 分布 (按是否患病)")
    st.pyplot(fig4)
    
    # 相关性热力图
    st.subheader("4. 特征相关性热力图")
    # 需要先将目标变量转换为数值才能计算相关性
    temp_df = train_df.copy()
    temp_df['Heart Disease'] = temp_df['Heart Disease'].map({'Presence': 1, 'Absence': 0})
    # 移除 id 列
    if 'id' in temp_df.columns:
        temp_df = temp_df.drop('id', axis=1)
        
    corr = temp_df.corr()
    fig5, ax5 = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax5)
    ax5.set_title("特征相关性矩阵")
    st.pyplot(fig5)


# D. 特征工程
elif page == "特征工程":
    st.title("🛠️ 特征工程")
    
    st.markdown("### 1. 目标变量编码")
    st.code("""
    # 将文本标签转换为数值
    train_df['Heart Disease'] = train_df['Heart Disease'].map({'Presence': 1, 'Absence': 0})
    """)
    
    # 实际执行转换以展示
    processed_df = train_df.copy()
    processed_df['Heart Disease'] = processed_df['Heart Disease'].map({'Presence': 1, 'Absence': 0})
    
    st.write("转换后的数据预览：")
    st.dataframe(processed_df.head())
    
    st.markdown("### 2. 特征选择")
    st.write("移除了 `id` 列，因为它不包含预测信息。")
    
    st.markdown("### 3. 处理后的数据分布")
    st.write("处理后的目标变量分布（0: Absence, 1: Presence）：")
    st.bar_chart(processed_df['Heart Disease'].value_counts())


# E. 模型可视化
elif page == "模型可视化":
    st.title("🤖 XGBoost 模型可视化")
    
    # 准备数据
    processed_df = train_df.copy()
    if processed_df['Heart Disease'].dtype == 'object':
         processed_df['Heart Disease'] = processed_df['Heart Disease'].map({'Presence': 1, 'Absence': 0})
    
    # 确保没有 NaN
    if processed_df['Heart Disease'].isnull().any():
        processed_df = processed_df.dropna(subset=['Heart Disease'])
        
    X = processed_df.drop(['id', 'Heart Disease'], axis=1)
    y = processed_df['Heart Disease']
    
    # 划分数据集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型 (使用优化后的参数)
    @st.cache_resource
    def train_model():
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
        return model

    with st.spinner('正在训练模型，请稍候...'):
        model = train_model()
    
    st.success("模型训练完成！")
    
    # 特征重要性
    st.subheader("1. 特征重要性")
    fig6, ax6 = plt.subplots(figsize=(10, 8))
    # 获取特征重要性
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X.columns
    
    sns.barplot(x=importances[indices], y=features[indices], ax=ax6, palette="viridis")
    ax6.set_title("XGBoost 特征重要性")
    ax6.set_xlabel("重要性分数")
    st.pyplot(fig6)
    
    # 混淆矩阵
    st.subheader("2. 混淆矩阵 (Validation Set)")
    y_pred = model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    
    fig7, ax7 = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax7)
    ax7.set_title("混淆矩阵")
    ax7.set_xlabel("预测值")
    ax7.set_ylabel("真实值")
    st.pyplot(fig7)
    
    st.text("分类报告：")
    st.text(classification_report(y_val, y_pred))
    
    # ROC 曲线
    st.subheader("3. ROC 曲线")
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
    st.pyplot(fig8)


# F. 预测结果
elif page == "预测结果":
    st.title("📋 最终预测结果")
    
    st.subheader("1. 提交文件预览")
    st.write("这是根据测试集生成的预测结果：")
    st.dataframe(submission_df.head(10))
    
    st.subheader("2. 预测结果分布")
    pred_counts = submission_df['Heart Disease'].value_counts()
    
    fig9, ax9 = plt.subplots(figsize=(8, 6))
    sns.barplot(x=pred_counts.index, y=pred_counts.values, ax=ax9, palette="pastel")
    ax9.set_title("测试集预测结果分布 (0: Absence, 1: Presence)")
    ax9.set_ylabel("数量")
    ax9.set_xlabel("预测类别")
    # 添加数值标签
    for i, v in enumerate(pred_counts.values):
        ax9.text(i, v + 50, str(v), ha='center', fontweight='bold')
    st.pyplot(fig9)
    
    st.subheader("3. 下载结果")
    
    # 将 DataFrame 转换为 CSV 字节流
    csv = submission_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 下载 submission.csv",
        data=csv,
        file_name='submission_optimized_cv.csv',
        mime='text/csv',
    )
