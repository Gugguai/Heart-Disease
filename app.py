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
import matplotlib.font_manager as fm
import os
import requests

# 优先使用本地 SimHei.ttf
font_path = 'SimHei.ttf'
if not os.path.exists(font_path):
    # 如果本地没有，尝试下载
    try:
        url = "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf"
        response = requests.get(url)
        with open(font_path, "wb") as f:
            f.write(response.content)
    except Exception as e:
        st.warning(f"无法下载中文字体文件，可能会导致中文显示乱码: {e}")

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 字体文件名为 SimHei.ttf，通常对应的 Family 是 SimHei
else:
    # 回退方案
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# GitHub 仓库原始图片的基础 URL (用于 Fallback)
GITHUB_REPO_URL = "https://raw.githubusercontent.com/Gugguai/Heart-Disease/main/images/"

def get_image_path(filename):
    """
    获取图片的路径。
    1. 尝试本地 images 目录
    2. 如果本地没有，尝试返回 GitHub Raw URL
    """
    local_path = os.path.join("images", filename)
    if os.path.exists(local_path):
        return local_path
    else:
        return f"{GITHUB_REPO_URL}{filename}"

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
    st.write("以下图表展示了基于训练集和验证集的模型性能。")
    
    # 自动检查并生成图片
    image_dir = "images"
    feature_importance_path = os.path.join(image_dir, "feature_importance.png")
    
    if not os.path.exists(feature_importance_path):
        with st.spinner("检测到本地缺少静态图表，正在首次生成（可能需要几分钟）..."):
            try:
                from generate_plots import generate_all_plots
                generate_all_plots(output_dir=image_dir)
                st.success("图表生成完成！")
            except Exception as e:
                pass # 忽略错误，尝试使用远程图片
    
    st.subheader("1. 特征重要性")
    st.write("展示了对模型预测贡献最大的特征。")
    try:
        st.image(get_image_path("feature_importance.png"), caption="XGBoost 特征重要性")
    except Exception:
        st.error("无法加载特征重要性图片。")
    
    st.subheader("2. 混淆矩阵 (Validation Set)")
    st.write("展示了模型在验证集上的分类准确度。")
    try:
        st.image(get_image_path("confusion_matrix.png"), caption="混淆矩阵")
    except Exception:
        st.error("无法加载混淆矩阵图片。")
    
    st.subheader("3. ROC 曲线")
    st.write("展示了模型的真正率与假正率之间的权衡。")
    try:
        st.image(get_image_path("roc_curve.png"), caption="ROC 曲线")
    except Exception:
        st.error("无法加载 ROC 曲线图片。")


# F. 预测结果
elif page == "预测结果":
    st.title("📋 最终预测结果")
    
    # 同样检查图片是否存在（如果用户直接进入此页面）
    image_dir = "images"
    pred_dist_path = os.path.join(image_dir, "prediction_distribution.png")
    
    if not os.path.exists(pred_dist_path):
         with st.spinner("检测到本地缺少静态图表，正在首次生成（可能需要几分钟）..."):
            try:
                from generate_plots import generate_all_plots
                generate_all_plots(output_dir=image_dir)
            except Exception as e:
                pass # 忽略错误，尝试使用远程图片
    
    st.subheader("1. 提交文件预览")
    st.write("这是根据测试集生成的预测结果：")
    st.dataframe(submission_df.head(10))
    
    st.subheader("2. 预测结果分布")
    try:
        st.image(get_image_path("prediction_distribution.png"), caption="测试集预测结果分布")
    except Exception:
         st.error("无法加载预测结果分布图片。")
    
    st.subheader("3. 下载结果")
    
    # 将 DataFrame 转换为 CSV 字节流
    csv = submission_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 下载 submission.csv",
        data=csv,
        file_name='submission_optimized_cv.csv',
        mime='text/csv',
    )
