# 心脏病预测分析系统

本项目旨在使用 XGBoost 机器学习模型预测心脏病风险，并提供一个基于 Streamlit 的交互式 Web 应用来展示数据分析、模型训练过程及预测结果。

## 项目结构

- `app.py`: Streamlit 应用程序主入口，包含数据概览、EDA、特征工程、模型可视化和预测结果展示。
- `train_predict.py`: 初始的 XGBoost 训练和预测脚本。
- `optimize_model.py`: 使用 RandomizedSearchCV 进行超参数优化的脚本。
- `train_optimized_direct.py`: 使用优化后的参数直接训练模型并生成最终提交文件的脚本。
- `requirements.txt`: 项目依赖库列表。
- `sample_submission.csv`: 示例提交文件格式。

## 数据集

本项目使用心脏病数据集，包含以下文件（由于文件较大，未直接包含在仓库中，请自行下载并放置在项目根目录）：

- `train.csv`: 训练数据集，包含特征和目标变量 `Heart Disease`。
- `test.csv`: 测试数据集，用于生成预测结果。
- `sample_submission.csv`: 提交文件示例格式。

## 依赖安装

建议使用 Python 3.8+ 环境。安装依赖：

```bash
pip install -r requirements.txt
```

## 运行项目

### 1. 训练模型

如果你想重新训练模型并生成预测文件：

```bash
python train_optimized_direct.py
```

这将生成 `submission_optimized_cv.csv` 文件。

### 2. 启动 Web 应用

运行 Streamlit 应用以查看交互式分析和可视化：

```bash
streamlit run app.py
```

应用将在浏览器中打开，通常地址为 `http://localhost:8501`。

## 模型优化

本项目使用了 XGBoost 模型，并进行了以下优化：
- 超参数调优（RandomizedSearchCV）
- 5折交叉验证（StratifiedKFold）
- 特征工程（标签编码、空值处理）
- 模型集成（Bagging）

最终模型在验证集上达到了约 88.9% 的准确率。
