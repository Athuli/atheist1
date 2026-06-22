# 设备振动故障诊断代码
人工智能专业课程设计

## 文件说明
- main.py：项目主程序，数据处理、模型训练、预测输出
- src/feature_extract.py：时域特征提取模块，包含防无穷值、标签清洗逻辑

## 运行环境
Python 3.14
依赖：numpy pandas scipy scikit-learn xgboost

## 安装依赖
pip install numpy pandas scipy scikit-learn xgboost

## 运行命令
python main.py

## 流程
数据读取 → 时域特征提取 → 异常清洗 → 模型训练(KNN/SVM/RF/XGBoost) → 生成预测文件
