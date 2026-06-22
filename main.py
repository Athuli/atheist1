from src.feature_extract import extract_features
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
# 屏蔽两类无关警告
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 绝对路径读取数据
train_raw = pd.read_csv(r"C:\Users\Atheist\Desktop\equipment_subhealth\data\train.csv")
test_raw = pd.read_csv(r"C:\Users\Atheist\Desktop\equipment_subhealth\data\test.csv")

# 快速测试：小样本验证全流程，跑完无报错再注释head跑20万全量
#train_raw = train_raw.head(2000)
#test_raw = test_raw.head(200)

print(f"训练集总样本数：{len(train_raw)}")
print(f"测试集总样本数：{len(test_raw)}")

# 训练集特征提取
train_feature_list = []
print("开始提取训练集特征...")
for idx, row in train_raw.iterrows():
    sig = row.iloc[[1,2]].values
    feats = extract_features(sig)
    feats["label"] = row.iloc[3]
    train_feature_list.append(feats)
    if idx % 200 == 0:
        print(f"已处理训练样本：{idx}/{len(train_raw)}")
train_df = pd.DataFrame(train_feature_list)
print(f"训练特征集构建完成，特征维度：{train_df.shape}")

# 测试集特征提取
test_feature_list = []
test_id = test_raw.iloc[:, 0].tolist()
print("开始提取测试集特征...")
for idx, row in test_raw.iterrows():
    sig = row.iloc[[1,2]].values
    feats = extract_features(sig)
    test_feature_list.append(feats)
    if idx % 200 == 0:
        print(f"已处理测试样本：{idx}/{len(test_raw)}")
test_df = pd.DataFrame(test_feature_list)
print(f"测试特征集构建完成，特征维度：{test_df.shape}")

# 划分训练验证集
X = train_df.drop("label", axis=1)
y = train_df["label"]

# 1. 清洗特征X：清除无穷大inf、负无穷、空值
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

# 2. 清洗标签y，删除标签为空的整行，解决y含NaN报错
df_all = pd.concat([X, y], axis=1)
df_all = df_all.dropna(subset=["label"])
X = df_all.drop("label", axis=1)
y = df_all["label"]

# 核心修复1：标签转整数
y = y.astype(int)

# 核心修复2：把负数标签映射为0起始连续数字，适配XGBoost
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
print("数据集划分完成")

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 同步清洗测试集特征
test_df = test_df.replace([np.inf, -np.inf], np.nan)
test_df = test_df.fillna(0)
test_scaled = scaler.transform(test_df)
print("数据标准化完成")

# 多模型训练评估
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf", random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=150, max_depth=6, random_state=42)
}

print("===== 各模型验证集指标 =====")
for name, model in models.items():
    print(f"正在训练 {name} ...")
    if name in ["KNN", "SVM"]:
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_val_scaled)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
    acc = accuracy_score(y_val, pred)
    print(f"{name} 验证集准确率：{acc:.4f}")
    print(classification_report(y_val, pred, zero_division=0))

# 生成提交文件：XGB预测后映射回原始标签
print("使用XGBoost预测测试集，生成提交文件...")
best_model = models["XGBoost"]
test_pred_encoded = best_model.predict(test_df)
# 解码还原原本的-2/-1/0/1/2标签
test_pred = le.inverse_transform(test_pred_encoded)
submit = pd.DataFrame({"id": test_id, "label": test_pred})
submit.to_csv(r"C:\Users\Atheist\Desktop\equipment_subhealth\output\submit.csv", index=False)
print("✅ 全部执行完毕！提交文件已生成至 output/submit.csv，无任何报错！")