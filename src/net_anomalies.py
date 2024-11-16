# 导入第三方依赖库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay
from sklearn.metrics import roc_curve, auc
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import numpy as np
from collections import Counter
import time
import seaborn as sns

df_full = pd.read_parquet('./dataSet/simargl-2021-combined.parquet')  # 读取数据集
df = df_full.sample(frac=0.15)  # 因GPU配置差，缩小原始数据集大小 （原数据集大小为1.34GB ， 包括40263811行数据）

# 探索性数据分析
print('------------------------------- 探索性数据分析 ------------------------------------------')
# 保证输出数据完整
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# 1. 数据概览
print("数据集概览：")
print("数据集形状 (行, 列):", df.shape)
print("\n数据信息：")
print(df.info())

# 2. 检查缺失值
print("\n缺失值概览：")
print(df.isnull().sum())

# 3. 统计描述
# print("\n数值特征的统计信息：")
# print(df.describe())

# 4. 类别特征分布
print("\n标签分布：")
print(df['LABEL'].value_counts())
# 可视化标签分布
plt.figure(figsize=(20, 12))
# 标签分布
sns.countplot(x='LABEL', data=df)
# 旋转 x 轴标签
plt.xticks(rotation=15, ha='right')
plt.tight_layout()  # 自动调整布局以适合所有内容
plt.savefig('./result/label_distribution.png')
plt.show()

# 5. 特征分布
# 数值特征直方图展示
numerical_features = [
    'DST_TO_SRC_SECOND_BYTES', 'FLOW_ACTIVE_TIMEOUT', 'FLOW_DURATION_MICROSECONDS',
    'FLOW_DURATION_MILLISECONDS', 'FRAME_LENGTH', 'IN_BYTES', 'IN_PKTS',
    'OUT_BYTES', 'OUT_PKTS', 'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT',
    'TCP_WIN_MIN_IN', 'TCP_WIN_MIN_OUT'
]
# 使用 hist 方法直接指定 figsize 和 layout
df[numerical_features].hist(bins=15, figsize=(15, 10), layout=(4, 4))
# 旋转 x 轴标签
plt.xticks(rotation=15, ha='right')
# 添加总标题
plt.suptitle("Numerical Features Histogram", fontsize=16)
plt.tight_layout()  # 自动调整布局以适合所有内容
# 保存图形
plt.savefig('./result/numerical_features_histogram.png')
# 显示图形
plt.show()

# 6. 协议和端口分布
# 源端口和目标端口的分布
plt.figure(figsize=(10, 8))
# 源端口分布
plt.subplot(1, 2, 1)
sns.histplot(df['L4_SRC_PORT'], bins=30, kde=True)
plt.title("L4_SRC_PORT")

# 目标端口分布
plt.subplot(1, 2, 2)
sns.histplot(df['L4_DST_PORT'], bins=30, kde=True)
plt.title("L4_DST_PORT")
plt.savefig('./result/port_distributions.png')
plt.tight_layout()  # 自动调整布局以适合所有内容
plt.show()

# 7. 相关性分析
# plt.figure(figsize=(12, 8))
# correlation_matrix = df[numerical_features].corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("数值特征之间的相关性")
# plt.savefig('./result/correlation_matrix.png')
# plt.show()

print('------------------------------- 探索性数据分析结束 ------------------------------------------')

df.dropna(inplace=True)  # 删除缺失值所在的行

# 将文本标签编码为数值
label_encoder = LabelEncoder()
df['LABEL'] = label_encoder.fit_transform(df['LABEL'])  # 将类别转换为数值

# 特征选择：基于特征重要性图表
features = [
    'TCP_WIN_MSS_IN', 'IN_BYTES', 'OUT_BYTES',
    'FLOW_DURATION_MILLISECONDS', 'L4_DST_PORT',
    'TCP_WIN_MIN_IN', 'TCP_FLAGS', 'TCP_WIN_MAX_IN',
    'LAST_SWITCHED'
]
X = df[features]
y = df['LABEL']

# 再次检查 X 和 y 的形状
print("X shape:", X.shape)
print("y shape:", y.shape)

# 将数值特征转换为浮点数
for col in features:
    # 转换数值特征
    X[col] = pd.to_numeric(X[col].astype(str).str.replace(',', ''), errors='coerce')

# 再次检查 X 和 y 的形状
print("X shape:", X.shape)
print("y shape:", y.shape)

# 归一化选定的特征
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# # # 打印原始类别分布
# print("Original dataset shape:", Counter(y))

# # 处理类别不平衡问题
# smote = SMOTE(k_neighbors=1)
# X_resampled, y_resampled = smote.fit_resample(X, y)

# # 打印重采样后的类别分布
# print("Resampled dataset shape:", Counter(y_resampled))

# 按 6:4 比例划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 训练并评估模型，确保正确的训练时间报告
results = {}


# 参数注释 ——————
# y_true: 真实标签，通常是一个一维数组或列表，表示实际的类别标签。
# y_pred: 预测标签，通常是一个一维数组或列表，表示模型预测的类别标签。
# model_name: 模型的名称，默认值为 "Model"，用于在输出中标识模型。
# y_pred_proba: 预测概率，通常是一个二维数组，表示每个样本属于每个类别的概率。如果提供，可以用于计算 ROC AUC 和绘制 ROC 曲线。
def evaluate_model(y_true, y_pred, training_time, model_name="Model", y_pred_proba=None):
    print('-------------------------------------------------------------------------')
    print(f"Evaluation results for {model_name}:")
    accuracy_score_res = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy_score_res)
    # 宏平均（Macro Average）：计算每个类别的精确率，然后取平均值
    precision_score_res = precision_score(y_true, y_pred, average='macro', zero_division=0)
    print("Precision:", precision_score_res)  # 使用宏平均（macro average）进行汇总。在某个类别没有预测样本时，将该类别的精确度设为 0。
    recall_score_res = recall_score(y_true, y_pred, average='macro', zero_division=0)
    print("Recall:", recall_score_res)
    f1_score_res = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print("F1 Score:", f1_score_res)
    print('-------------------------------------------------------------------------')

    if y_pred_proba is not None:
        roc_auc_score_res = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
        print("ROC AUC:", roc_auc_score_res)

        results[model_name] = {
            "Accuracy": accuracy_score_res,
            "Precision": precision_score_res,
            "Recall": recall_score_res,
            "F1 Score": f1_score_res,
            "AUC": roc_auc_score_res,
            "Training time": training_time
        }

        # 绘制多类的 ROC 曲线
        # 多分类问题中的 One-vs-Rest (OvR) 方法
        # 使用 roc_auc_score 函数，并设置参数 multi_class='ovr'。这表示每个类别都分别被视为正类，其他所有类别被视为负类，然后计算每个类别的二分类 AUC 值。
        # 最终的 AUC 值是这些二分类 AUC 值的平均值。
        fpr = {}
        tpr = {}
        roc_auc = {}

        for i in range(len(label_encoder.classes_)):
            fpr[i], tpr[i], _ = roc_curve(y_true, y_pred_proba[:, i], pos_label=i)
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.figure(figsize=(10, 8))
            plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {label_encoder.classes_[i]} (area = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - {model_name}')
        plt.legend(loc="lower right")
        # 保存 AUC 图
        auc_image_path = f'./result/{model_name}_roc_auc.png'
        plt.tight_layout()  # 自动调整布局以适合所有内容
        plt.savefig(auc_image_path)
        plt.show()


# 逻辑回归
lr_model = LogisticRegression()
start_time_lr = time.time()
lr_model.fit(X_train, y_train)
end_time_lr = time.time()
training_time_lr = end_time_lr - start_time_lr
y_pred_lr = lr_model.predict(X_test)
y_pred_lr_proba = lr_model.predict_proba(X_test)  # 获取预测概率
evaluate_model(y_test, y_pred_lr, training_time_lr, "Logistic Regression", y_pred_proba=y_pred_lr_proba)

# 朴素贝叶斯
# 创建一个 GaussianNB 模型实例
nb_model = GaussianNB()
start_time_nb = time.time()
nb_model.fit(X_train, y_train)  # 使用训练数据拟合模型
end_time_nb = time.time()
training_time_nb = end_time_nb - start_time_nb
# 使用测试数据进行预测
y_pred_nb = nb_model.predict(X_test)
# 获取预测的概率
# X_test: 测试数据的特征矩阵，通常是二维数组
# 返回值 y_pred_nb_proba 是一个二维数组，形状为 (n_samples, n_classes)
# 其中 n_classes 是类别的数量，每一行对应一个测试样本，每一列对应一个类别的预测概率
y_pred_nb_proba = nb_model.predict_proba(X_test)
# 调用 evaluate_model 函数评估模型性能
evaluate_model(y_test, y_pred_nb, training_time_nb, "Naive Bayes", y_pred_proba=y_pred_nb_proba)

# 随机森林
# 创建一个 RandomForestClassifier 模型实例
# n_estimators: 决策树的数量，默认为 100
# random_state: 用于初始化随机数生成器的种子，设置为固定值以便复现结果
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
start_time_rf = time.time()
rf_model.fit(X_train, y_train)
end_time_rf = time.time()
training_time_rf = end_time_rf - start_time_rf
y_pred_rf = rf_model.predict(X_test)
y_pred_rf_proba = rf_model.predict_proba(X_test)  # 获取预测概率
evaluate_model(y_test, y_pred_rf, training_time_rf, "Random Forest", y_pred_proba=y_pred_rf_proba)

# RNN

# 将训练和测试标签转换为独热编码
# to_categorical 将整数标签转换为二进制（独热）矩阵
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

# 重塑训练和测试数据，使其符合 RNN 输入要求
# RNN 需要输入数据的形状为 (samples, timesteps, features)
X_train_rnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_rnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 创建 RNN 模型
rnn_model = Sequential([
    # 添加一个 SimpleRNN 层
    # 50: 输出维度（即 RNN 单元的数量）
    # input_shape: 输入数据的形状，(timesteps, features)
    # activation: 激活函数，这里使用 'relu'
    SimpleRNN(50, input_shape=(X_train.shape[1], 1), activation='relu'),
    # 添加一个全连接层
    # len(label_encoder.classes_): 输出维度，即类别的数量
    # activation: 激活函数，这里使用 'softmax'，适用于多分类任务
    Dense(len(label_encoder.classes_), activation='softmax')
])
start_time_rnn = time.time()
# 编译模型
# optimizer: 优化器，这里使用 'adam'
# loss: 损失函数，这里使用 'categorical_crossentropy'，适用于多分类任务
# metrics: 评估指标，这里使用 'accuracy'
rnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# 训练模型
# X_train_rnn: 训练数据的特征矩阵
# y_train_categorical: 训练数据的标签
# epochs: 训练的轮数
# batch_size: 每个批次的样本数量
# validation_data: 验证数据，用于评估模型在每个 epoch 结束时的性能
history = rnn_model.fit(X_train_rnn, y_train_categorical, epochs=5, batch_size=64,
                        validation_data=(X_test_rnn, y_test_categorical))
# 记录训练结束时间
end_time_rnn = time.time()
# 计算训练所花费的时间
training_time_rnn = end_time_rnn - start_time_rnn

# 使用测试数据进行预测
# X_test_rnn: 测试数据的特征矩阵
# 返回值 y_pred_rnn_proba 是一个二维数组，形状为 (samples, num_classes)，每行对应一个测试样本，每列对应一个类别的预测概率
y_pred_rnn_proba = rnn_model.predict(X_test_rnn)

# 将预测概率转换为预测标签，np.argmax 沿指定轴返回最大值的索引，axis=1 表示沿行方向
y_pred_rnn = np.argmax(y_pred_rnn_proba, axis=1)

# 调用 evaluate_model 函数评估模型性能
# y_test: 测试数据的真实标签，通常是一维数组，形状为 (samples,)
# y_pred_rnn: 模型预测的标签，通常是一维数组，形状为 (samples,)
# "RNN": 模型的名称，用于标识
# y_pred_proba: 模型预测的概率，通常是一个二维数组，形状为 (samples, num_classes)
evaluate_model(y_test, y_pred_rnn, training_time_rnn, "RNN", y_pred_proba=y_pred_rnn_proba)

# 将结果转换为 DataFrame 进行显示
result_df = pd.DataFrame(results).T
print(result_df)
# 保存结果
result_df.to_csv('./result/model_results.csv', index=True)

# Define the plot
plt.figure(figsize=(12, 8))

# Plot each metric
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
colors = ['blue', 'orange', 'green', 'red']
width = 0.2  # width of the bars

# Create bar positions for each metric, with offsets for each model
positions = range(len(result_df.index))

for i, metric in enumerate(metrics):
    plt.bar([p + i * width for p in positions], result_df[metric], width=width, label=metric, color=colors[i])

# Add labels and title
plt.xlabel("Model")
plt.ylabel("Score")
plt.title("Model Comparison on Evaluation Metrics")
plt.xticks([p + 1.5 * width for p in positions], result_df.index)
plt.ylim(0, 1)

# Add legend
plt.legend(loc="upper right")
plt.tight_layout()  # 自动调整布局以适合所有内容

# Show plot
plt.show()

# 保存图像
plt.savefig('./result/model_comparison.png')
