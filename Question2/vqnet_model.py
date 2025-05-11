from pyvqnet.dtype import *
from pyvqnet.tensor.tensor import QTensor
from pyvqnet.tensor import tensor
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
from pyvqnet.data.data import data_generator
from pyvqnet.nn.module import Module
from pyvqnet.nn import Linear, ReLu
from pyvqnet.optim.adam import Adam
from pyvqnet.nn.loss import CrossEntropyLoss
from data_preprocess import data_preprocess

def load_data(data_csv):
    """
    1. 读取数据并转换为张量
    """
    features, labels = data_preprocess(data_csv)
    
    # 确保数据格式正确
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)  # 分类标签应为整型
    
    # 转换为QTensor
    X_tensor = QTensor(X)
    y_tensor = QTensor(y)
    
    return X_tensor, y_tensor

class AirQualityNN(Module):
    """
    2. 在这里完成初始化神经网络模型的代码
    """
    def __init__(self):
        """
        在这里初始化神经网络结构
        """
        # 层的定义
        super().__init__()  
        self.fc1 = Linear(9, 8)    # 9×8 + 8 = 80
        self.fc2 = Linear(8, 8)    # 8×8 + 8 = 72
        self.fc3 = Linear(8, 4)    # 8×4 + 4 = 36
        self.relu = ReLu()

    def forward(self, x):
        """
        定义前向传播过程
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def model_train():
    """
    3. 在这里完成训练模型的代码
    """
    # 加载数据
    train_csv = './train_data.csv'
    X, y = load_data(train_csv)

    X = X.to_numpy()
    y = y.to_numpy()
    
    # 训练循环
    num_epoch = 300
    batch = 32
    lr = 0.05

    # 初始化模型和优化器
    model = AirQualityNN()
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()

    for epoch in range(num_epoch):
        if epoch % 50 == 0 and epoch != 0:
            optimizer.lr *= 0.5
            
        train_loader = data_generator(X, y, batch_size=batch)

        for xb, yb in train_loader:
            xb = QTensor(xb)
            yb = QTensor(yb)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(yb, outputs)
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {loss.item()}, {model_test(model)}')

    return model

def model_test(model):
    """
    4. 使用测试数据集验证模型
    """
    # 前向传播
    test_csv = './test_data.csv'
    X_test, y_test = load_data(test_csv)

    output = model(X_test)

    # 计算准确率和F1分数
    predicted = np.argmax(output.data, axis=1)  # 获取每个样本的预测类别
    accuracy = accuracy_score(y_test.to_numpy(), predicted)
    f1 = f1_score(y_test.to_numpy(), predicted, average='weighted')

    # 输出结果并保存到文件
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average F1 Score: {f1:.4f}")

    with open('vqnet_model_results.txt', 'w') as f:
        f.write(f"{accuracy:.4f} {f1:.4f}")


if __name__ == "__main__":
    model = model_train()

    model_test(model)