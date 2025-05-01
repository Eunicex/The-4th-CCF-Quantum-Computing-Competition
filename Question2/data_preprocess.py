import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def data_preprocess(data_csv):
    """
    1. 对数据的预处理，获取训练集和测试集数据的全部特征
    """
    # 加载数据
    df = pd.read_csv(data_csv)
    
    # 打印数据基本信息
    print("数据维度:", df.shape)
    print("列名:", df.columns.tolist())
    print("是否有缺失值:\n", df.isnull().sum())
    
    # 打印统计信息
    print("\n统计信息：\n", df.describe())

    # 特征与标签分离
    X = df.drop("Air Quality", axis=1)
    y = df["Air Quality"]

    # 标签编码
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y)
    print("\n标签映射：", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

    # 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X)

    return X_train, y_train


if __name__ == "__main__":
    data_csv = './train_data.csv'  
    data_preprocess(data_csv)