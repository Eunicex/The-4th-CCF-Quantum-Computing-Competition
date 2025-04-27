import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def data_preprocess(data_csv):
    """
    1. 对数据的预处理，获取训练集和测试集数据的全部特征
    """


if __name__ == "__main__":
    data_csv = './code/train_data.csv'  # 替换为你的数据路径
    data_preprocess(data_csv)