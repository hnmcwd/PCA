import numpy as np

def get_cov_matrix(data,transpose=False):
    '''
    求原始数据的协方差矩阵
    :param data: 原始数据，可以为list，arra，matrix
    :param transpose: 每一行为一个指标变量，每一列为一个观测样本，根据输入的原始数据判断是否需要转置
    :return: 各指标的协方差矩阵
    '''
    dataM = np.mat(data)
    if transpose == True:
        dataM = dataM.T
    mean = np.mean(dataM,axis=1)  # 对各行指标求平均
    dataM = dataM - mean  # 原始矩阵每行(也即每个样本)减去均值向量得到标准化后的矩阵
    covM = np.cov(dataM)  # 指标变量为行向量，其rowvar为默认值true
    return covM

def get_correlation_marix(data,transpose=False):
    '''
    求原始数据的相关系数矩阵
    :param data: 原始数据，可以为list，arra，matrix
    :param transpose: 每一行为一个指标变量，每一列为一个观测样本，根据输入的原始数据判断是否需要转置
    :return: 各指标的相关系数矩阵
    '''
    dataM = np.mat(data)
    if transpose == True:
        dataM = dataM.T
    correlation = np.corrcoef(dataM)
    return correlation

def calculate_eigen(cMatrix):
    '''
    计算特征值和特征向量
    :param cMatrix: 进行计算的协方差矩阵，或相关系数矩阵
    :return: 按特征值大小降序排列的特征值及对应特征向量
    '''
    eigenvalue, eigenvector = np.linalg.eig(cMatrix)   # 计算特征值和特征向量
    conn = [(eigenvalue[i], eigenvector[:, i]) for i in range(len(eigenvalue))]
    conn.sort(reverse=True)  # 降序排列
    eigenvalue = np.array([conn[i][0] for i in range(len(eigenvalue))])
    eigenvector = np.array([conn[i][1] for i in range(len(eigenvalue))])
    return eigenvalue,eigenvector

def reduce_dimension(cMatrix,component=3):
    '''
    求解主成分
    :param cMatrix: 进行计算的协方差矩阵，或相关系数矩阵
    :param component: 该参数小于1时，按主成分贡献率降维；大于等于1时，按主成分个数降维
    :return:arg1,主特征值，arg2，主成分贡献率，arg3，特征向量矩阵，也即转换矩阵
    '''
    component = np.abs(component)  # 对输入的component进行预处理，取绝对值，
    if component > len(cMatrix):
        component = len(cMatrix)  # 超过指标变量数按最大值处理
    value,vector = calculate_eigen(cMatrix)  # 调用函数计算特征值和特征向量
    pValue = []  # 主特征值
    pVector = []  # 主特征向量
    if component < 1:  # 按贡献率降维
        for i in range(len(value)):
            pValue.append(value[i])
            pVector.append(vector[i])
            ratio = np.divide(pValue, np.sum(value))  # 特征值贡献率
            if np.sum(ratio) >= component:
                break
        pVector = np.array(pVector)
    else:  # 按主成分个数降维
        pValue = np.array([value[i] for i in range(component)])
        pVector = np.array([vector[i] for i in range(component)])
        ratio = np.divide(pValue,np.sum(value))
    return pValue,ratio,pVector

def principal_componen(data,transpose=True,matvar=True,component=3):
    '''
    对原始数据进行主成分分析，分别返回主成分贡献率和主成分矩阵
    :param data: 原始数据
    :param transpose: 每一行为一个指标变量，每一列为一个观测样本，根据输入的原始数据判断是否需要转置
    :param matvar: 为true表示计算协方差矩阵，为false表示计算相关系数矩阵
    :param component: 该参数小于1时，按主成分贡献率降维；大于等于1时，按主成分个数降维
    :return: arg1，主成分贡献率，arg2，主成分矩阵
    '''
    dataM = np.mat(data)
    if transpose == True:
        dataM = dataM.T
    if matvar:
        cMatrix = get_cov_matrix(dataM)
    else:
        cMatrix = get_correlation_marix(dataM)
    pValue, ratio, pVector = reduce_dimension(cMatrix,component)
    tMatrix = np.mat(pVector)
    pMatrix = np.dot(tMatrix,dataM)
    return ratio,pMatrix
