import numpy as np
import pandas as pd
import pylab
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes._axes import _log as matplotlib_axes_logger

matplotlib_axes_logger.setLevel('ERROR')
from sklearn.datasets import load_iris

data = load_iris()  # 得到数据特征
iris_target = data.target  # 得到数据对应的标签
iris_features = pd.DataFrame(data=data.data, columns=data.feature_names)  # 利用Pandas转化为DataFrame格式

iris_features.info()

pd.Series(iris_target).value_counts()

#### 数据可视化 ####
# 散点图
# 合并标签和特征信息
iris_all = iris_features.copy()  ##进行浅拷贝，防止对于原始数据的修改
iris_all['target'] = iris_target
sns.pairplot(data=iris_all, diag_kind='hist', hue='target')  # 特征与标签组合的散点可视化
plt.show()

# 箱型图
for col in iris_features.columns:
    sns.boxplot(x='target', y=col, saturation=0.5, palette='pastel', data=iris_all)
    plt.title(col)
plt.show()

# 选取其前三个特征绘制三维散点图
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
iris_all_class0 = iris_all[iris_all['target'] == 0].values
iris_all_class1 = iris_all[iris_all['target'] == 1].values
iris_all_class2 = iris_all[iris_all['target'] == 2].values
# 'setosa'(0), 'versicolor'(1), 'virginica'(2)
ax.scatter(iris_all_class0[:, 0], iris_all_class0[:, 1], iris_all_class0[:, 2], label='setosa')
ax.scatter(iris_all_class1[:, 0], iris_all_class1[:, 1], iris_all_class1[:, 2], label='versicolor')
ax.scatter(iris_all_class2[:, 0], iris_all_class2[:, 1], iris_all_class2[:, 2], label='virginica')
plt.legend()
plt.show()

iris = datasets.load_iris()
a, b = 0, 2
X_reduced = iris.data[:, :4]
X = X_reduced[:, [a, b]]  # 二维可视化，即只取两个属性
y = iris.target  # 由上述程序结果可知取值为0,1,2
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5  # x值的最小值和最大值分别是第一列最小值和最大值-5和+5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5  # y值的最小值和最大值分别是第二列最小值和最大值-5和+5
plt.figure(2, figsize=(8, 6))
plt.clf
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='w')  # 绘制散点图，c即color，cmap是将y不同的值画出不同颜色，edgecolor为白色
plt.xlabel(iris.feature_names[a])
plt.ylabel(iris.feature_names[b])
plt.xlim(x_min, x_max)  # x轴的作图范围
plt.ylim(y_min, y_max)  # x轴的作图范围
plt.xticks(())  # x轴的刻度内容的范围
plt.yticks(())  # y轴的刻度内容的范围

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)  # 调用训练集训练
X_train_std = sc.transform(X_train)
X_test_std: object = sc.transform(X_test)


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim = (xx2.min(), xx2.max())
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='black', alpha=0.8, linewidths=1, marker='o', s=10, label='test set')


# 调整2*2图像大小比例
plt.figure(2, figsize=(10, 8))

pylab.subplot(2, 2, 1)  # 子图像为2*2的第一个
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
svm = SVC(kernel='linear', random_state=0, C=1.0)  # 调用SVM核函数,’linear’核函数，以及两个超参数
svm.fit(X_train_std, y_train)  # 训练
plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
plt.ylabel(iris.feature_names[b])
plt.title('Linear')

pylab.subplot(2, 2, 2)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
svm = SVC(kernel='poly', random_state=0, degree=2, gamma=0.3, C=100)
# 调用SVM核函数,’poly’以及四个参数，多项式核函数专属的超参数d
svm.fit(X_train_std, y_train)  # 训练
plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
plt.title('poly')

pylab.subplot(2, 2, 3)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
svm = SVC(kernel='rbf', random_state=0, gamma=0.9, C=1.5)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
plt.ylabel(iris.feature_names[b])
plt.xlabel(iris.feature_names[a])
plt.title('rbf')

pylab.subplot(2, 2, 4)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
svm = SVC(kernel='sigmoid', random_state=0, gamma=0.3, C=50)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
plt.xlabel(iris.feature_names[a])
plt.title('sigmoid')

plt.show()
