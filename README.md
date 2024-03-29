# 信用卡违约率分析 credit_default
#### 针对台湾某银行的信用卡数据，构建了一个分析信用卡违约率的分类器。采用Random Forest算法，信用卡违约率识别率在80%左右。https://zhuanlan.zhihu.com/p/67420113

我们在数据挖掘过程中，经常需要选择使用何种分类器或分类算法，如SVM、决策树、随机森林等，以及优化相应分类器参数以得到更好的分类准确率。

随机森林Random Forest，简称RF。它是一个包含多个决策树的分类器，每一个子分类器都是一个CART分类回归树。所以随机森林既可以做回归，也可以做分类。

当RF用来做分类时，输出的结果是所有子分类器的分类结果中最多的那个，当RF用做回归时，输出结果是所有CART树的回归结果的平均值。

我们使用sklearn中的RandomForestClassfier()方法来构造随机森林模型。该函数输入的常用参数有：

n_estimators表示随机森林的决策树的个数，默认值为0。
criterion表示决策树分裂的标准，默认是基尼系数（CART算法），也可以选择使用entropy（ID3算法）。
max_depth表示决策树的最大深度，默认值为None，也就是不限制决策树的深度。也可以设置一个整数，表示限制决策树的最大深度。
n_jobs表示拟合和预测的时候CPU的核数，默认是1，可以设置其它整数，设置为-1表示CPU的核数。

构造完随机森林后，就可以使用fit函数拟合，使用predict函数预测。

我们可以使用Python中的参数自动搜索模块GridSearchCV工具对模型参数进行调优，它的构造方法是GridSearchCV(estimator,param_grid,cv=None,scoring=None)。
其中，estimator代表采用的分类器，如随机森林、决策树、SVM、KNN等；param_grid表示想要优化的参数以及取值，输入的是字典或者列表的形式；
cv表示交叉验证的折数，默认为None，代表使用三折交叉验证，也可是使用其它整数表示折数；
scoring表示准确度的评价标准，默认为None，也就是需要使用score函数，也可以设置具体的评价标准，比如accuray、f1等。

构造完自动搜索模块之后，就可以使用fit函数拟合训练模型，使用predict函数预测。预测是采用得到的最优参数。
