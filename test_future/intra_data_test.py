from sklearn import linear_model

reg_LinearRegression = linear_model.LinearRegression()

reg_Ridge = linear_model.Ridge(alpha=.5)

reg_Lasso = linear_model.Lasso(alpha=0.1)

reg_MultiTaskLasso = linear_model.MultiTaskLasso()

reg_LassoLars = linear_model.LassoLars(alpha=.1)

reg_BayesianRidge = linear_model.BayesianRidge()

reg_LogisticRegression = linear_model.LogisticRegression()


class A:
    def fun(self):
        object.__setattr__(self, 's', 1234)
        setattr()
        getattr()
