# ===============================================================================
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import tree
from sklearn.pipeline import make_pipeline
import pandas as pd


hyper_params_svm = {
    'svc__gamma': [1e-3, 1e-4],
    'svc__C': [1, 10, 100, 1000],
    'pipeline__polynomialfeatures__degree': [2, 3, 4],
    'pipeline__selectkbest__k': range(4, 100)
}
hyper_params_logistic = {
    'logisticregression__penalty': ['l1', 'l2'],
    'logisticregression__C': [1, 10, 100, 1000],
    'pipeline__polynomialfeatures__degree': [2, 3, 4],
    'pipeline__selectkbest__k': range(4, 100)
}
hyper_params_knn = {}

hyper_params_radomForest = {}

hyper_params_MLP = {}

hyper_params_AdaBoost = {}

hyper_params_lda = {
    'lda__solver' : ['svd', 'lsqr', 'eigen'],
    'lda__shrinkage' : ['auto','float'] ,
    'pipeline__polynomialfeatures__degree': [2, 3, 4],
    'pipeline__selectkbest__k': range(4, 100)
}

hyper_params_tree = {
    'pipeline__polynomialfeatures__degree': [2, 3, 4],
    'pipeline__selectkbest__k': range(4, 100)
}





##===================LIEN DU FICHIER DE TEST====================#
source2 = r'train_u6lujuX_CVtuZ9i.csv'
source = r'chunk.csv'

##==================New dataset customer ========================
new_dataset = r'new_customers.csv'

df_new = pd.read_csv(new_dataset)
df_new_scustomer = df_new.drop(['Churn'], axis=1)
