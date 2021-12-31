# importation des bibliothèques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


data = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')

print(data.head(20))


#supression des valeurs manquantes
"""datacopy = df.copy()
df = df.dropna()
missing = df.isna().sum()/df.shape[0]
df.shape"""

#df = df.drop(0,axis=1)

#dfc = df2
df = data
df = df.drop('Loan_ID',axis = 1)
df


def split_datas(df, target='Loan_Status'):
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y


def separation_data(df):
    num_data = []
    cat_data = []

    for i, c in enumerate(df.dtypes):
        if c == object:
            cat_data.append(df.iloc[:, i])
        else:
            num_data.append(df.iloc[:, i])

    cat_data = pd.DataFrame(cat_data)
    num_data = pd.DataFrame(num_data)
    return cat_data.transpose(), num_data.transpose()


X, y_ = split_datas(df)
#target = pd.DataFrame(y).columns

categorical_data, numerical_data = separation_data(X)
categorical_features = list(categorical_data.columns)
numerical_features = list(numerical_data.columns)

# df[numerical_features]
# df[categorical_features]

# on encode la variable cible si celui ci n'est pas une variable numérique
from sklearn.preprocessing import LabelEncoder

encodage = LabelEncoder()
y = encodage.fit_transform(y_)
y = pd.DataFrame({"Loan_Status": y})

##en remplace  la variable target par la variable encoddé
df = pd.concat([X, y], axis=1)

df

#split data
from sklearn.model_selection import train_test_split
trainset,testset = train_test_split(df,test_size = 0.3,random_state=0)
X_train ,y_train = split_datas(trainset)

print(df.shape)
print(X_train.shape)
print(y_train.shape)

X_test ,y_test = split_datas(testset)

print(testset.shape)
print(X_test.shape)
print(y_test)

#--------------------------
# SELECTION DES VARIABLES PAR TYPES
# Spilt data
def separation_data(df):
    num_data = []
    cat_data = []

    for i, c in enumerate(df.dtypes):
        if c == object:
            cat_data.append(df.iloc[:, i])
        else:
            num_data.append(df.iloc[:, i])

    cat_data = pd.DataFrame(cat_data)
    num_data = pd.DataFrame(num_data)
    return cat_data.transpose(), num_data.transpose()


X, y = split_datas(df)
target = pd.DataFrame(y).columns

categorical_data, numerical_data = separation_data(X)

categorical_features = list(categorical_data.columns)
numerical_features = list(numerical_data.columns)
categorical_features


#------------------------------------------------------------------------
##pretraitement des données

from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import make_column_selector


class PreprocessingData:

    def __init__(self, dataset):
        self.dataset = dataset

    def preprocessingVariableNumerique(self, strategy_val_manquante='mean', methode_encodage=StandardScaler()):
        numerical_pipeline = make_pipeline(SimpleImputer(strategy=strategy_val_manquante), methode_encodage)
        return numerical_pipeline

    ## strategie de prétraitement des valeurs  catégorielles
    def preprocessingVariableCategoriel(self, strategy_val_manquante='most_frequent',methode_normalisation=OneHotEncoder()):
        numerical_pipeline = make_pipeline(SimpleImputer(strategy=strategy_val_manquante), methode_normalisation)
        return numerical_pipeline

    ##  appliquer le strategie sur les variables
    def preprocessingVariable(self, numerical_features, categorical_features):
        preprocessingVariableNumerique1 = self.preprocessingVariableNumerique(strategy_val_manquante='mean',
                                                                              methode_encodage=StandardScaler())

        preprocessingVariableCategoriel1 = self.preprocessingVariableCategoriel(strategy_val_manquante='most_frequent',
                                                                                methode_normalisation=OneHotEncoder())

        preprocessorVar = make_column_transformer((preprocessingVariableNumerique1, numerical_features),
                                                  (preprocessingVariableCategoriel1, categorical_features),
                                                  )

        return preprocessorVar

    ## Pipeline final de prétraitement des données (include polynomial features and select best Variables)
    def pipelinePreprocessing(self, methode_selectionVariable=SelectKBest(f_classif, k=10)):
        preprocessingVariable1 = self.preprocessingVariable(numerical_features, categorical_features)
        preprocessor = make_pipeline(preprocessingVariable1, PolynomialFeatures(2, include_bias=False),
                                     methode_selectionVariable)
        return preprocessor

    def transfom(self):
        resultat = self.pipelinePreprocessing(methode_selectionVariable=SelectKBest(f_classif, k=10))
        return resultat.fit_transform(X_train, y_train)


preprocessorObject = PreprocessingData(df)
preprocessor = preprocessorObject.pipelinePreprocessing(methode_selectionVariable=SelectKBest(f_classif, k=5))
# k = k.fit_transform(X_train,y_train)

# k.transform()

#-----------------------------------------------------
#LISTE DES MODELS DEFINIS POUR LE PROBLEME A RESOURDRE
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

RandomForest = make_pipeline(preprocessor,RandomForestClassifier(random_state=0))
AdaBoost = make_pipeline(preprocessor,AdaBoostClassifier(random_state=0))
SVM = make_pipeline(preprocessor,StandardScaler(),SVC(random_state=0))
KNN = make_pipeline(preprocessor,StandardScaler(),KNeighborsClassifier())
Logistic = make_pipeline(preprocessor,LogisticRegression(random_state=0))
MLP = make_pipeline(preprocessor,MLPClassifier())

hyper_params_svm = {
    'svc__gamma':[1e-3,1e-4],
    'svc__C':[1,10,100,1000],
    'pipeline__polynomialfeatures__degree':[2,3,4],
    'pipeline__selectkbest__k':range(4,100)
}
hyper_params_logistic = {
    'logisticregression__penalty':['l1','l2'],
    'logisticregression__C':[1,10,100,1000],
    'pipeline__polynomialfeatures__degree':[2,3,4],
    'pipeline__selectkbest__k':range(4,100)
}

#Dictionnaire des models
list_of_model = {
    "RandomForest":[RandomForest,{}],
    "AdaBoost":[AdaBoost,{}],
    "SVM":[SVM,hyper_params_svm],
    "KNN":[KNN,{}],
    "Logistic":[Logistic,hyper_params_logistic],
    "MLP":[MLP,{}]
}

#list_of_model["SVM"][1]


#---------------------------------------------------------------------
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score


class RMFramme:
    def __init__(self, listeModel):
        self.models = listeModel

    # PROCEDURE D'EVALUATION DES DIFFERRENTS MODELS

    # mesure = ['f1','precision','recall']
    def evaluerModel(self, model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        precision = accuracy_score(y_test, y_pred)
        return precision

    def rapport(self, model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # print(confusion_matrix(y_test,y_pred))
        # report = print(classification_report(y_test,y_pred))
        report = classification_report(y_test, y_pred)
        return report

    def matrixConfusion(self, model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # matrix = print(confusion_matrix(y_test,y_pred))
        matrix = confusion_matrix(y_test, y_pred)
        return matrix

    # Fonction qui prend en paramètre une liste de modèles prédéfinis et les variables test et entrainement
    # Puis renvoie un dictionnaire contenant tous les modèles ainsi que leurs scores respectifs

    def evalModels(self, X_train, y_train, X_test, y_test):
        resultat_dico_models = {}
        for name, model in self.models.items():
            precision = self.evaluerModel(model[0], X_train, y_train, X_test, y_test)
            resultat_dico_models[name] = precision
        return resultat_dico_models

    # Fonction qui permet de faire la comparaison entre les modèles entrainés et retourne celui ayant la meilleure
    # performance.
    def compareModels(self, dictionnaire):
        best = 0
        model = ''
        for item, value in dictionnaire.items():
            if best < value:
                best = value
                model = item
        return model, best

    # optimisation du modèle le plus performant
    def optimisationHyperParam(self, model, scoring='f1', cv=10):
        # print("----------:",self.models[model][0],self.models[model][1])
        grid = RandomizedSearchCV(self.models[model][0], self.models[model][1], scoring=scoring, cv=cv, n_iter=40)
        grid.fit(X_train, y_train)
        #print(grid.best_params_)
        y_pred = grid.predict(X_test)
        #print(classification_report(y_test, y_pred))
        return grid


#-----------------------------------------------------------------------------------------------------------------
###test

modelObject = RMFramme(list_of_model)

liste =modelObject.evalModels(X_train,y_train,X_test,y_test)

model = modelObject.compareModels(liste)

## list_of_model[model[0]][0] : accès au nom du modèle definit dans le dictionnaires des modèles
rapport = modelObject.rapport(list_of_model[model[0]][0],X_train,y_train,X_test,y_test)


print("model non optimiser",rapport)

"""
model_opt = modelObject.optimisationHyperParam(model[0])
print("model optimiser",modelObject.rapport(list_of_model[model_opt[0]][0],X_train,y_train,X_test,y_test))
pickle.dump(model_opt,open('modelx.pkl','wb'))"""

#print(opt)

