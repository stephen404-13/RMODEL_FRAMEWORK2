##===================CLASSE D'ACQUISION DES DONNEES====================#

import os.path
##===================PRETRAITEMENT DES DONNEES
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import make_column_selector
from sklearn.model_selection import train_test_split
import pandas as pd





class Acquisision:

    def __init__(self,dataFrame,target):
        self.dataFrame = dataFrame
        self.target = target
        self.X = ''
        self.columns = ''
        self.values = ''

    def split_datas(self,df): #update
        target = self.target
        X = df.drop(target, axis=1)
        self.X = X
        y = df[target]
        #self.target = y
        return X, y

    def separation_data(self):
        num_data = []
        cat_data = []
        donneeVariable = self.apply_split(self.dataFrame)[0]
        for i, c in enumerate(donneeVariable.dtypes):
            if c == object:
                cat_data.append(donneeVariable.iloc[:, i])
            else:
                num_data.append(donneeVariable.iloc[:, i])
        cat_data = pd.DataFrame(cat_data)
        num_data = pd.DataFrame(num_data)
        return cat_data.transpose(), num_data.transpose()

    def apply_split(self,df):
        X, y_ = self.split_datas(df)
        #target = pd.DataFrame(y).columns
        return X,y_

    def apply_separation_data(self,df):
        X = self.split_datas(df)[0]
        categorical_data, numerical_data = self.separation_data()
        categorical_features = list(categorical_data.columns)
        numerical_features = list(numerical_data.columns)
        return numerical_features,categorical_features

    def encodage_label(self):
        target_name = self.target
        liste = self.apply_split(self.dataFrame)
        y_ = liste[1]
        X = liste[0]
        encodage = LabelEncoder()
        y = encodage.fit_transform(y_)
        y = pd.DataFrame({target_name: y})
        #en remplace  la variable target par la variable encoddé
        df = pd.concat([X, y], axis=1)
        self.dataFrame = df
        return df

    def train_test_set(self,test_size=0.3):
        df  = self.dataFrame
        trainset, testset = train_test_split(self.dataFrame, test_size=test_size, random_state=0)
        X_train, y_train = self.split_datas(trainset)
        X_test, y_test = self.split_datas(testset)
        return list([X_train, y_train]),list([X_test, y_test])

    def matrix_corelation(self):
        target_name = self.target
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns_plot = sns.pairplot(self.dataFrame, hue=target_name, height=2.5)
        plt.savefig('output.png')