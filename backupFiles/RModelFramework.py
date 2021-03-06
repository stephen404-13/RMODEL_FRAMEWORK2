import pandas as pd
import pickle
##===================BIBLIO CLASSE ABSTRAITE
from abc import ABC, abstractmethod

##===================BIBLIO ACQUISION DES DONNEES
import os.path

##===================IMPORTATION DES BIBLIOTHEQUES DE MACHINE LEARNIND====================#


##===================PRETRAITEMENT DES DONNEES
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import make_column_selector
from sklearn.model_selection import train_test_split


##===================BIBLIOTHEQUE POUR POUR LA MESURE DE PERFORMENCE DU MODEL L'OPTIMISATION DES HYPERPARAMETRES
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, confusion_matrix, classification_report
#from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score


##=================== Visualisation
import  seaborn as sns


##=================== CLASS D'IMPORTATION DES DONNEES ====================#
class DataImport:
    EXTENSION = ['.csv', '.xlsx']
    def __init__(self,filename):
        self.extension = None
        self.filename = filename
        self.file = None
        self.df = None

    def upload_file(self):
        #ploaded = files.upload()
        #self.filename =  source
        self.extension = os.path.splitext(self.filename)[1].lower()
        #print(self.extension )
        if self.check_extension_file():
            print('Extension {} n\'est pas prise en charge!!!'.format(self.extension))

    def check_extension_file(self):
        return self.extension not in DataImport.EXTENSION

    def load_data(self,
                  additional_data={
                      'separator': ';',
                      'sheet_name': None
                  }):
        if self.extension == 'csv':
            self.df = pd.read_csv(self.filename)
        elif self.extension == 'xlsx':
            self.df = pd.read_xlsx(self.filename, sheet_name=additional_data['sheet_name'])
        else:
            self.df = pd.read_csv(self.filename)

    def display_data(self, lines=20):
        if self.df is None:
            self.load_data()
        print(self.df.head(lines))

    def delete_data_entry(self, field, axis=1):
        return self.df.drop(field, axis=axis)

    def save_data(self):
        pass


    def chargement(self):
        self.upload_file()
        self.load_data()
        return self.df



##===================CLASSE D'ACQUISION DES DONNEES====================#
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
        #en remplace  la variable target par la variable encodd??
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




##=================== CLASSE  DE PRETRAITEMENT DES DONNEES ====================#
class PreprocessingData(Acquisision):

    def __init__(self,dataset,target):
        super().__init__(dataset,target)
        self.numerical_features = self.apply_separation_data(self.dataFrame)[0]
        self.categorical_features = self.apply_separation_data(self.dataFrame)[1]

    def preprocessingVariableNumerique(self, strategy_val_manquante='mean', methode_normalisation=StandardScaler()):
        numerical_pipeline = make_pipeline(SimpleImputer(strategy=strategy_val_manquante), methode_normalisation)
        return numerical_pipeline

    ## strategie de pr??traitement des valeurs  cat??gorielles
    def preprocessingVariableCategoriel(self, strategy_val_manquante='most_frequent',methode_encodage=OneHotEncoder()):
        categoriel_pipeline = make_pipeline(SimpleImputer(strategy=strategy_val_manquante), methode_encodage)
        return categoriel_pipeline

    ##  appliquer le strategie sur les variables
    def preprocessingVariable(self, numerical_features, categorical_features):
        preprocessingVariableNumerique1 = self.preprocessingVariableNumerique()

        preprocessingVariableCategoriel1 = self.preprocessingVariableCategoriel()

        preprocessorVar = make_column_transformer((preprocessingVariableNumerique1, numerical_features),
                                                  (preprocessingVariableCategoriel1, categorical_features),
                                                  )

        return preprocessorVar


    ## Pipeline final de pr??traitement des donn??es (include polynomial features and select best Variables)
    def pipelinePreprocessing(self, methode_selectionVariable=SelectKBest(f_classif, k=10)):

        preprocessingVariable1 = self.preprocessingVariable(self.numerical_features, self.categorical_features)

        preprocessor = make_pipeline(preprocessingVariable1, PolynomialFeatures(2, include_bias=False),
                                     methode_selectionVariable)
        return preprocessor

    def transfom(self):
        self.encodage_label()
        train,test = self.train_test_set()
        X_train,y_train = train
        #print(X_train,y_train)
        resultat = self.pipelinePreprocessing(methode_selectionVariable=SelectKBest(f_classif, k=10))


        #print(pd.DataFrame(resultat.fit_transform(X_train, y_train)))
        return resultat.fit_transform(X_train, y_train)

##===================CLASSE ABSTRAITE DES FONCTIONS APPLICABLES SUR  UN MODEL====================#
class RModel_i:
    @abstractmethod
    def evaluerModel(self):
        pass

    @abstractmethod
    def evalModels(self):
        pass

    @abstractmethod
    def compareModels(self):
        pass

    @abstractmethod
    def rapport(self, data):
        pass

    @abstractmethod
    def  optimisationHyperParam(self,data):
        pass

    @abstractmethod
    def predire(self, data):
        pass

    @abstractmethod
    def save_model(self):
        pass



class RMFrammeClassification(RModel_i,PreprocessingData):

    def __init__(self, listeModel,dataset,target):

        super().__init__(dataset,target)

        train_set, test_set = self.train_test_set()
        self.models = listeModel
        self.best_model = ''
        self.X_train = train_set[0]
        self.y_train = train_set[1]
        self.X_test = test_set[0]
        self.y_test = test_set[1]
        self.model_save = ''
        self.new_dataFrame = ''


    # PROCEDURE D'EVALUATION DES DIFFERRENTS MODELS

    # mesure = ['f1','precision','recall']
    def evaluerModel(self,model):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        precision = accuracy_score(self.y_test, y_pred)
        return precision

    def rapport(self, model):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        # print(confusion_matrix(y_test,y_pred))
        # report = print(classification_report(y_test,y_pred))
        report = classification_report(self.y_test, y_pred)
        return report

    def matrixConfusion(self, model):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        # matrix = print(confusion_matrix(y_test,y_pred))
        matrix = confusion_matrix(self.y_test, y_pred)
        return matrix

    # Fonction qui prend en param??tre une liste de mod??les pr??d??finis et les variables test et entrainement
    # Puis renvoie un dictionnaire contenant tous les mod??les ainsi que leurs scores respectifs

    def evalModels(self):
        precision_dico_models = {}
        for name, model in self.models.items():
            precision = self.evaluerModel(model[0])
            precision_dico_models[name] = precision
        return precision_dico_models

    # Fonction qui permet de faire la comparaison entre les mod??les entrain??s et retourne celui ayant la meilleure
    # performance.
    def compareModels(self, dictionnaire):
        best = 0
        model = ''
        for item, value in dictionnaire.items():
            if best < value:
                best = value
                model = item
        self.best_model = model
        return model, best

    # optimisation du mod??le le plus performant
    def optimisationHyperParam(self, scoring='f1', cv=10):

        model_algo = self.best_model
        grid = RandomizedSearchCV(self.models[model_algo][0], self.models[model_algo][1], scoring=scoring, cv=cv, n_iter=100)
        #grid = GridSearchCV(self.models[model_algo][0], self.models[model_algo][1], scoring=scoring, cv=cv,n_jobs=5, verbose=2)

        model = grid.fit(self.X_train, self.y_train)
        self.model_save = model
        y_pred = grid.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
        return grid


    def show_permences_model(self,dictionnaire):
        import matplotlib.pyplot as plt
        """data = {'Logistic Regression': acc_lr, 'KNN': acc_knn,
                'Support Vector Classifier': acc_svc, 'Decision Tree Classifier': acc_dtc,
                'Random Forest Classifier': acc_rf,
                'Ada Boost Classifier': acc_adc, 'Extra Trees Classifier': acc_etc,
                'Bagging Classifier': acc_bgc, 'Gradient Boosting Classifier': acc_gbc,
                'XGBoost Classifier': acc_xgbc}"""

        data = dict(sorted(dictionnaire.items(), key=lambda x: x[1], reverse=True))
        models = list(data.keys())
        score = list(data.values())
        fig = plt.figure(figsize=(15, 10))
        sns.barplot(x=score, y=models)
        plt.xlabel("Models Utilis??s", size=20)
        plt.xticks(size=12)
        plt.ylabel("Score", size=20)
        plt.yticks(size=12)
        plt.title("Score des mod??les non optimis??s ", size=25)
        plt.show()


    def importance_features(self):
        model_rf = self.model_save.best_estimator_._final_estimator
        X = self.X
        print(X.columns.values)
        #importances = model_rf.feature_importances_
        coef = model_rf.coef_[0]
        print(len(coef))
        weights = pd.Series(coef[:8],
                            index=X.columns.values)
        print(weights.sort_values()[-10:].plot(kind='barh'))


    def executer(self):
        dico_models = self.evalModels()
        self.show_permences_model(dico_models)
        result = self.compareModels(dico_models)

        return dico_models,result


class Scoring(RMFrammeClassification):

    def __init__(self, listeModel,dataset,target):
        super().__init__(listeModel,dataset,target)
        self.dico_score = {}

    def scoring(self):
        data_users = self.dataFrame
        model = self.model_save
        resultat_scoring  =  model.predict_proba(data_users)*100
        classe = model.predict(data_users)
        new_dataFrame = self.dataFrame
        new_dataFrame['score_customer'] = resultat_scoring[:,1]
        new_dataFrame['classe_affecter'] = classe
        self.new_dataFrame = new_dataFrame
        return new_dataFrame

    def scoring_new_data(self,df):
        #df_save = df
        #df = df.drop(=)
        model = self.model_save
        resultat_scoring = model.predict_proba(df) * 100
        classe = model.predict(df)
        new_dataFrame = df
        new_dataFrame['score_customer'] = resultat_scoring[:, 1]
        new_dataFrame['class_customer'] = classe
        new_dataFrame = new_dataFrame.sort_values(by = 'score_customer', ascending=False)
        return new_dataFrame


    def save_model(self):
        model = self.model_save
        filename = 'model_final.sav'
        pickle.dump(model, open(filename, 'wb'))



if __name__ == "__main__":
    ##importation des donn??es
    data = DataImport(source)
    data.chargement()
    id = data.df["Loan_ID"]
    dataset = data.delete_data_entry("Loan_ID",axis=1)
    datat = data.df
    datat['test'] = id
    print(datat)
    print(type(id))

    #print(dataset)
    #acquisition : (visualisation ,  separation des donn??es , pipeline de pretraitement)
    #acquisition = Acquisision(dataset)
    #numeral_data = acquisition.separation_data()
    #train_set,test_set= acquisition.train_test_set()
    #print(train_set,test_set)

    preprocessor = PreprocessingData(dataset)

    preprocessor.encodage_label()

    preprocessor = preprocessor.pipelinePreprocessing()

    #print(preprocessor)

    #===============================================================================
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier

    RandomForest = make_pipeline(preprocessor, RandomForestClassifier(random_state=0))
    AdaBoost = make_pipeline(preprocessor, AdaBoostClassifier(random_state=0))
    SVM = make_pipeline(preprocessor, StandardScaler(), SVC(random_state=0))
    KNN = make_pipeline(preprocessor, StandardScaler(), KNeighborsClassifier())
    Logistic = make_pipeline(preprocessor, LogisticRegression(random_state=0))
    MLP = make_pipeline(preprocessor, MLPClassifier())

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

    # Dictionnaire des models
    list_of_model = {
        "RandomForest": [RandomForest, {}],
        "AdaBoost": [AdaBoost, {}],
        "SVM": [SVM, hyper_params_svm],
        "KNN": [KNN, {}],
        "Logistic": [Logistic, hyper_params_logistic],
        "MLP": [MLP, {}]
    }
    scoring = Scoring(list_of_model,dataset)

    val = scoring.executer()

    scoring.optimisationHyperParam()

    print("le meilleur model est : ",val)

    result = scoring.scoring()




