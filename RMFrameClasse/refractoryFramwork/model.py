##===================CLASSE ABSTRAITE DES FONCTIONS APPLICABLES SUR  UN MODEL====================#
from .pretraitement import *

##===================BIBLIOTHEQUE POUR POUR LA MESURE DE PERFORMENCE DU MODEL L'OPTIMISATION DES HYPERPARAMETRES
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, confusion_matrix, classification_report
#from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score

import  seaborn as sns

##===================BIBLIO CLASSE ABSTRAITE
from abc import ABC, abstractmethod

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

    # Fonction qui prend en paramètre une liste de modèles prédéfinis et les variables test et entrainement
    # Puis renvoie un dictionnaire contenant tous les modèles ainsi que leurs scores respectifs

    def evalModels(self):
        precision_dico_models = {}
        for name, model in self.models.items():
            precision = self.evaluerModel(model[0])
            precision_dico_models[name] = precision
        return precision_dico_models

    # Fonction qui permet de faire la comparaison entre les modèles entrainés et retourne celui ayant la meilleure
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

    # optimisation du modèle le plus performant
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
        plt.xlabel("Models Utilisés", size=20)
        plt.xticks(size=12)
        plt.ylabel("Score", size=20)
        plt.yticks(size=12)
        plt.title("Score des modèles non optimisés ", size=25)
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



