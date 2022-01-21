import pandas as pd

##===================BIBLIO CLASSE ABSTRAITE
from abc import ABC, abstractmethod

##===================BIBLIO ACQUISION DES DONNEES


##===================IMPORTATION DES BIBLIOTHEQUES DE MACHINE LEARNIND====================#

##=================== Visualisation
import  seaborn as sns
import pickle






if __name__ == "__main__":
    ##importation des données
    data = DataImport(source)
    data.chargement()
    id = data.df["Loan_ID"]
    dataset = data.delete_data_entry("Loan_ID",axis=1)
    datat = data.df
    datat['test'] = id
    print(datat)
    print(type(id))

    #print(dataset)
    #acquisition : (visualisation ,  separation des données , pipeline de pretraitement)
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




