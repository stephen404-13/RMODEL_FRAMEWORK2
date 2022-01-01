from biblioFramework import DataImport

from biblioFramework import PreprocessingData

from  biblioFramework import Scoring

from resources import *


if __name__ == "__main__":

    data = DataImport(source)

    data.chargement()

    dataset = data.delete_data_entry("Loan_ID",axis=1)   

    preprocessor = PreprocessingData(dataset)

    preprocessor.encodage_label()

    dataset = preprocessor.dataFrame

    preprocessor = preprocessor.pipeline_preprocessing()

    ###################### INITIALISATION DES Algorithmes NECESSAIRES POUR LE SCORING #################

    RandomForest = make_pipeline(preprocessor, RandomForestClassifier(random_state=0))
    AdaBoost = make_pipeline(preprocessor, AdaBoostClassifier(random_state=0))
    SVM = make_pipeline(preprocessor, SVC(random_state=0))
    KNN = make_pipeline(preprocessor, KNeighborsClassifier())
    Logistic = make_pipeline(preprocessor, LogisticRegression(random_state=0))
    MLP = make_pipeline(preprocessor, MLPClassifier())

    #####Dictionnaire des Algorithmes avec leurs hyperparamètres
    list_of_model = {
        "RandomForest": [RandomForest, {}],
        "AdaBoost": [AdaBoost, {}],
        "SVM": [SVM, hyper_params_svm],
        "KNN": [KNN, {}],
        "Logistic": [Logistic, hyper_params_logistic],
        "MLP": [MLP, {}]
    }


    scoring = Scoring(list_of_model,dataset)

    pair_plot = scoring.matrix_corelation()


    val = scoring.executer()

    print("le meilleur model est : ",val)

    scoring.optimisationHyperParam()


    score_dataFrame = scoring.scoring()


    scoring.save_model()



    print(score_dataFrame)
