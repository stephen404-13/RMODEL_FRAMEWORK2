#from RModelFramework import DataImport
#from RModelFramework import PreprocessingData
#from  RModelFramework import Scoring
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from RMFRAMEWORK.bibliotheque.RMFrameClasse.ressources.resources import *

from RMFRAMEWORK.bibliotheque.RMFrameClasse.refractoryFramwork.importation import  DataImport
from RMFRAMEWORK.bibliotheque.RMFrameClasse.refractoryFramwork.pretraitement import PreprocessingData
from RMFRAMEWORK.bibliotheque.RMFrameClasse.refractoryFramwork.scoring import Scoring
from RMFRAMEWORK.bibliotheque.RMFrameClasse.refractoryFramwork.classification import Classification
if __name__ == "__main__":
    #pd.set_option('display.max_columns', None)

    data = DataImport(source)
    data.chargement()
    data.display_data(20)

    #target = 'Loan_Status'
    # dataset = data.delete_data_entry("Loan_ID",axis=1)

    target = "Churn"
    dataset = data.delete_data_entry("customerID", axis=1)

    print(dataset.columns)
    preprocessor1 = PreprocessingData(dataset,target,strategy_val_manquante_num='mean',methode_normalisation=StandardScaler(), strategy_val_manquante_cat='most_frequent',methode_encodage=OneHotEncoder())

    preprocessor1.encodage_label()

    ##donnee transformees

    data_traiter = preprocessor1.transfom()

    print("REPRESENTATION DES DONNEES PRETRAITER")
    print(data_traiter)

    dataset = preprocessor1.dataFrame

    print(dataset.head(20))

    preprocessor = preprocessor1.pipelinePreprocessing()


    ###################### INITIALISATION DES Algorithmes NECESSAIRES POUR LE SCORING #################

    RandomForest = make_pipeline(preprocessor, RandomForestClassifier(random_state=0))
    AdaBoost = make_pipeline(preprocessor, AdaBoostClassifier(random_state=0))
    SVM = make_pipeline(preprocessor, SVC(random_state=0))
    KNN = make_pipeline(preprocessor, KNeighborsClassifier())
    Logistic = make_pipeline(preprocessor, LogisticRegression(random_state=0))
    MLP = make_pipeline(preprocessor, MLPClassifier())
    LDA = make_pipeline(preprocessor,LinearDiscriminantAnalysis())
    Binary_tree = make_pipeline(preprocessor,tree.DecisionTreeClassifier())

    #####Dictionnaire des Algorithmes avec leurs hyperparamètres

    list_of_model = {
        "RandomForest": [RandomForest, hyper_params_radomForest],
        #"AdaBoost": [AdaBoost, hyper_params_AdaBoost],
        #"SVM": [SVM, hyper_params_svm],
        #"KNN": [KNN, hyper_params_knn],
        "Logistic": [Logistic, hyper_params_logistic],
        "MLP": [MLP, hyper_params_MLP],
        "LDA":[LDA,hyper_params_lda],
        "Binary_tree":[Binary_tree,hyper_params_tree]
    }

    classement = Classification(list_of_model,dataset,target)

    #pair_plot = scoring.matrix_corelation()

    #print(pair_plot)

    performences_models,best_model = classement.executer()

    print("la performence de tous les models  : ",performences_models)
    print("Le meilleur model est models:",best_model)

    classement.optimisationHyperParam()

    classement.save_model()

    classement.importance_features()

    #score_dataFrame = classement.scoring()
    #====================SCORING SUR DE NOUVEAU DONNEES ======================"=====


    class_dataFrame = classement.do_classification(df_new_scustomer)

    print(class_dataFrame)
