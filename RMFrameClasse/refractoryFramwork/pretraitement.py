##=================== CLASSE  DE PRETRAITEMENT DES DONNEES ====================#
from .aquisition import *

class PreprocessingData(Acquisision):

    def __init__(self,dataset,target,strategy_val_manquante_num='mean',methode_normalisation=StandardScaler(), strategy_val_manquante_cat='most_frequent',methode_encodage=OneHotEncoder()):
        super().__init__(dataset,target)
        self.numerical_features = self.apply_separation_data(self.dataFrame)[0]
        self.categorical_features = self.apply_separation_data(self.dataFrame)[1]
        self.strategy_val_manquante_num= strategy_val_manquante_num
        self.methode_normalisation=methode_normalisation
        self.strategy_val_manquante_cat = strategy_val_manquante_cat
        self.methode_encodage = methode_encodage

    def preprocessingVariableNumerique(self):
        numerical_pipeline = make_pipeline(SimpleImputer(strategy=self.strategy_val_manquante_num), self.methode_normalisation)
        return numerical_pipeline

    ## strategie de prétraitement des valeurs  catégorielles
    def preprocessingVariableCategoriel(self):
        categoriel_pipeline = make_pipeline(SimpleImputer(strategy=self.strategy_val_manquante_cat), self.methode_encodage)
        return categoriel_pipeline

    ##  appliquer le strategie sur les variables
    def preprocessingVariable(self, numerical_features, categorical_features):
        preprocessingVariableNumerique1 = self.preprocessingVariableNumerique()

        preprocessingVariableCategoriel1 = self.preprocessingVariableCategoriel()

        preprocessorVar = make_column_transformer((preprocessingVariableNumerique1, numerical_features),
                                                  (preprocessingVariableCategoriel1, categorical_features),
                                                  )

        return preprocessorVar

    ## Pipeline final de prétraitement des données (include polynomial features and select best Variables)
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