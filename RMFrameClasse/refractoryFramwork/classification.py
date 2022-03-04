from .model import *


import pickle



class Classification(RMFrammeClassification):

    def __init__(self, listeModel,dataset,target):
        super().__init__(listeModel,dataset,target)
        self.dico_score = {}

    """def scoring(self):
        data_users = self.dataFrame
        model = self.model_save
        resultat_scoring  =  model.predict_proba(data_users)*100
        classe = model.predict(data_users)
        new_dataFrame = self.dataFrame
        new_dataFrame['score_customer'] = resultat_scoring[:,1]
        new_dataFrame['classe_affecter'] = classe
        self.new_dataFrame = new_dataFrame
        return new_dataFrame"""

    def do_classification(self):
        data_users = self.dataFrame
        model = self.model_save
        classe = model.predict(data_users)
        new_dataFrame = self.dataFrame
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