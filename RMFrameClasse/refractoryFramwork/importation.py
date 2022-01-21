##=================== CLASS D'IMPORTATION DES DONNEES ====================#
import os

import pandas as pd

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
