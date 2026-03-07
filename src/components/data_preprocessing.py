import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder

class DataPreprocessing:
    def PreProcessData(self,data):
    
        x=data.drop(['Loan_ID','Loan_Status'],axis=1)
        y=data['Loan_Status']

        cat_col=x.select_dtypes(include='object').columns
        num_col=x.select_dtypes(exclude='object').columns

        num_pipeline=Pipeline([('Imputer',SimpleImputer(strategy='median')),
                               ('Scaler',StandardScaler())])
        cat_pipeline=Pipeline([('Imputer',SimpleImputer(strategy='most_frequent')),
                               ('Encoder',OneHotEncoder(handle_unknown='ignore'))])
        preprocessor=ColumnTransformer([('num',num_pipeline,num_col),
                                        ('cat',cat_pipeline,cat_col)])
        
        return x,y,preprocessor