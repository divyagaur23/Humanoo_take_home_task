import pandas as pd
from kmodes.kprototypes import KPrototypes
import joblib
import numpy as np

class KModeClustering:
    def __init__(self, num_clusters:int,seed:int):
        self.num_clusters = num_clusters
        self.seed = seed
        self.model = KPrototypes(n_clusters=self.num_clusters, init='Huang',random_state=self.seed) 
        self.indices_cat = [0,1,3]

    def data_prep(self, df:pd.DataFrame)->pd.DataFrame:
        """Data preprocessing for clustering"""
        df['user_goal'] = df['user_goal'].astype(str)
        df['content_preference'] = df['content_preference'].astype(str)
        df['gender'] = df['gender'].astype(str)
        return df
    
    def fit_model(self, df:pd.DataFrame)->pd.DataFrame:
        """Groups users based on categorical (goal, preference, gender) and numerical (age) features"""
        df = self.data_prep(df)
        train_data = df[['user_goal','content_preference','age','gender']].copy()
        train_matrix = train_data.to_numpy()

        clustres = self.model.fit_predict(train_matrix, categorical=self.indices_cat)
        df['user_cluster'] = clustres

        return df
    
    def save_file(self, path="cluster_model.joblib") -> None:
        """Save the resulting clustering mdoel for further use"""
        joblib.dump(self.model, path)

    def load_file(self, path = "cluster_model.joblib") ->None:
        """load the saved model"""
        self.model = joblib.load(path)

    def predict_cluster(self, user_data:dict) ->int:
        """Predict the cluster of the new user"""
        row = [
            str(user_data['user_goal']),
            str(user_data['content_preference']),
            user_data['age'],
            str(user_data['gender'])
        ]

        data_row = np.array([row])
        cluster_label = self.model.predict(data_row, categorical=self.indices_cat)[0]
        return cluster_label

 
