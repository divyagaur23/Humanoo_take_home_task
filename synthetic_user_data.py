import numpy as np
import pandas as pd

class SyntheticUserData:
    def __init__(self, user_count:int, content_type:list[str], goal:list[str], gender:list[str]):
        self.user_count = user_count
        self.content_type = content_type
        self.goal = goal
        self.gender = gender

    def simulate_user_data(self)->pd.DataFrame:
        """The method creates the synthetic user data for mimicking the user sessions and preferences"""
        np.random.seed(50)

        user_ids = np.arange(1,self.user_count+1)
        gender = np.random.choice(self.gender, size = self.user_count, p=[0.5,0.5])
        age = np.random.randint(18,90,size=self.user_count)
        content_preference = np.random.choice(self.content_type,size = self.user_count,p=[0.4,0.6])
        user_goal = np.random.choice(self.goal, size = self.user_count, p=[0.6,0.4])
        sessions =  np.random.randint(1,10,size=self.user_count)
        avg_session_duration =  np.random.randint(5,25,size=self.user_count)
        drop_off = ((sessions <5) & (avg_session_duration <17)).astype(int)

        user_df = pd.DataFrame({
            'user_id':user_ids,
            'gender':gender,
            'age':age,
            'content_preference':content_preference,
            'user_goal': user_goal,
            'sessions': sessions,
            'avg_session_duration': avg_session_duration,
            'drop_off': drop_off

        })

        return user_df
    
    def save_data(self, df:pd.DataFrame, filename:str="user_data.csv")->pd.DataFrame:
        """save generated syntehtic data"""
        df.to_csv(filename,index=False)

    @staticmethod
    def load_data(filename:str = "user_data.csv")->None:
        """load the saved synthetic data"""
        return pd.read_csv(filename)

