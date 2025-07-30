from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib

class ChurnPrediction:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.encoder = {}
        self.feature_order = list[str]

    def model_fit(self, df:pd.DataFrame, target_column ="drop_off", category_cols = None)->None:
        if category_cols is None:
            category_cols = ['gender','user_goal','content_preference','user_cluster']
        
        for col in category_cols:
            encode = LabelEncoder()
            df[col] = encode.fit_transform(df[col].astype(str))
            self.encoder[col] = encode

        independent_features = df.drop(columns=[target_column,'user_id'])
        label = df[target_column]
        self.feature_order = independent_features.columns.tolist()
        self.model.fit(independent_features,label)

    def save(self, model_path:str="churn_model.joblib",encoder_path:str="encoder_churn_model.joblib",feature_order_path="feature_order.joblib")->None:
        joblib.dump(self.model,model_path)
        joblib.dump(self.encoder,encoder_path)
        joblib.dump(self.feature_order,feature_order_path)

    def load(self, model_path:str="churn_model.joblib", encoder_path:str="encoder_churn_model.joblib",feature_order_path="feature_order.joblib")->None:
        self.model = joblib.load(model_path)
        self.encoder = joblib.load(encoder_path)
        self.feature_order = joblib.load(feature_order_path)


    def predict_prob(self, user_data:dict)->float:
        if not self.model or not self.encoder:
            raise ValueError("Model and encoders are not loaded ")
        
        encoded = {}

        for col,encoder in self.encoder.items():
            val = str(user_data[col])
            encoded[col] = encoder.transform([val])[0]

        encoded['age'] = user_data['age']
        encoded['sessions'] = user_data['sessions']
        encoded['avg_session_duration'] = user_data['avg_session_duration']

        input_df = pd.DataFrame([encoded])
        input_df = input_df[self.feature_order]
        return float(self.model.predict_proba(input_df)[0][1])

