from synthetic_user_data import SyntheticUserData
from user_clusters import KModeClustering
from churn_model import ChurnPrediction
import os

def main():
    """Number of usesr and the features have been hardcoded here along with the features"""
    user_count = 150
    """This is the content type present in the app"""
    all_content_types = ['text', 'video']
    """These are the goals offered by the app"""
    all_goals = ['weight loss', 'eat better']
    gender = ['female','male']
    synth_data = SyntheticUserData(user_count,all_content_types,all_goals,gender)
    """These paths will directly save the data in the working directory however these could be updated here"""
    DATA_PATH = "user_data.csv"
    CLUSTER_MODEL_PATH = "cluster_model.joblib"
    CHURN_MODEL_PATH = "churn_model.joblib"
    ENCODER_PATH =  "encoder_churn_model.joblib"
    FEATURE_ORDER_PATH = "feature_order.joblib"

    """The number of clusters is hardcoded"""
    num_clusters = 3
    seed = 50
    clustering = KModeClustering(num_clusters,seed)
    churn_model = ChurnPrediction()
    

    while True:
        print("\nMenu:")
        print("1. Create data, cluster, and train churn model (overwrite existing)")
        print("2. Predict Cluster (for incoming users where we do not have any session info yet)")
        print("3. Predict Churn Rate (only works if the user has had at least one session in the app)")
        print("4. Exit")

        choice = input("Enter choice: ").strip()

        if choice == '1':
            print("Creating synthetic data and training models...")
            df = synth_data.simulate_user_data()
            synth_data.save_data(df, DATA_PATH)

            df = clustering.fit_model(df)
            clustering.save_file(CLUSTER_MODEL_PATH)
            synth_data.save_data(df, DATA_PATH) 

            churn_model.model_fit(df)
            churn_model.save(CHURN_MODEL_PATH, ENCODER_PATH, FEATURE_ORDER_PATH)

            print("Data and models saved successfully.")

        elif choice == '2':

            if not (os.path.exists(DATA_PATH) and os.path.exists(CLUSTER_MODEL_PATH) and os.path.exists(CHURN_MODEL_PATH)):
                print("Error: Data and models not found. Please run option 1 first.")
                continue

            df = SyntheticUserData.load_data(DATA_PATH)
            clustering.load_file(CLUSTER_MODEL_PATH)
            churn_model.load(CHURN_MODEL_PATH, ENCODER_PATH, FEATURE_ORDER_PATH)

            print("Enter user profile info for cluster prediction:")
            user_profile = {}
            user_profile['user_goal'] = input("User goal (weight loss/eat better): ").strip()
            user_profile['content_preference'] = input("Content preference (text/video): ").strip()
            user_profile['age'] = int(input("Age: ").strip())
            user_profile['gender'] = input("Gender (female/male): ").strip()

            cluster = clustering.predict_cluster(user_profile)
            avg_churn = df[df['user_cluster'] == cluster]['drop_off'].mean()

            print(f"User assigned to cluster {cluster} with average churn rate {avg_churn:.2f}")

        elif choice == '3':

            if not (os.path.exists(DATA_PATH) and os.path.exists(CLUSTER_MODEL_PATH) and os.path.exists(CHURN_MODEL_PATH)):
                print("Error: Data and models not found. Please run option 1 first.")
                continue

            df = SyntheticUserData.load_data(DATA_PATH)
            clustering.load_file(CLUSTER_MODEL_PATH)
            churn_model.load(CHURN_MODEL_PATH, ENCODER_PATH, FEATURE_ORDER_PATH)

            print("Enter user profile info for cluster prediction:")
            user_profile = {}
            user_profile['user_goal'] = input("User goal (weight loss/eat better): ").strip()
            user_profile['content_preference'] = input("Content preference (text/video): ").strip()
            user_profile['age'] = int(input("Age: ").strip())
            user_profile['gender'] = input("Gender (female/male): ").strip()
            

            cluster = clustering.predict_cluster(user_profile)

            print("\nEnter session info for churn prediction:")
            user_profile['sessions'] = int(input("Session: ").strip())
            if user_profile['sessions'] == 1:
                avg_session_duration = 1
            else:
                avg_session_duration = int(input("Average session duration: ").strip())
            user_data = {
                         "gender": user_profile['gender'],
                         "user_goal": user_profile['user_goal'],
                         "content_preference": user_profile['content_preference'],
                         "user_cluster": cluster,
                          "age": user_profile['age'],
                         "sessions": user_profile['sessions'],
                         "avg_session_duration": avg_session_duration
    }
            churn_prob = churn_model.predict_prob(user_data)
            cluster_sessions_avg = df[df['user_cluster'] == cluster]['sessions'].mean()
            cluster_avg_session_dur = df[df['user_cluster'] == cluster]['avg_session_duration'].mean()
            avg_churn = df[df['user_cluster'] == cluster]['drop_off'].mean()
            print(f"Churn Probablity of the user: {churn_prob:.2f}, average number of sessions for users of this cluster is: {cluster_sessions_avg:.0f} and average session duration is: {cluster_avg_session_dur:.0f} minutes with average churn rate {avg_churn:.2f}")



        elif choice == '4':
            print("Exiting.")
            break


if __name__ == "__main__":
    main()
