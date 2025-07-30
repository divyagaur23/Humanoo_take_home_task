This project is a functional prototype for predicting early user drop-off in a wellness app using clustering and churn prediction models. 

**Objective**

We needed to create a AI/ML implementation to address the issue of early user drop-off.

It addresses two main issues:
1. Assigning new users to clusters to predict user behavuours early-on when we do not have enough data from the user themselves.
2. Monitoring users when there is some behavioural data is collected and predicting the likellihood of drop-out.

**Project Type:**
It is a CLI (Command Line Interface) based tool. All interactions with the code happen with the terminal inputs.

**Python Version:** 3.12.8

**Install Dependencies:** 
requirements.txt mentions all the libraries required. 

To install the dependencies the following needs to be run:

pip(python_version) install -r requirements.txt

**Running the CLI tool:**
python main.py
Menu-driven tool to:
  1. Create data and train models and save their outcome for further interaction with the tool.
  2. Predict user cluster (for new users)
  3. Predict churn probability (once session data is available)

**Features:**

1. **Synthetic Data Generation** – Simulates realistic user data with engagement signals (it captures age, gender, session and preferences information)
2. **K-Prototypes Clustering** – Groups users based on categorical (goal, preference, gender) and numerical (age) features
3. **Churn Prediction Model** – Trained using Random Forest on session behavior + cluster info


**Folder Structure:**

user-churn-predictor/

├── churn_model.py                    # Random Forest churn model class
├── user_clusters.py                  # K-Prototypes clustering class
├── synthetic_user_data.py            # Synthetic user data generator
├── main.py                           # Main entry for the CLI tool
├── README.md                         # contains information about the logic and the how-to for code

