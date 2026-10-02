import os
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

Train_path = 'data/machine_condition_train.csv'
Test_path = 'data/machine_condition_test.csv'
Predict_path = 'data/machine_condition_predict.csv'

Train_cleaned = 'processed_data/machine_condition_train_cleaned.csv'
Test_cleaned = 'processed_data/machine_condition_test_cleaned.csv'
Predict_cleaned = 'processed_data/machine_condition_predict_cleaned.csv'

Logistic_path = 'artifacts/logistic_regression_model.pkl'
Svc_path = 'artifacts/svc_model.pkl'

Output_path = 'output/machine_failure_risk_predictions.csv'

def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df.drop(columns=['machine_id','inspection_date','plant_location','production_line','shift_engineer','maintenance_order_ref'])
    df = df.drop_duplicates(keep='first')

    for i in ['vibration_mm_s','bearing_temperature_celsius','average_daily_runtime_hours']:
        df[i] = df[i].fillna(df[i].median())

    df['machine_type'] = df['machine_type'].str.lower().str.strip().map({
        'cnc_lathe': 0,
        'milling_center': 1,
        'hydraulic_press': 2
    })

    df['load_profile'] = df['load_profile'].str.lower().str.strip().map({
        'light': 0,
        'normal': 1,
        'heavy': 2,
        'cyclic': 3
    })

    df['lubrication_system'] = df['lubrication_system'].str.lower().str.strip().map({
        'manual': 0,
        'automatic': 1
    })

    if 'failure_risk_band' in df.columns:
        df['failure_risk_band'] = df['failure_risk_band'].str.upper().str.strip().map({
            'VERY_LOW': 0,
            'LOW': 1,
            'MODERATE': 2,
            'HIGH': 3,
            'CRITICAL': 4
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

def train_logistic_model(processed_path, model_path):
    df = pd.read_csv(processed_path)
    X = df.drop(columns=['failure_risk_band'])
    y = df['failure_risk_band']

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    model.fit(X, y)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    return model

def train_svc_model(processed_path, model_path):
    df = pd.read_csv(processed_path)
    X = df.drop(columns=['failure_risk_band'])
    y = df['failure_risk_band']
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(
            kernel='rbf',
            C=1.0,
            random_state=42
        ))
    ])
    model.fit(X, y)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    return model

def compare_models(logistic_model_path,svc_model_path,features_test,target_test):
    logistic = joblib.load(logistic_model_path)
    svc = joblib.load(svc_model_path)
    logistic_pred = logistic.predict(features_test)
    svc_pred = svc.predict(features_test)
    result = pd.DataFrame([
        {
            'model_name': 'LogisticRegression',
            'accuracy': accuracy_score( target_test, logistic_pred),
            'f1_score': f1_score(target_test,logistic_pred,average='macro')
        },
        {
            'model_name': 'SVC',
            'accuracy': accuracy_score(target_test,svc_pred),
            'f1_score': f1_score(target_test,svc_pred,average='macro')
        }
    ])

    result = result.sort_values('accuracy',ascending=False).reset_index(drop=True)
    return result

def evaluate_model(model_path, features_test, target_test):
    model = joblib.load(model_path)
    pred = model.predict(features_test)
    accuracy = accuracy_score(target_test,pred)
    f1_score_val = f1_score(target_test,pred,average='macro')
    return accuracy, f1_score_val

def predict_new_data(model_path,input_path,processed_path,output_path):
    model = joblib.load(model_path)
    raw_df = pd.read_csv(input_path)

    ids = raw_df['machine_id']
    clean_df = clean_data(input_path,processed_path)

    pred = model.predict(clean_df)
    risk_map = {
        0: 'VERY_LOW',
        1: 'LOW',
        2: 'MODERATE',
        3: 'HIGH',
        4: 'CRITICAL'
    }
    map_pred = [risk_map[int(p)] for p in pred]

    result = pd.DataFrame({
        'machine_id': ids,
        'predicted_failure_risk_band': map_pred })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path,index=False)
    return result

if __name__ == '__main__':
    train = clean_data(Train_path,Train_cleaned)
    test = clean_data(Test_path,Test_cleaned)
    print(train.size)
    print(test.size)
    train_logistic_model(Train_cleaned,Logistic_path)
    train_svc_model(Train_cleaned,Svc_path)
    print('Models trained and saved.')
    features = test.drop(columns=['failure_risk_band'])
    target = test['failure_risk_band']
    comparison = compare_models(Logistic_path,Svc_path,features,target)
    print(comparison)
    best_model = comparison.loc[0,'model_name']
    path = {'LogisticRegression': Logistic_path,'SVC': Svc_path}
    best_path = path.get(best_model)
    a, b = evaluate_model(best_path,features,target)
    print(f'accuracy:{a:.4f}')
    print(f'f1_score:{b:.4f}')
    df = predict_new_data(best_path,Predict_path,Predict_cleaned,Output_path)
    print(df.shape)