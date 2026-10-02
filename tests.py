import os
import tempfile
import joblib
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

RAW_TRAIN_PATH='data/machine_condition_train.csv'
RAW_TEST_PATH='data/machine_condition_test.csv'
RAW_PREDICT_PATH='data/machine_condition_predict.csv'
CLEAN_TRAIN_PATH='processed_data/machine_condition_train_cleaned.csv'
CLEAN_TEST_PATH='processed_data/machine_condition_test_cleaned.csv'
LOGISTIC_MODEL_PATH='artifacts/logistic_regression_model.pkl'
SVC_MODEL_PATH='artifacts/svc_model.pkl'
RECORD=['machine_id','inspection_date','plant_location','production_line','shift_engineer','maintenance_order_ref']
FEATURES=['operating_hours_thousands','days_since_service','vibration_mm_s','bearing_temperature_celsius','average_daily_runtime_hours','machine_type','load_profile','lubrication_system']
TARGET='failure_risk_band'
BANDS=['VERY_LOW','LOW','MODERATE','HIGH','CRITICAL']
_CACHE={}

def scratch(name): return os.path.join(tempfile.mkdtemp(),name)

def raw(hours,days,vib,temp,runtime,mtype,load,lub,target=None):
    n=len(hours)
    d=pd.DataFrame({'machine_id':[f'MC-CHECK-{i:04d}' for i in range(n)],'inspection_date':['2026-06-15']*n,'plant_location':['Pune']*n,'production_line':['Line II']*n,'shift_engineer':['R-Mehta']*n,'maintenance_order_ref':['MO-55102']*n,'operating_hours_thousands':hours,'days_since_service':days,'vibration_mm_s':vib,'bearing_temperature_celsius':temp,'average_daily_runtime_hours':runtime,'machine_type':mtype,'load_profile':load,'lubrication_system':lub})
    if target is not None: d[TARGET]=target
    return d

@pytest.fixture(scope='session', autouse=True)
def prepare_pipeline_files():
    from main import clean_data, train_logistic_model, train_svc_model
    os.makedirs('processed_data',exist_ok=True); os.makedirs('artifacts',exist_ok=True); os.makedirs('output',exist_ok=True)
    clean_data(RAW_TRAIN_PATH,CLEAN_TRAIN_PATH)
    clean_data(RAW_TEST_PATH,CLEAN_TEST_PATH)
    train_logistic_model(CLEAN_TRAIN_PATH,LOGISTIC_MODEL_PATH)
    train_svc_model(CLEAN_TRAIN_PATH,SVC_MODEL_PATH)

def heldback():
    if 'v' not in _CACHE:
        d=pd.read_csv(CLEAN_TEST_PATH).iloc[::5].reset_index(drop=True)
        _CACHE['v']=(d[FEATURES],d[TARGET])
    return _CACHE['v']

def metrics(model,X,y):
    p=model.predict(X)
    return accuracy_score(y,p),f1_score(y,p,average='macro')

def test_clean_data_keeps_only_model_fields(tmp_path):
    from main import clean_data
    src=tmp_path/'r.csv'; out=tmp_path/'c.csv'
    raw([1,2,3],[10,20,30],[2,3,4],[55,65,75],[8,10,12],['cnc_lathe','milling_center','hydraulic_press'],['light','normal','heavy'],['manual','automatic','manual'],['VERY_LOW','LOW','HIGH']).to_csv(src,index=False)
    d=clean_data(str(src),str(out)); assert isinstance(d,pd.DataFrame); assert len(d)==3; assert not set(RECORD)&set(d.columns); assert set(d.columns)==set(FEATURES+[TARGET]); assert pd.read_csv(out).shape==d.shape

def test_duplicate_removal_precedes_medians(tmp_path):
    from main import clean_data
    base=raw([1,2,3,4,5],[10,20,30,40,50],[1,2,np.nan,4,20],[40,np.nan,60,80,140],[4,8,12,np.nan,24],['cnc_lathe']*5,['light']*5,['manual']*5,['VERY_LOW','LOW','MODERATE','HIGH','CRITICAL'])
    pd.concat([base,base.iloc[[4,4,4]]],ignore_index=True).to_csv(tmp_path/'r.csv',index=False)
    d=clean_data(str(tmp_path/'r.csv'),str(tmp_path/'c.csv')); assert len(d)==5
    assert 3.0 in d.vibration_mm_s.values; assert 70.0 in d.bearing_temperature_celsius.values; assert 10.0 in d.average_daily_runtime_hours.values

def test_text_and_target_encoding_handles_case_spaces(tmp_path):
    from main import clean_data
    src=tmp_path/'r.csv'; raw([1]*4,[10]*4,[2]*4,[60]*4,[8]*4,[' CNC_LATHE ','milling_center',' Hydraulic_Press ','cnc_lathe'],[' LIGHT ','normal','HEAVY ',' cyclic '],[' MANUAL ','automatic','Manual',' AUTOMATIC '],[' VERY_LOW ','low',' Moderate ','CRITICAL ']).to_csv(src,index=False)
    d=clean_data(str(src),str(tmp_path/'c.csv')); assert d.machine_type.tolist()==[0,1,2,0]; assert d.load_profile.tolist()==[0,1,2,3]; assert d.lubrication_system.tolist()==[0,1,0,1]; assert d[TARGET].tolist()==[0,1,2,4]; assert not any(d[c].dtype=='object' for c in d.columns)

def test_unlabelled_cleaning_does_not_invent_target(tmp_path):
    from main import clean_data
    src=tmp_path/'r.csv'; raw([2],[30],[3],[60],[10],['cnc_lathe'],['normal'],['automatic']).to_csv(src,index=False); d=clean_data(str(src),str(tmp_path/'c.csv')); assert TARGET not in d.columns

def test_logistic_model_uses_scaler_and_required_classifier(tmp_path):
    from main import train_logistic_model
    m=train_logistic_model(CLEAN_TRAIN_PATH,str(tmp_path/'lr.pkl')); assert isinstance(m,Pipeline); assert isinstance(m.named_steps['scaler'],StandardScaler); c=m.named_steps['classifier']; assert isinstance(c,LogisticRegression); assert c.max_iter==1000; assert TARGET not in list(m.feature_names_in_)

def test_svc_model_uses_scaler_and_required_classifier(tmp_path):
    from main import train_svc_model
    m=train_svc_model(CLEAN_TRAIN_PATH,str(tmp_path/'svc.pkl')); assert isinstance(m,Pipeline); assert isinstance(m.named_steps['scaler'],StandardScaler); c=m.named_steps['classifier']; assert isinstance(c,SVC); assert c.kernel=='rbf' and c.C==1.0 and c.random_state==42

def test_scaler_statistics_come_from_training_features():
    d=pd.read_csv(CLEAN_TRAIN_PATH); X=d[FEATURES]; m=joblib.load(SVC_MODEL_PATH); s=m.named_steps['scaler']; assert np.allclose(s.mean_,X.mean().values,rtol=1e-7,atol=1e-7)

def test_compare_models_reports_macro_f1_and_ranks_accuracy():
    from main import compare_models
    X,y=heldback(); got=compare_models(LOGISTIC_MODEL_PATH,SVC_MODEL_PATH,X,y); assert list(got.columns)==['model_name','accuracy','f1_score']; assert set(got.model_name)=={'LogisticRegression','SVC'}; assert got.accuracy.tolist()==sorted(got.accuracy,reverse=True)
    for _,r in got.iterrows():
        path=LOGISTIC_MODEL_PATH if r.model_name=='LogisticRegression' else SVC_MODEL_PATH; exp=metrics(joblib.load(path),X,y); assert np.allclose([r.accuracy,r.f1_score],exp,rtol=1e-7,atol=1e-7)

def test_compare_models_uses_passed_batch_not_disk():
    from main import compare_models
    X,y=heldback(); a=compare_models(LOGISTIC_MODEL_PATH,SVC_MODEL_PATH,X.iloc[:100],y.iloc[:100]); b=compare_models(LOGISTIC_MODEL_PATH,SVC_MODEL_PATH,X.iloc[100:200],y.iloc[100:200]); assert not np.allclose(a[['accuracy','f1_score']],b[['accuracy','f1_score']])

def test_evaluate_model_matches_manual_metrics():
    from main import evaluate_model
    X,y=heldback(); got=evaluate_model(SVC_MODEL_PATH,X,y); exp=metrics(joblib.load(SVC_MODEL_PATH),X,y); assert isinstance(got,tuple) and len(got)==2; assert np.allclose(got,exp,rtol=1e-7,atol=1e-7)

def test_evaluate_model_respects_model_path():
    from main import evaluate_model
    X,y=heldback(); assert not np.allclose(evaluate_model(LOGISTIC_MODEL_PATH,X,y),evaluate_model(SVC_MODEL_PATH,X,y))

def test_prediction_covers_every_machine_and_decodes_band_names(tmp_path):
    from main import predict_new_data
    out=predict_new_data(SVC_MODEL_PATH,RAW_PREDICT_PATH,str(tmp_path/'prepared.csv'),str(tmp_path/'pred.csv')); rawdf=pd.read_csv(RAW_PREDICT_PATH); assert list(out.columns)==['machine_id','predicted_failure_risk_band']; assert len(out)==len(rawdf)==800; assert out.machine_id.tolist()==rawdf.machine_id.tolist(); assert set(out.predicted_failure_risk_band).issubset(set(BANDS)); assert pd.read_csv(tmp_path/'pred.csv').shape==(800,2)

def test_prediction_values_come_from_stored_model(tmp_path):
    from main import predict_new_data
    out=predict_new_data(SVC_MODEL_PATH,RAW_PREDICT_PATH,str(tmp_path/'prepared.csv'),str(tmp_path/'pred.csv')); prepared=pd.read_csv(tmp_path/'prepared.csv'); m=joblib.load(SVC_MODEL_PATH); expected=[BANDS[int(x)] for x in m.predict(prepared)]; assert out.predicted_failure_risk_band.tolist()==expected
