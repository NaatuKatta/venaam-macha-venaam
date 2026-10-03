import os, tempfile, joblib
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

import os, tempfile, joblib
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def prepare_pipeline_files():

    from main import (
        clean_data,
        train_tree_model,
        train_boosting_model
    )

    os.makedirs("processed_data", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    clean_data(
        "data/cold_storage_train.csv",
        "processed_data/cold_storage_train_cleaned.csv"
    )

    clean_data(
        "data/cold_storage_test.csv",
        "processed_data/cold_storage_test_cleaned.csv"
    )

    train_tree_model(
        "processed_data/cold_storage_train_cleaned.csv",
        "artifacts/decision_tree_model.pkl"
    )

    train_boosting_model(
        "processed_data/cold_storage_train_cleaned.csv",
        "artifacts/gradient_boosting_model.pkl"
    )

RAW_TEST_PATH='data/cold_storage_test.csv'; RAW_PREDICT_PATH='data/cold_storage_predict.csv'
TREE_MODEL_PATH='artifacts/decision_tree_model.pkl'; BOOSTING_MODEL_PATH='artifacts/gradient_boosting_model.pkl'
RECORD=['run_id','operation_date','facility_location','chamber_code','shift_supervisor','client_contract_ref']
FEATURES=['stored_load_tonnes','outside_temperature_celsius','door_open_minutes','compressor_age_months','temperature_setpoint_celsius','storage_mode','insulation_grade','uses_rooftop_solar']
TARGET='electricity_consumed_kwh'; _CACHE={}
def scratch(n): return os.path.join(tempfile.mkdtemp(),n)
def raw(load,outside,door,age,setpoint,mode,insulation,solar,target=None):
    n=len(load); d=pd.DataFrame({'run_id':[f'CS-CHECK-{i:04d}' for i in range(n)],'operation_date':['2026-06-15']*n,'facility_location':['Nashik']*n,'chamber_code':['CH-014']*n,'shift_supervisor':['S-Iyer']*n,'client_contract_ref':['CC-55102']*n,'stored_load_tonnes':load,'outside_temperature_celsius':outside,'door_open_minutes':door,'compressor_age_months':age,'temperature_setpoint_celsius':setpoint,'storage_mode':mode,'insulation_grade':insulation,'uses_rooftop_solar':solar})
    if target is not None:d[TARGET]=target
    return d
def heldback():
    if 'v' not in _CACHE:
        from main import clean_data
        d=clean_data(RAW_TEST_PATH,scratch('held.csv')); d=d.iloc[::5].reset_index(drop=True); _CACHE['v']=(d[FEATURES],d[TARGET])
    return _CACHE['v']
def align(m,X):
    f=list(getattr(m,'feature_names_in_',[])); return X[f] if f and set(f)==set(X.columns) else X
def metrics(m,X,y):
    p=np.asarray(m.predict(align(m,X)),float); a=np.asarray(y,float); e=a-p
    return np.sqrt((e**2).mean()),np.abs(e).mean(),1-(e**2).sum()/((a-a.mean())**2).sum()

def test_clean_data_keeps_only_model_fields(tmp_path):
    from main import clean_data
    src=tmp_path/'r.csv'; out=tmp_path/'c.csv'; raw([20,40,60],[25,30,35],[10,20,30],[12,24,36],[-18,-5,4],['frozen','chilled','mixed'],['standard','enhanced','high-efficiency'],['no','yes','no'],[4100,5200,6300]).to_csv(src,index=False)
    d=clean_data(str(src),str(out)); assert isinstance(d,pd.DataFrame); assert len(d)==3; assert not set(RECORD)&set(d.columns); assert set(d.columns)==set(FEATURES+[TARGET]); assert pd.read_csv(out).shape==d.shape

def test_duplicate_removal_precedes_medians(tmp_path):
    from main import clean_data
    base=raw([10,20,30,40,50],[20,24,np.nan,32,100],[10,20,30,np.nan,80],[10,np.nan,30,40,90],[-18,-12,-6,0,6],['frozen','chilled','controlled','mixed','frozen'],['standard','enhanced','high-efficiency','standard','enhanced'],['no','yes','no','yes','no'],[3000,4000,5000,6000,7000])
    pd.concat([base,base.iloc[[4,4,4]]],ignore_index=True).to_csv(tmp_path/'r.csv',index=False)
    d=clean_data(str(tmp_path/'r.csv'),str(tmp_path/'c.csv')); assert len(d)==5
    assert 28.0 in d['outside_temperature_celsius'].values; assert 25.0 in d['door_open_minutes'].values; assert 35.0 in d['compressor_age_months'].values

def test_text_encoding_handles_case_and_spaces(tmp_path):
    from main import clean_data
    src=tmp_path/'r.csv'; raw([10,20,30,40],[25]*4,[10]*4,[12]*4,[-10]*4,[' FROZEN ','chilled',' Controlled','MIXED '],[' STANDARD','enhanced ','HIGH-EFFICIENCY','standard'],[' NO ','yes','No',' YES'],[1,2,3,4]).to_csv(src,index=False)
    d=clean_data(str(src),str(tmp_path/'c.csv')); assert d.storage_mode.tolist()==[0,1,2,3]; assert d.insulation_grade.tolist()==[0,1,2,0]; assert d.uses_rooftop_solar.tolist()==[0,1,0,1]; assert not any(d[c].dtype=='object' for c in d.columns)

def test_unlabelled_cleaning_does_not_invent_target(tmp_path):
    from main import clean_data
    src=tmp_path/'r.csv'; raw([25],[31],[15],[18],[-12],['frozen'],['standard'],['no']).to_csv(src,index=False); d=clean_data(str(src),str(tmp_path/'c.csv')); assert TARGET not in d.columns

def test_tree_model_configuration():
    from main import train_tree_model
    m=train_tree_model('processed_data/cold_storage_train_cleaned.csv',scratch('tree.pkl')); assert isinstance(m,DecisionTreeRegressor); assert m.max_depth==10 and m.random_state==42; assert TARGET not in list(m.feature_names_in_)

def test_boosting_model_configuration():
    from main import train_boosting_model
    m=train_boosting_model('processed_data/cold_storage_train_cleaned.csv',scratch('boost.pkl')); assert isinstance(m,GradientBoostingRegressor); assert m.n_estimators==200 and m.max_depth==3 and abs(m.learning_rate-.1)<1e-12 and m.random_state==42

def test_stored_boosting_model_beats_naive_mean():
    X,y=heldback(); m=joblib.load(BOOSTING_MODEL_PATH); rmse,*_=metrics(m,X,y); baseline=np.sqrt(((y-y.mean())**2).mean()); assert rmse < baseline*.45

def test_compare_models_reports_and_ranks_actual_scores():
    from main import compare_models
    X,y=heldback(); got=compare_models(TREE_MODEL_PATH,BOOSTING_MODEL_PATH,X,y); assert list(got.columns)==['model_name','rmse','mae','r2_score']; assert set(got.model_name)=={'DecisionTreeRegressor','GradientBoostingRegressor'}; assert got.r2_score.tolist()==sorted(got.r2_score,reverse=True)
    for _,r in got.iterrows():
        path=TREE_MODEL_PATH if r.model_name=='DecisionTreeRegressor' else BOOSTING_MODEL_PATH; exp=metrics(joblib.load(path),X,y); assert np.allclose([r.rmse,r.mae,r.r2_score],exp,rtol=1e-7,atol=1e-7)

def test_compare_models_uses_passed_batch_not_disk():
    from main import compare_models
    X,y=heldback(); a=compare_models(TREE_MODEL_PATH,BOOSTING_MODEL_PATH,X.iloc[:80],y.iloc[:80]); b=compare_models(TREE_MODEL_PATH,BOOSTING_MODEL_PATH,X.iloc[80:160],y.iloc[80:160]); assert not np.allclose(a[['rmse','mae','r2_score']],b[['rmse','mae','r2_score']])

def test_evaluate_model_matches_manual_metrics():
    from main import evaluate_model
    X,y=heldback(); got=evaluate_model(BOOSTING_MODEL_PATH,X,y); exp=metrics(joblib.load(BOOSTING_MODEL_PATH),X,y); assert isinstance(got,tuple) and len(got)==3; assert np.allclose(got,exp,rtol=1e-7,atol=1e-7)

def test_evaluate_model_respects_model_path():
    from main import evaluate_model
    X,y=heldback(); assert not np.allclose(evaluate_model(TREE_MODEL_PATH,X,y),evaluate_model(BOOSTING_MODEL_PATH,X,y))

def test_prediction_covers_every_queued_run(tmp_path):
    from main import predict_new_data
    out=predict_new_data(BOOSTING_MODEL_PATH,RAW_PREDICT_PATH,str(tmp_path/'prepared.csv'),str(tmp_path/'pred.csv')); rawdf=pd.read_csv(RAW_PREDICT_PATH); assert list(out.columns)==['run_id','predicted_electricity_consumed_kwh']; assert len(out)==len(rawdf)==800; assert out.run_id.tolist()==rawdf.run_id.tolist(); assert pd.read_csv(tmp_path/'pred.csv').shape==(800,2)

def test_prediction_values_come_from_stored_model(tmp_path):
    from main import predict_new_data, clean_data
    out=predict_new_data(BOOSTING_MODEL_PATH,RAW_PREDICT_PATH,str(tmp_path/'prepared.csv'),str(tmp_path/'pred.csv')); prepared=pd.read_csv(tmp_path/'prepared.csv'); m=joblib.load(BOOSTING_MODEL_PATH); expected=m.predict(align(m,prepared)); assert np.allclose(out.predicted_electricity_consumed_kwh,expected)


RAW_TEST_PATH='data/cold_storage_test.csv'; RAW_PREDICT_PATH='data/cold_storage_predict.csv'
TREE_MODEL_PATH='artifacts/decision_tree_model.pkl'; BOOSTING_MODEL_PATH='artifacts/gradient_boosting_model.pkl'
RECORD=['run_id','operation_date','facility_location','chamber_code','shift_supervisor','client_contract_ref']
FEATURES=['stored_load_tonnes','outside_temperature_celsius','door_open_minutes','compressor_age_months','temperature_setpoint_celsius','storage_mode','insulation_grade','uses_rooftop_solar']
TARGET='electricity_consumed_kwh'; _CACHE={}
def scratch(n): return os.path.join(tempfile.mkdtemp(),n)
def raw(load,outside,door,age,setpoint,mode,insulation,solar,target=None):
    n=len(load); d=pd.DataFrame({'run_id':[f'CS-CHECK-{i:04d}' for i in range(n)],'operation_date':['2026-06-15']*n,'facility_location':['Nashik']*n,'chamber_code':['CH-014']*n,'shift_supervisor':['S-Iyer']*n,'client_contract_ref':['CC-55102']*n,'stored_load_tonnes':load,'outside_temperature_celsius':outside,'door_open_minutes':door,'compressor_age_months':age,'temperature_setpoint_celsius':setpoint,'storage_mode':mode,'insulation_grade':insulation,'uses_rooftop_solar':solar})
    if target is not None:d[TARGET]=target
    return d
def heldback():
    if 'v' not in _CACHE:
        from main import clean_data
        d=clean_data(RAW_TEST_PATH,scratch('held.csv')); d=d.iloc[::5].reset_index(drop=True); _CACHE['v']=(d[FEATURES],d[TARGET])
    return _CACHE['v']
def align(m,X):
    f=list(getattr(m,'feature_names_in_',[])); return X[f] if f and set(f)==set(X.columns) else X
def metrics(m,X,y):
    p=np.asarray(m.predict(align(m,X)),float); a=np.asarray(y,float); e=a-p
    return np.sqrt((e**2).mean()),np.abs(e).mean(),1-(e**2).sum()/((a-a.mean())**2).sum()

def test_clean_data_keeps_only_model_fields(tmp_path):
    from main import clean_data
    src=tmp_path/'r.csv'; out=tmp_path/'c.csv'; raw([20,40,60],[25,30,35],[10,20,30],[12,24,36],[-18,-5,4],['frozen','chilled','mixed'],['standard','enhanced','high-efficiency'],['no','yes','no'],[4100,5200,6300]).to_csv(src,index=False)
    d=clean_data(str(src),str(out)); assert isinstance(d,pd.DataFrame); assert len(d)==3; assert not set(RECORD)&set(d.columns); assert set(d.columns)==set(FEATURES+[TARGET]); assert pd.read_csv(out).shape==d.shape

def test_duplicate_removal_precedes_medians(tmp_path):
    from main import clean_data
    base=raw([10,20,30,40,50],[20,24,np.nan,32,100],[10,20,30,np.nan,80],[10,np.nan,30,40,90],[-18,-12,-6,0,6],['frozen','chilled','controlled','mixed','frozen'],['standard','enhanced','high-efficiency','standard','enhanced'],['no','yes','no','yes','no'],[3000,4000,5000,6000,7000])
    pd.concat([base,base.iloc[[4,4,4]]],ignore_index=True).to_csv(tmp_path/'r.csv',index=False)
    d=clean_data(str(tmp_path/'r.csv'),str(tmp_path/'c.csv')); assert len(d)==5
    assert 28.0 in d['outside_temperature_celsius'].values; assert 25.0 in d['door_open_minutes'].values; assert 35.0 in d['compressor_age_months'].values

def test_text_encoding_handles_case_and_spaces(tmp_path):
    from main import clean_data
    src=tmp_path/'r.csv'; raw([10,20,30,40],[25]*4,[10]*4,[12]*4,[-10]*4,[' FROZEN ','chilled',' Controlled','MIXED '],[' STANDARD','enhanced ','HIGH-EFFICIENCY','standard'],[' NO ','yes','No',' YES'],[1,2,3,4]).to_csv(src,index=False)
    d=clean_data(str(src),str(tmp_path/'c.csv')); assert d.storage_mode.tolist()==[0,1,2,3]; assert d.insulation_grade.tolist()==[0,1,2,0]; assert d.uses_rooftop_solar.tolist()==[0,1,0,1]; assert not any(d[c].dtype=='object' for c in d.columns)

def test_unlabelled_cleaning_does_not_invent_target(tmp_path):
    from main import clean_data
    src=tmp_path/'r.csv'; raw([25],[31],[15],[18],[-12],['frozen'],['standard'],['no']).to_csv(src,index=False); d=clean_data(str(src),str(tmp_path/'c.csv')); assert TARGET not in d.columns

def test_tree_model_configuration():
    from main import train_tree_model
    m=train_tree_model('processed_data/cold_storage_train_cleaned.csv',scratch('tree.pkl')); assert isinstance(m,DecisionTreeRegressor); assert m.max_depth==10 and m.random_state==42; assert TARGET not in list(m.feature_names_in_)

def test_boosting_model_configuration():
    from main import train_boosting_model
    m=train_boosting_model('processed_data/cold_storage_train_cleaned.csv',scratch('boost.pkl')); assert isinstance(m,GradientBoostingRegressor); assert m.n_estimators==200 and m.max_depth==3 and abs(m.learning_rate-.1)<1e-12 and m.random_state==42

def test_stored_boosting_model_beats_naive_mean():
    X,y=heldback(); m=joblib.load(BOOSTING_MODEL_PATH); rmse,*_=metrics(m,X,y); baseline=np.sqrt(((y-y.mean())**2).mean()); assert rmse < baseline*.45

def test_compare_models_reports_and_ranks_actual_scores():
    from main import compare_models
    X,y=heldback(); got=compare_models(TREE_MODEL_PATH,BOOSTING_MODEL_PATH,X,y); assert list(got.columns)==['model_name','rmse','mae','r2_score']; assert set(got.model_name)=={'DecisionTreeRegressor','GradientBoostingRegressor'}; assert got.r2_score.tolist()==sorted(got.r2_score,reverse=True)
    for _,r in got.iterrows():
        path=TREE_MODEL_PATH if r.model_name=='DecisionTreeRegressor' else BOOSTING_MODEL_PATH; exp=metrics(joblib.load(path),X,y); assert np.allclose([r.rmse,r.mae,r.r2_score],exp,rtol=1e-7,atol=1e-7)

def test_compare_models_uses_passed_batch_not_disk():
    from main import compare_models
    X,y=heldback(); a=compare_models(TREE_MODEL_PATH,BOOSTING_MODEL_PATH,X.iloc[:80],y.iloc[:80]); b=compare_models(TREE_MODEL_PATH,BOOSTING_MODEL_PATH,X.iloc[80:160],y.iloc[80:160]); assert not np.allclose(a[['rmse','mae','r2_score']],b[['rmse','mae','r2_score']])

def test_evaluate_model_matches_manual_metrics():
    from main import evaluate_model
    X,y=heldback(); got=evaluate_model(BOOSTING_MODEL_PATH,X,y); exp=metrics(joblib.load(BOOSTING_MODEL_PATH),X,y); assert isinstance(got,tuple) and len(got)==3; assert np.allclose(got,exp,rtol=1e-7,atol=1e-7)

def test_evaluate_model_respects_model_path():
    from main import evaluate_model
    X,y=heldback(); assert not np.allclose(evaluate_model(TREE_MODEL_PATH,X,y),evaluate_model(BOOSTING_MODEL_PATH,X,y))

def test_prediction_covers_every_queued_run(tmp_path):
    from main import predict_new_data
    out=predict_new_data(BOOSTING_MODEL_PATH,RAW_PREDICT_PATH,str(tmp_path/'prepared.csv'),str(tmp_path/'pred.csv')); rawdf=pd.read_csv(RAW_PREDICT_PATH); assert list(out.columns)==['run_id','predicted_electricity_consumed_kwh']; assert len(out)==len(rawdf)==800; assert out.run_id.tolist()==rawdf.run_id.tolist(); assert pd.read_csv(tmp_path/'pred.csv').shape==(800,2)

def test_prediction_values_come_from_stored_model(tmp_path):
    from main import predict_new_data, clean_data
    out=predict_new_data(BOOSTING_MODEL_PATH,RAW_PREDICT_PATH,str(tmp_path/'prepared.csv'),str(tmp_path/'pred.csv')); prepared=pd.read_csv(tmp_path/'prepared.csv'); m=joblib.load(BOOSTING_MODEL_PATH); expected=m.predict(align(m,prepared)); assert np.allclose(out.predicted_electricity_consumed_kwh,expected)
