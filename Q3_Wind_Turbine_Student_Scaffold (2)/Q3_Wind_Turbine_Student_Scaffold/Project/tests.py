import os, tempfile, joblib, pytest, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error,mean_absolute_error,r2_score
RAW_TRAIN_PATH='data/wind_turbine_train.csv'; RAW_TEST_PATH='data/wind_turbine_test.csv'; RAW_PREDICT_PATH='data/wind_turbine_predict.csv'
CLEAN_TRAIN_PATH='processed_data/wind_turbine_train_cleaned.csv'; CLEAN_TEST_PATH='processed_data/wind_turbine_test_cleaned.csv'; SVR_MODEL_PATH='artifacts/svr_model.pkl'; FOREST_MODEL_PATH='artifacts/random_forest_model.pkl'
RECORD=['turbine_id','inspection_date','wind_farm','service_region','maintenance_vendor','work_order_ref']; FEATURES=['turbine_age_months','operating_hours','rated_capacity_mw','prior_fault_count','gearbox_vibration_mm_s','drive_train_type','weather_exposure','offshore_installation']; TARGET='annual_maintenance_cost_lakh'; _CACHE={}

def raw(age,hours,cap,faults,vib,drive,weather,offshore,target=None):
 n=len(age); d=pd.DataFrame({'turbine_id':[f'WT-CHECK-{i:03d}' for i in range(n)],'inspection_date':['2026-07-15']*n,'wind_farm':['Kutch']*n,'service_region':['West']*n,'maintenance_vendor':['MV-Atlas']*n,'work_order_ref':['WO-X']*n,'turbine_age_months':age,'operating_hours':hours,'rated_capacity_mw':cap,'prior_fault_count':faults,'gearbox_vibration_mm_s':vib,'drive_train_type':drive,'weather_exposure':weather,'offshore_installation':offshore});
 if target is not None: d[TARGET]=target
 return d
@pytest.fixture(scope='session',autouse=True)
def prepare_pipeline_files():
 from main import clean_data,train_svr_model,train_forest_model
 os.makedirs('processed_data',exist_ok=True); os.makedirs('artifacts',exist_ok=True); os.makedirs('output',exist_ok=True); clean_data(RAW_TRAIN_PATH,CLEAN_TRAIN_PATH); clean_data(RAW_TEST_PATH,CLEAN_TEST_PATH); train_svr_model(CLEAN_TRAIN_PATH,SVR_MODEL_PATH); train_forest_model(CLEAN_TRAIN_PATH,FOREST_MODEL_PATH)
def heldback():
 if 'v' not in _CACHE:
  d=pd.read_csv(CLEAN_TEST_PATH).iloc[::5].reset_index(drop=True); _CACHE['v']=(d[FEATURES],d[TARGET])
 return _CACHE['v']
def metrics(m,X,y):
 p=m.predict(X); return root_mean_squared_error(y,p),mean_absolute_error(y,p),r2_score(y,p)
def test_clean_data_keeps_only_model_fields(tmp_path):
 from main import clean_data
 s=tmp_path/'r.csv'; raw([20,30],[10000,20000],[2,3],[1,2],[2,3],['geared','hybrid'],['low','high'],['no','yes'],[8,12]).to_csv(s,index=False); d=clean_data(str(s),str(tmp_path/'c.csv')); assert set(d.columns)==set(FEATURES+[TARGET]); assert not set(RECORD)&set(d.columns); assert len(d)==2; assert pd.read_csv(tmp_path/'c.csv').shape==d.shape
def test_duplicate_removal_precedes_medians(tmp_path):
 from main import clean_data
 b=raw([20,30,40,50,60],[10,20,np.nan,40,100],[1,2,3,np.nan,9],[1]*5,[1,2,3,np.nan,20],['geared']*5,['low']*5,['no']*5,[5,6,7,8,9]); pd.concat([b,b.iloc[[4,4,4]]],ignore_index=True).to_csv(tmp_path/'r.csv',index=False); d=clean_data(str(tmp_path/'r.csv'),str(tmp_path/'c.csv')); assert len(d)==5; assert 30.0 in d.operating_hours.values; assert 2.5 in d.rated_capacity_mw.values; assert 2.5 in d.gearbox_vibration_mm_s.values
def test_text_encoding_handles_case_and_spaces(tmp_path):
 from main import clean_data
 s=tmp_path/'r.csv'; raw([20]*4,[10000]*4,[2]*4,[1]*4,[2]*4,[' GEARED ','direct_drive',' HYBRID ','geared'],[' LOW ','moderate','HIGH ',' extreme '],[' NO ','yes','No',' YES '],[5]*4).to_csv(s,index=False); d=clean_data(str(s),str(tmp_path/'c.csv')); assert d.drive_train_type.tolist()==[0,1,2,0]; assert d.weather_exposure.tolist()==[0,1,2,3]; assert d.offshore_installation.tolist()==[0,1,0,1]; assert not any(d[c].dtype=='object' for c in d.columns)
def test_unlabelled_cleaning_does_not_invent_target(tmp_path):
 from main import clean_data
 s=tmp_path/'r.csv'; raw([20],[10000],[2],[1],[2],['geared'],['low'],['no']).to_csv(s,index=False); assert TARGET not in clean_data(str(s),str(tmp_path/'c.csv')).columns
def test_svr_model_uses_scaler_and_default_rbf_svr(tmp_path):
 from main import train_svr_model
 sample=pd.read_csv(CLEAN_TRAIN_PATH).iloc[:2000]; sp=tmp_path/'sample.csv'; sample.to_csv(sp,index=False); m=train_svr_model(str(sp),str(tmp_path/'m.pkl')); assert isinstance(m,Pipeline); assert isinstance(m.named_steps['scaler'],StandardScaler); r=m.named_steps['regressor']; assert isinstance(r,SVR); assert r.kernel=='rbf' and r.C==1.0 and r.epsilon==0.1; assert TARGET not in list(m.feature_names_in_)
def test_forest_model_has_required_configuration(tmp_path):
 from main import train_forest_model
 sample=pd.read_csv(CLEAN_TRAIN_PATH).iloc[:2000]; sp=tmp_path/'sample.csv'; sample.to_csv(sp,index=False); m=train_forest_model(str(sp),str(tmp_path/'m.pkl')); assert isinstance(m,RandomForestRegressor); assert m.n_estimators==100 and m.max_depth==14 and m.min_samples_leaf==15 and m.random_state==42; assert TARGET not in list(m.feature_names_in_)
def test_svr_scaler_statistics_come_from_training_features():
 d=pd.read_csv(CLEAN_TRAIN_PATH); m=joblib.load(SVR_MODEL_PATH); assert np.allclose(m.named_steps['scaler'].mean_,d[FEATURES].mean().values,rtol=1e-7,atol=1e-7)
def test_compare_models_reports_real_metrics_and_ranks_r2():
 from main import compare_models
 X,y=heldback(); g=compare_models(SVR_MODEL_PATH,FOREST_MODEL_PATH,X,y); assert list(g.columns)==['model_name','rmse','mae','r2_score']; assert set(g.model_name)=={'SVR','RandomForestRegressor'}; assert g.r2_score.tolist()==sorted(g.r2_score,reverse=True)
 for _,r in g.iterrows():
  p=SVR_MODEL_PATH if r.model_name=='SVR' else FOREST_MODEL_PATH; assert np.allclose([r.rmse,r.mae,r.r2_score],metrics(joblib.load(p),X,y),rtol=1e-7,atol=1e-7)
def test_compare_models_uses_passed_batch_not_disk():
 from main import compare_models
 X,y=heldback(); a=compare_models(SVR_MODEL_PATH,FOREST_MODEL_PATH,X.iloc[:100],y.iloc[:100]); b=compare_models(SVR_MODEL_PATH,FOREST_MODEL_PATH,X.iloc[100:200],y.iloc[100:200]); assert not np.allclose(a[['rmse','mae','r2_score']],b[['rmse','mae','r2_score']])
def test_evaluate_model_matches_manual_metrics():
 from main import evaluate_model
 X,y=heldback(); assert np.allclose(evaluate_model(SVR_MODEL_PATH,X,y),metrics(joblib.load(SVR_MODEL_PATH),X,y),rtol=1e-7,atol=1e-7)
def test_evaluate_model_respects_model_path():
 from main import evaluate_model
 X,y=heldback(); assert not np.allclose(evaluate_model(SVR_MODEL_PATH,X,y),evaluate_model(FOREST_MODEL_PATH,X,y))
def test_prediction_covers_every_turbine_and_preserves_ids(tmp_path):
 from main import predict_new_data
 o=predict_new_data(SVR_MODEL_PATH,RAW_PREDICT_PATH,str(tmp_path/'p.csv'),str(tmp_path/'o.csv')); r=pd.read_csv(RAW_PREDICT_PATH); assert list(o.columns)==['turbine_id','predicted_annual_maintenance_cost_lakh']; assert len(o)==len(r)==800; assert o.turbine_id.tolist()==r.turbine_id.tolist(); assert np.isfinite(o.predicted_annual_maintenance_cost_lakh).all(); assert pd.read_csv(tmp_path/'o.csv').shape==(800,2)
def test_prediction_values_come_from_stored_model(tmp_path):
 from main import predict_new_data
 o=predict_new_data(SVR_MODEL_PATH,RAW_PREDICT_PATH,str(tmp_path/'p.csv'),str(tmp_path/'o.csv')); p=pd.read_csv(tmp_path/'p.csv'); assert np.allclose(o.predicted_annual_maintenance_cost_lakh,joblib.load(SVR_MODEL_PATH).predict(p),rtol=1e-7,atol=1e-7)
