"""Biomedical System Design Capsule - Machine Learning Lab 1.
Recreates experiments A-D from two bundled scikit-learn biomedical datasets.
All fitted scalers use training data only; the test sets are never used for fitting.
"""
from __future__ import annotations
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

BASE = Path(__file__).resolve().parents[1]
FIG = BASE/'results'/'figures'; DATA=BASE/'results'/'data'
FIG.mkdir(parents=True,exist_ok=True); DATA.mkdir(parents=True,exist_ok=True)
SEED=42
plt.rcParams.update({'font.size':10, 'axes.spines.top':False,
    'axes.spines.right':False,'axes.titlesize':12,'figure.dpi':130,
    'savefig.dpi':200,'savefig.bbox':'tight', 'font.family':'DejaVu Sans'})
C1='#22628a'; C2='#ce7f34'; C3='#16816a'; C4='#ad4d5e'

def save(name):
    plt.savefig(FIG/name, facecolor='white');plt.close()

def calc_cls(y,p,thr):
    pred=(p>=thr).astype(int)
    cm=confusion_matrix(y,pred,labels=[0,1]);tn,fp,fn,tp=cm.ravel()
    return {'threshold':float(thr),'accuracy':float(accuracy_score(y,pred)),
        'precision':float(precision_score(y,pred,zero_division=0)),
        'recall':float(recall_score(y,pred,zero_division=0)),
        'f1':float(f1_score(y,pred,zero_division=0)),
        'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

# A / B: Regression (BMI, mean blood pressure -> 1-year diabetes progression)
reg=load_diabetes(scaled=False,as_frame=True)
reg_features=['bmi','bp']; Xr=reg.data[reg_features];yr=reg.target
train_ir,test_ir=train_test_split(np.arange(len(yr)),test_size=.2,random_state=SEED)
Xrtr=Xr.iloc[train_ir];Xrte=Xr.iloc[test_ir]; yrtr=yr.iloc[train_ir];yrte=yr.iloc[test_ir]
ols=LinearRegression().fit(Xrtr,yrtr); pred_ols=ols.predict(Xrte)
ols_scaled=make_pipeline(StandardScaler(),LinearRegression()).fit(Xrtr,yrtr)
pred_scaled=ols_scaled.predict(Xrte)
std_reg=ols_scaled.named_steps['standardscaler']; std_ols=ols_scaled.named_steps['linearregression']
rmse=float(np.sqrt(mean_squared_error(yrte,pred_ols)))
metrics_reg={'mae':float(mean_absolute_error(yrte,pred_ols)),
             'mse':float(mean_squared_error(yrte,pred_ols)),'rmse':rmse,
             'r2':float(r2_score(yrte,pred_ols))}
metrics_scaled={'mae':float(mean_absolute_error(yrte,pred_scaled)),
             'mse':float(mean_squared_error(yrte,pred_scaled)),
             'r2':float(r2_score(yrte,pred_scaled))}
fig,axs=plt.subplots(1,2,figsize=(10,3.7))
for j,(f,xlabel) in enumerate(zip(reg_features,['Body mass index (BMI)','Average blood pressure'])):
    axs[j].scatter(Xr[f],yr,s=13,alpha=.5,c=C1,edgecolor='none')
    axs[j].set(xlabel=xlabel,ylabel='One-year disease progression',title=f'{f.upper()} vs. outcome (n=442)')
    axs[j].grid(alpha=.15)
fig.tight_layout(); save('fig01_diabetes_exploration.png')
fig,ax=plt.subplots(figsize=(6,4.4))
ax.scatter(yrte,pred_ols,s=25,alpha=.7,c=C1,edgecolors='white',linewidth=.4)
lo=min(float(yrte.min()),float(pred_ols.min()));hi=max(float(yrte.max()),float(pred_ols.max()))
ax.plot([lo,hi],[lo,hi],ls='--',color=C4,label='Perfect prediction')
ax.set(xlabel='Observed progression score',ylabel='Predicted progression score',title='B1  |  Predictions on held-out test data')
ax.text(.03,.96,f"MAE = {metrics_reg['mae']:.2f}   R² = {metrics_reg['r2']:.3f}", transform=ax.transAxes,va='top')
ax.legend(loc='lower right'); ax.grid(alpha=.17);fig.tight_layout();save('fig02_regression_actual_predicted.png')
fig,axs=plt.subplots(1,2,figsize=(10,3.8))
axs[0].bar(['Without scaling','Standardized'],[metrics_reg['mae'],metrics_scaled['mae']],color=[C1,C3],width=.6)
axs[0].set(ylabel='Test MAE (progression units)',title='B2  |  Test error')
for i,v in enumerate([metrics_reg['mae'],metrics_scaled['mae']]): axs[0].text(i,v+.25,f'{v:.3f}',ha='center')
axs[0].set_ylim(0,metrics_reg['mae']*1.22)
x=np.arange(2);w=.34
axs[1].bar(x-w/2,ols.coef_,width=w,label='Raw-unit coefficients',color=C1)
axs[1].bar(x+w/2,std_ols.coef_,width=w,label='Per-SD coefficients',color=C3)
axs[1].set_xticks(x,['BMI','Blood pressure']);axs[1].set(ylabel='Coefficient (different units)',title='Coefficient magnitudes are not directly comparable')
axs[1].legend(fontsize=8);axs[1].axhline(0,color='#444',lw=.6)
fig.tight_layout();save('fig03_scaling_comparison.png')

# B3 gradient descent: standardize training predictors and center y, then gradient descent.
# Loss J = (1/(2n)) * ||X w + b - (y - mean(y_train))||^2
Xgd=std_reg.transform(Xrtr); ygd=yrtr.to_numpy()-float(yrtr.mean())
Z=np.column_stack([np.ones(len(Xgd)),Xgd]);
theta_star=np.linalg.lstsq(Z,ygd,rcond=None)[0]
minloss=float(.5*np.mean((Z@theta_star-ygd)**2))
learning_rates=[0.01,0.1,0.8,1.5]; iterations=150
GD={};fig,ax=plt.subplots(figsize=(8.1,4.7))
for lr,col in zip(learning_rates,[C1,C3,C2,C4]):
    theta=np.zeros(Z.shape[1]);trace=[]
    for i in range(iterations):
        resid=Z@theta-ygd
        grad=Z.T@resid/len(ygd)
        theta=theta-lr*grad
        loss=float(.5*np.mean((Z@theta-ygd)**2));trace.append(loss)
    within=next((i+1 for i,v in enumerate(trace) if v<=minloss*1.01),None)
    GD[str(lr)]={'loss_first':float(trace[0]),'loss_10':float(trace[9]),
                 'loss_50':float(trace[49]),'loss_final':float(trace[-1]),
                 'first_iteration_within_1pct_of_optimum':within}
    ax.plot(np.arange(1,iterations+1),trace,linewidth=1.65,label=f'η = {lr:g}',color=col)
ax.axhline(minloss,linestyle='--',color='#4c5155',linewidth=1.1,label=f'OLS minimum = {minloss:.1f}')
ax.set_yscale('log');ax.set(xlabel='Gradient-descent iteration',ylabel='Training loss J (log scale)',title='B3  |  Learning rate and convergence (150 iterations)')
ax.legend(ncol=2); ax.grid(alpha=.22,which='both');fig.tight_layout();save('fig04_learning_rate_convergence.png')

# A / C: Breast cancer diagnostic data, malignant is the positive class.
bc=load_breast_cancer(as_frame=True)
cls_features=['worst radius','worst concave points'];Xc=bc.data[cls_features]
yc=pd.Series((bc.target.to_numpy()==0).astype(int),name='malignant')
train_ic,test_ic=train_test_split(np.arange(len(yc)),test_size=.2,random_state=SEED,stratify=yc)
Xctr=Xc.iloc[train_ic];Xcte=Xc.iloc[test_ic]; yctr=yc.iloc[train_ic]; ycte=yc.iloc[test_ic]
cls=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,solver='lbfgs',max_iter=2000,random_state=SEED)).fit(Xctr,yctr)
prob=cls.predict_proba(Xcte)[:,1]
cls_scale=cls.named_steps['standardscaler'];cls_est=cls.named_steps['logisticregression']
one=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,solver='lbfgs',max_iter=2000,random_state=SEED))
one.fit(Xctr[[cls_features[0]]],yctr)
prob_one=one.predict_proba(Xcte[[cls_features[0]]])[:,1]
base_cls=calc_cls(ycte,prob,0.5)
threshold_results=[calc_cls(ycte,prob,t) for t in [0.3,0.5,0.7]]
one_result=calc_cls(ycte,prob_one,.5)
fig,ax=plt.subplots(figsize=(7.1,4.5))
for lab,color,label in [(0,C3,'Benign'),(1,C4,'Malignant')]:
    sel=(yc==lab)
    ax.scatter(Xc.loc[sel,cls_features[0]],Xc.loc[sel,cls_features[1]],s=16,alpha=.65,c=color,label=f'{label} (n={sel.sum()})',edgecolor='none')
ax.set(xlabel='Worst radius',ylabel='Worst concave points',title='Dataset 2  |  Two-feature class distribution')
ax.legend();ax.grid(alpha=.15);fig.tight_layout();save('fig05_breast_cancer_exploration.png')
cm=np.asarray([[base_cls['tn'],base_cls['fp']],[base_cls['fn'],base_cls['tp']]])
fig,ax=plt.subplots(figsize=(5.8,4.4));im=ax.imshow(cm,cmap='Blues',vmin=0)
for (r,c),v in np.ndenumerate(cm):
    ax.text(c,r,str(v),ha='center',va='center',fontsize=15,fontweight='bold',color='white' if v>cm.max()*.55 else '#13314d')
ax.set_xticks([0,1],['Benign (0)','Malignant (1)']);ax.set_yticks([0,1],['Benign (0)','Malignant (1)'])
ax.set(xlabel='Predicted class',ylabel='True class',title='C1  |  Confusion matrix at threshold 0.5')
fig.colorbar(im,ax=ax,label='Number of test observations',shrink=.85);fig.tight_layout();save('fig06_confusion_matrix.png')
fig,ax=plt.subplots(figsize=(6.7,4.2))
vals=[z['threshold'] for z in threshold_results]
for lab,col,k in [('Precision',C1,'precision'),('Recall',C4,'recall')]:
    yvals=[z[k] for z in threshold_results]
    ax.plot(vals,yvals,'o-',lw=2,markersize=7,label=lab,color=col)
    for xx,yy in zip(vals,yvals): ax.annotate(f'{yy:.3f}',(xx,yy),xytext=(0,(-16 if k=='recall' else 10)),textcoords='offset points',ha='center',fontsize=9)
ax.set(xlabel='Probability threshold',ylabel='Test-set score',title='C2  |  Precision-recall threshold trade-off',xticks=vals,ylim=(0.66,1.055))
ax.legend();ax.grid(alpha=.18);fig.tight_layout();save('fig07_thresholds.png')
# Decision boundary in standardized coordinates with only the training-fitted scaler.
Xc_std=cls_scale.transform(Xc)
x0lo,x0hi=np.percentile(Xc_std[:,0],[0,100]); x1lo,x1hi=np.percentile(Xc_std[:,1],[0,100]);
a=np.linspace(x0lo-.5,x0hi+.5,230);b=np.linspace(x1lo-.5,x1hi+.5,230)
A,B=np.meshgrid(a,b)
P=cls_est.predict_proba(np.column_stack([A.ravel(),B.ravel()]))[:,1].reshape(A.shape)
fig,ax=plt.subplots(figsize=(7.6,5.0))
ax.contourf(A,B,P,levels=[0,.2,.4,.5,.6,.8,1.0],cmap='RdYlBu_r',alpha=.32,extend='both')
ax.contour(A,B,P,levels=[.5],colors=['#263b55'],linewidths=2)
# Test data only: observed diagnosis and prediction at 0.5
Xte_s=cls_scale.transform(Xcte)
for lab,col,txt in [(0,C3,'Benign (test)'),(1,C4,'Malignant (test)')]:
    mask=ycte.to_numpy()==lab
    ax.scatter(Xte_s[mask,0],Xte_s[mask,1],s=36,color=col,edgecolor='white',linewidth=.5,alpha=.85,label=txt)
wrong=((prob>=.5).astype(int)!=ycte.to_numpy())
if np.any(wrong): ax.scatter(Xte_s[wrong,0],Xte_s[wrong,1],facecolors='none',edgecolors='black',marker='o',s=115,linewidth=1.8,label='Misclassified')
ax.set(xlabel='Standardized worst radius (training-set z score)',ylabel='Standardized worst concave points (training-set z score)',title='C3  |  P(malignant) = 0.5 decision boundary')
ax.legend(fontsize=8,loc='lower right');ax.grid(alpha=.11);fig.tight_layout();save('fig08_decision_boundary.png')
fig,axs=plt.subplots(1,2,figsize=(9,3.6))
for i,(metric,label) in enumerate([('accuracy','Accuracy'),('recall','Malignant recall')]):
    v=[one_result[metric],base_cls[metric]];axs[i].bar(['Worst radius\nonly','Both features'],v,color=[C1,C3],width=.58)
    axs[i].set(ylim=(0,1.09),ylabel=label,title=f'C3  |  {label} on the same test split')
    for j,val in enumerate(v):axs[i].text(j,val+.015,f'{val:.3f}',ha='center')
fig.tight_layout();save('fig09_one_vs_two_features.png')

# Explicit train/test indices, preserving exact experimental data and decisions.
reg_record=Xr.copy();reg_record['target_progression']=yr.to_numpy();reg_record['split']='test';reg_record.loc[train_ir,'split']='train'
cls_record=Xc.copy();cls_record['malignant_positive']=yc.to_numpy();cls_record['split']='test';cls_record.loc[train_ic,'split']='train'
reg_record.to_csv(DATA/'diabetes_two_feature_split.csv',index_label='row_index')
cls_record.to_csv(DATA/'breast_cancer_two_feature_split.csv',index_label='row_index')
res={'meta':{'project':'Biomedical System Design Capsule - ML Lab 1','student':'Burak Bekir Çakırer',
    'student_id':'2211011008','seed':SEED,'test_fraction':.2,
    'python':platform.python_version(),'scikit_learn':sklearn_version,
    'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__},
  'regression':{'dataset':'scikit-learn diabetes','n':len(Xr),'n_train':len(train_ir),'n_test':len(test_ir),
       'features':reg_features,'target':'one-year disease progression score',
       'raw_coef':dict(zip(reg_features,[float(t) for t in ols.coef_])),
       'raw_intercept':float(ols.intercept_), 'test':metrics_reg,
       'scaled_coef':dict(zip(reg_features,[float(t) for t in std_ols.coef_])),
       'scaled_intercept':float(std_ols.intercept_), 'scaled_test':metrics_scaled,
       'raw_feature_training_mean':dict(zip(reg_features,[float(t) for t in std_reg.mean_])),
       'raw_feature_training_std':dict(zip(reg_features,[float(t) for t in std_reg.scale_])),
       'max_prediction_difference_scaled_unscaled':float(np.max(np.abs(pred_ols-pred_scaled))),
       'gd':{'objective':'J(theta)=sum_i(yhat_i-ycenter_i)^2/(2*n)', 'n_iterations':iterations,
             'min_train_loss':minloss,'learning_rates':GD}},
  'classification':{'dataset':'Wisconsin diagnostic breast cancer','n':len(Xc),'n_train':len(train_ic),'n_test':len(test_ic),
        'n_malignant':int(yc.sum()),'n_benign':int((yc==0).sum()),
        'features':cls_features,'class_encoding':{'malignant':1,'benign':0},
        'standardized_coef':dict(zip(cls_features,[float(t) for t in cls_est.coef_[0]])),
        'intercept':float(cls_est.intercept_[0]),'test_at_0_5':base_cls,
        'thresholds':threshold_results,'one_feature_test_at_0_5':one_result,
        'logistic_settings':{'C':1.0,'solver':'lbfgs','max_iter':2000,'class_weight':None}}}
(DATA/'metrics.json').write_text(json.dumps(res,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'linear':res['regression']['test'],'coefficients':res['regression']['raw_coef'],
                  'logistic':base_cls,'thresholds':threshold_results,
                  'one_feature':one_result,'gd':res['regression']['gd']},indent=2))
