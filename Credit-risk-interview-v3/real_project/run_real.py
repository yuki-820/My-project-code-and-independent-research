"""Real UCI data: quality checks -> long tables -> SQL features -> honest held-out comparison.
No rejection-policy causal claim. No claim of an out-of-time test.
"""
from pathlib import Path
import json,hashlib,sqlite3,platform,importlib.metadata
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,roc_curve,brier_score_loss
R=Path(__file__).resolve().parent

def metrics(y,p,t):
    y=np.asarray(y);flag=np.asarray(p)>=t
    tp=int(((y==1)&flag).sum());fp=int(((y==0)&flag).sum())
    fn=int(((y==1)&~flag).sum());tn=int(((y==0)&~flag).sum())
    fpr,tpr,_=roc_curve(y,p)
    return {'n':len(y),'defaults':int(y.sum()),'auc':float(roc_auc_score(y,p)),
      'ks':float(max(tpr-fpr)),'brier':float(brier_score_loss(y,p)),
      'threshold':float(t),'flagged':int(flag.sum()),'flag_rate':float(flag.mean()),
      'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn),
      'false_positive_rate':fp/(fp+tn),'retained_default_rate':fn/(fn+tn) if fn+tn else None,
      'tp':tp,'fp':fp,'fn':fn,'tn':tn}

def build():
    src=json.loads((R/'data/source.json').read_text())
    raw=R/'data/clients.csv'
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==src['csv_sha256']
    d=pd.read_csv(raw);y=d['default payment next month']
    assert len(d)==30000 and d.ID.is_unique and not d.isna().any().any()
    assert set(y)=={0,1} and (d.LIMIT_BAL>0).all()
    # IDs define disjoint observations. One cohort means random holdout, not OOT.
    train,other=train_test_split(d.ID,test_size=.4,stratify=y,random_state=42)
    indexed=d.set_index('ID')
    val,test=train_test_split(other,test_size=.5,stratify=indexed.loc[other,'default payment next month'],random_state=43)
    split={int(i):name for name,ids in [('train',train),('validation',val),('test',test)] for i in ids}
    clients=d[['ID','LIMIT_BAL']].rename(columns={'ID':'client_id','LIMIT_BAL':'credit_limit'})
    labels=pd.DataFrame({'client_id':d.ID,'default_next_month':y,'split':d.ID.map(split)})
    history=[]
    for k,status in enumerate(['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'],1):
        history.append(pd.DataFrame({'client_id':d.ID,'month':f'2005-{10-k:02}',
          'pay_status':d[status],'bill_amount':d[f'BILL_AMT{k}'],'payment_amount':d[f'PAY_AMT{k}']}))
    h=pd.concat(history,ignore_index=True)
    assert len(h)==180000 and not h.duplicated(['client_id','month']).any()
    c=sqlite3.connect(R/'data/uci.db')
    for name,df in [('clients',clients),('monthly_history',h),('labels',labels)]:df.to_sql(name,c,index=False,if_exists='replace')
    c.execute('CREATE INDEX IF NOT EXISTS history_key ON monthly_history(client_id,month)')
    x=pd.read_sql_query((R/'feature.sql').read_text(),c).set_index('client_id')
    # Independent vectorized check of every feature, not a second call to same SQL.
    pay=d[['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']]
    expected=pd.DataFrame({'credit_limit':d.LIMIT_BAL,'latest_status':d.PAY_0,
      'latest_util':d.BILL_AMT1/d.LIMIT_BAL,'worst_delay6':pay.clip(lower=0).max(axis=1),
      'delay_months6':pay.gt(0).sum(axis=1),'delay_months3':pay.iloc[:,:3].gt(0).sum(axis=1),
      'bill_trend6':(d.BILL_AMT1-d.BILL_AMT6)/d.LIMIT_BAL,
      'mean_paid_to_limit6':d[[f'PAY_AMT{i}' for i in range(1,7)]].mean(axis=1)/d.LIMIT_BAL,
      'paid_trend6':(d[['PAY_AMT1','PAY_AMT2','PAY_AMT3']].sum(axis=1)-d[['PAY_AMT4','PAY_AMT5','PAY_AMT6']].sum(axis=1))/d.LIMIT_BAL})
    expected.index=d.ID;expected.index.name='client_id'
    pd.testing.assert_frame_equal(x,expected.sort_index(),check_dtype=False,rtol=1e-9,atol=1e-10)
    assert set(train).isdisjoint(val) and set(train).isdisjoint(test) and set(val).isdisjoint(test)
    (R/'reports').mkdir(exist_ok=True)
    labels.to_csv(R/'reports/splits.csv',index=False)
    quality={'rows':len(d),'defaults':int(y.sum()),'default_rate':float(y.mean()),'missing_cells':int(d.isna().sum().sum()),
      'negative_bill_cells':int(d[[f'BILL_AMT{i}' for i in range(1,7)]].lt(0).sum().sum()),
      'status_values':sorted(map(int,np.unique(pay))),
      'education_values':sorted(map(int,d.EDUCATION.unique())),
      'marriage_values':sorted(map(int,d.MARRIAGE.unique())),
      'split_sizes':labels.groupby('split').size().to_dict(),
      'checks':['source hash','unique ID','no missing','positive limits','binary labels','unique client/month','disjoint splits','all SQL features equal pandas']}
    yy=labels.set_index('client_id').default_next_month
    numeric_base=['credit_limit','latest_util']
    numeric_full=[col for col in x if col!='latest_status']
    results={};scores={};selection=[]
    for name,num,grid in [('latest_only',numeric_base,[1.0]),('six_month_features',numeric_full,[.01,.1,1.0])]:
        best=None
        for C in grid:
            prep=ColumnTransformer([('num',StandardScaler(),num),('status',OneHotEncoder(handle_unknown='ignore'),['latest_status'])])
            model=make_pipeline(prep,LogisticRegression(C=C,max_iter=3000,solver='lbfgs',random_state=42))
            model.fit(x.loc[train],yy.loc[train])
            pv=model.predict_proba(x.loc[val])[:,1];auc=roc_auc_score(yy.loc[val],pv)
            selection.append({'model':name,'C':C,'validation_auc':float(auc)})
            if best is None or auc>best[0]:best=(auc,C,model,pv)
        auc,C,model,pv=best
        # Freeze threshold on validation scores; aim ~20% review, actual test volume may differ.
        threshold=float(np.quantile(pv,.8))
        pt=model.predict_proba(x.loc[test])[:,1]
        results[name]={'C':C,'validation':metrics(yy.loc[val],pv,threshold),'test':metrics(yy.loc[test],pt,threshold)}
        scores[name]=pt
    # Same held-out customers for a paired descriptive AUC uncertainty interval.
    rng=np.random.default_rng(20261011);yt=yy.loc[test].to_numpy();diff=[]
    for _ in range(300):
        idx=rng.integers(0,len(test),len(test))
        diff.append(roc_auc_score(yt[idx],scores['six_month_features'][idx])-roc_auc_score(yt[idx],scores['latest_only'][idx]))
    flagged=x.loc[test,'latest_status'].to_numpy()>=2
    tp=int(((yt==1)&flagged).sum());fp=int(((yt==0)&flagged).sum())
    simple={'rule':'latest_status >= 2','flag_rate':float(flagged.mean()),'flagged':int(flagged.sum()),'precision':tp/int(flagged.sum()),'recall':tp/int(yt.sum()),'tp':tp,'fp':fp}
    report={'source':src,'quality':quality,'selection':selection,'models':results,'simple_rule_test':simple,
      'paired_auc_delta_ci95_bootstrap300':list(map(float,np.quantile(diff,[.025,.975]))),
      'environment':{'python':platform.python_version(),'pandas':pd.__version__,'sklearn':importlib.metadata.version('scikit-learn')},
      'limits':['Historical 2005 Taiwan credit-card cohort, not current Kuaishou applicants','Random client holdout, not out-of-time validation',
        'Flagged = simulated review queue, not actual denial or collection action','Default next month is source label, not independently defined FPD7',
        'No causal claim of prevented default or increased approval','Sex, age, marital status, education excluded; this does not prove fairness']}
    (R/'reports/results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[]
    for name,z in results.items():rows.append({'model':name,**z['test']})
    pd.DataFrame(rows).to_csv(R/'reports/test_metrics.csv',index=False)
    pd.DataFrame(selection).to_csv(R/'reports/validation_selection.csv',index=False)
    print(json.dumps({'quality':quality,'models':results,'auc_delta_ci':report['paired_auc_delta_ci95_bootstrap300']},ensure_ascii=False,indent=2))
    c.close()
if __name__=='__main__':build()
