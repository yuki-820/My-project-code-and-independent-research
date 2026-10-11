"""P01-P08 reference implementations on the author's unchanged data."""
from pathlib import Path
import pandas as pd
R=Path(__file__).resolve().parent
def load():
    return {t:pd.read_csv(R/'data/original'/f'{t}.csv') for t in
            ['customers','branches','loan_applications','loans','repayments']}
def P01(d):
    a=d['loan_applications']
    mask=(a['approval_status']=='Approved') & (a['requested_amount']>=100000)
    return a.loc[mask,['application_id','customer_id','requested_amount']].sort_values('application_id').reset_index(drop=True)
def P02(d):
    a=d['loan_applications'].copy()
    a['approved_flag']=a['approval_status'].eq('Approved').astype(int)
    out=a.groupby('branch_id',as_index=False).agg(applications=('application_id','size'),approved=('approved_flag','sum'))
    out['approval_rate_pct']=100*out['approved']/out['applications']
    return out.sort_values('branch_id').reset_index(drop=True)
def P03(d):
    out=d['loans'].merge(d['customers'][['customer_id','credit_score']],on='customer_id',how='left',validate='many_to_one',indicator=True)
    assert len(out)==len(d['loans'])
    assert out['_merge'].eq('both').all()
    return out[['loan_id','loan_amount','credit_score']].sort_values('loan_id').reset_index(drop=True)
def P04(d):
    r=d['repayments'].groupby('loan_id',as_index=False).agg(total_paid=('amount_paid','sum'))
    out=d['loans'][['loan_id','loan_amount']].merge(r,on='loan_id',how='left',validate='one_to_one')
    out['total_paid']=out['total_paid'].fillna(0)
    return out.sort_values('loan_id').reset_index(drop=True)
def P05(d):
    a=d['loan_applications'].copy()
    a['application_date']=pd.to_datetime(a['application_date'])
    a=a.sort_values(['customer_id','application_date','application_id'],ascending=[True,False,False])
    out=a.drop_duplicates('customer_id',keep='first')
    return out[['customer_id','application_id','application_date','approval_status']].sort_values('customer_id').reset_index(drop=True)
def P06(d):
    out=d['loans'][['loan_id','branch_id','loan_amount']].copy()
    out['branch_total']=out.groupby('branch_id')['loan_amount'].transform('sum')
    out['share']=out['loan_amount']/out['branch_total']
    return out.sort_values('loan_id').reset_index(drop=True)
def P07(d):
    # pivot requires unique row/column pairs; pivot_table aggregates duplicate pairs.
    out=d['loans'].pivot_table(index='branch_id',columns='loan_type',values='loan_amount',aggfunc='sum',fill_value=0)
    return out.sort_index().sort_index(axis=1)
def P08(d):
    out=d['loans'].merge(d['customers'][['customer_id','credit_score']],on='customer_id',validate='many_to_one')
    out['band']=pd.cut(out['credit_score'],bins=[float('-inf'),600,700,750,800,float('inf')],right=False,
                       labels=['<600','600-699','700-749','750-799','800+'])
    out['bad']=out['status'].eq('defaulted').astype(int)
    g=out.groupby('band',observed=True,as_index=False).agg(loans=('loan_id','size'),defaults=('bad','sum'))
    g['default_rate_pct']=(100*g['defaults']/g['loans']).round(2)
    return g
if __name__=='__main__':
    d=load();dest=R/'reports/pandas';dest.mkdir(parents=True,exist_ok=True)
    for i in range(1,9):
        key=f'P{i:02}';out=globals()[key](d);out.to_csv(dest/f'{key}.csv',index=key=='P07')
        print(key,'rows=',len(out));print(out.head(3).to_string(index=key=='P07'))
