# 可逐格运行的代码参考

建议先在train.ipynb中自己尝试，再展开本章。以下代码来自已运行的项目文件。

## P01 完整参考

```python
def P01(d):
    a=d['loan_applications']
    mask=(a['approval_status']=='Approved') & (a['requested_amount']>=100000)
    return a.loc[mask,['application_id','customer_id','requested_amount']].sort_values('application_id').reset_index(drop=True)

```

## P02 完整参考

```python
def P02(d):
    a=d['loan_applications'].copy()
    a['approved_flag']=a['approval_status'].eq('Approved').astype(int)
    out=a.groupby('branch_id',as_index=False).agg(applications=('application_id','size'),approved=('approved_flag','sum'))
    out['approval_rate_pct']=100*out['approved']/out['applications']
    return out.sort_values('branch_id').reset_index(drop=True)

```

## P03 完整参考

```python
def P03(d):
    out=d['loans'].merge(d['customers'][['customer_id','credit_score']],on='customer_id',how='left',validate='many_to_one',indicator=True)
    assert len(out)==len(d['loans'])
    assert out['_merge'].eq('both').all()
    return out[['loan_id','loan_amount','credit_score']].sort_values('loan_id').reset_index(drop=True)

```

## P04 完整参考

```python
def P04(d):
    r=d['repayments'].groupby('loan_id',as_index=False).agg(total_paid=('amount_paid','sum'))
    out=d['loans'][['loan_id','loan_amount']].merge(r,on='loan_id',how='left',validate='one_to_one')
    out['total_paid']=out['total_paid'].fillna(0)
    return out.sort_values('loan_id').reset_index(drop=True)

```

## P05 完整参考

```python
def P05(d):
    a=d['loan_applications'].copy()
    a['application_date']=pd.to_datetime(a['application_date'])
    a=a.sort_values(['customer_id','application_date','application_id'],ascending=[True,False,False])
    out=a.drop_duplicates('customer_id',keep='first')
    return out[['customer_id','application_id','application_date','approval_status']].sort_values('customer_id').reset_index(drop=True)

```

## P06 完整参考

```python
def P06(d):
    out=d['loans'][['loan_id','branch_id','loan_amount']].copy()
    out['branch_total']=out.groupby('branch_id')['loan_amount'].transform('sum')
    out['share']=out['loan_amount']/out['branch_total']
    return out.sort_values('loan_id').reset_index(drop=True)

```

## P07 完整参考

```python
def P07(d):
    # pivot requires unique row/column pairs; pivot_table aggregates duplicate pairs.
    out=d['loans'].pivot_table(index='branch_id',columns='loan_type',values='loan_amount',aggfunc='sum',fill_value=0)
    return out.sort_index().sort_index(axis=1)

```

## P08 完整参考

```python
def P08(d):
    out=d['loans'].merge(d['customers'][['customer_id','credit_score']],on='customer_id',validate='many_to_one')
    out['band']=pd.cut(out['credit_score'],bins=[float('-inf'),600,700,750,800,float('inf')],right=False,
                       labels=['<600','600-699','700-749','750-799','800+'])
    out['bad']=out['status'].eq('defaulted').astype(int)
    g=out.groupby('band',observed=True,as_index=False).agg(loans=('loan_id','size'),defaults=('bad','sum'))
    g['default_rate_pct']=(100*g['defaults']/g['loans']).round(2)
    return g

```

## A01 完整参考

```python
def three_sum(nums):
    a=sorted(nums)                     # sorted copy, keep input unchanged
    out=[]
    for i in range(len(a)-2):
        if i>0 and a[i]==a[i-1]:continue
        if a[i]>0:break                # remaining numbers cannot sum to zero
        left,right=i+1,len(a)-1
        while left<right:
            total=a[i]+a[left]+a[right]
            if total<0:left+=1
            elif total>0:right-=1
            else:
                out.append([a[i],a[left],a[right]])
                left+=1;right-=1
                while left<right and a[left]==a[left-1]:left+=1
                while left<right and a[right]==a[right+1]:right-=1
    return out

```

## A02 完整参考

```python
def quick_sort(nums):
    # Pedagogical three-way version, not in-place. Choose a random pivot in production
    # if adversarial inputs matter; Python's built-in sorted is usually preferable.
    if len(nums)<=1:return list(nums)
    pivot=nums[len(nums)//2]
    less=[x for x in nums if x<pivot]
    equal=[x for x in nums if x==pivot]
    greater=[x for x in nums if x>pivot]
    return quick_sort(less)+equal+quick_sort(greater)

```

## A03 完整参考

```python
def max_below(n,digits):
    """Largest nonnegative integer < n, unlimited digit reuse, no leading zeros.
    Zero is allowed only if 0 is in digits. None means no feasible answer.
    At most 10 allowed digits. Greedy equal-prefix walk with backtracking.
    """
    if not isinstance(n,int):raise TypeError('n must be an integer')
    if any(not isinstance(d,int) or not 0<=d<=9 for d in digits):raise ValueError('digits must be 0..9')
    allowed=sorted(set(digits),reverse=True)
    if n<=0 or not allowed:return None
    target=[int(x) for x in str(n)];L=len(target)
    def search(pos,prefix):
        if pos==L:return None           # identical n is not strictly smaller
        for d in allowed:
            if pos==0 and L>1 and d==0:continue
            if d>target[pos]:continue
            if d<target[pos]:
                return prefix+[d]+[allowed[0]]*(L-pos-1)
            result=search(pos+1,prefix+[d])
            if result is not None:return result
        return None
    out=search(0,[])
    if out is not None:return int(''.join(map(str,out)))
    for length in range(L-1,0,-1):
        first=next((d for d in allowed if d>0 or length==1),None)
        if first is not None:return int(str(first)+str(allowed[0])*(length-1))
    return None

```

## I01 完整SQL

```sql
-- Per-video valid-play completion rate. Deduplicate exact repeated event reports.
WITH clean AS (
 SELECT DISTINCT event_id,video_id,watch_sec FROM plays
 WHERE watch_sec IS NOT NULL AND watch_sec>=0
)
SELECT v.video_id,COUNT(c.event_id) AS valid_plays,
 SUM(CASE WHEN c.watch_sec>=v.duration_sec THEN 1 ELSE 0 END) AS complete_plays,
 ROUND(100.0*SUM(CASE WHEN c.watch_sec>=v.duration_sec THEN 1 ELSE 0 END)
       /NULLIF(COUNT(c.event_id),0),2) AS completion_rate_pct
FROM videos v LEFT JOIN clean c ON v.video_id=c.video_id
WHERE v.duration_sec>0
GROUP BY v.video_id ORDER BY v.video_id;

```

## I02 完整SQL

```sql
-- A duplicate login on the same date must count only once.
WITH days AS (SELECT DISTINCT user_id,login_date FROM logins),
numbered AS (
 SELECT user_id,login_date,
 ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY login_date) AS rn FROM days
), islands AS (
 SELECT user_id,login_date,DATE(login_date,'-'||rn||' days') AS island FROM numbered
)
SELECT user_id,MIN(login_date) AS start_date,MAX(login_date) AS end_date,COUNT(*) AS consecutive_days
FROM islands GROUP BY user_id,island HAVING COUNT(*)>=3 ORDER BY user_id,start_date;

```

## I03 完整SQL

```sql
-- Paid gross GMV, includes today and previous 6 calendar days. No refund table in fixture.
WITH RECURSIVE calendar(day) AS (
 SELECT '2026-10-01'
 UNION ALL SELECT DATE(day,'+1 day') FROM calendar WHERE day<'2026-10-10'
), daily AS (
 SELECT paid_date,SUM(amount) AS gmv FROM orders WHERE status='paid' GROUP BY paid_date
), filled AS (
 SELECT c.day,COALESCE(d.gmv,0) AS gmv FROM calendar c LEFT JOIN daily d ON c.day=d.paid_date
)
SELECT day,gmv,SUM(gmv) OVER(ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS gmv_7d
FROM filled ORDER BY day;

```
