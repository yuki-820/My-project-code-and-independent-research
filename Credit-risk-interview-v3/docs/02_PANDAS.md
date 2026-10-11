# pandas：把SQL思路搬进Python

DataFrame就是内存里一张带列名的表，Series通常是一列。先理解每一步输入和输出是什么，再背函数。运行pandas_lab.py可查看八题真实输出；pandas_exercises.py留给你自己填写，不要直接复制答案。

## P01 筛选：先做“每行是否留下”的布尔列

目标：申请通过、金额至少10万，输出申请ID、客户ID和申请金额，按申请ID排。

```python
a = d['loan_applications']
mask = (a['approval_status'] == 'Approved') & (a['requested_amount'] >= 100000)
out = a.loc[mask, ['application_id','customer_id','requested_amount']]
out = out.sort_values('application_id').reset_index(drop=True)
```

a['approval_status']=='Approved'得到一串True/False，不是一个真假值。&逐行做“且”，|逐行做“或”，~做“非”。Python的and/or用于单个真假判断，不能直接替代Series的&/|。两边要加括号，避免运算优先级出错。loc[行条件,列清单]同时控制留下哪些行和列。reset_index只重置显示索引，不是改申请ID。

练习变式：把“且”换成“或”，先猜行数变大还是变小。自检：所有输出行通过且>=100000，列只有三列，原表未被改动。

## P02 分组：agg汇总成每支行一行

```python
a = d['loan_applications'].copy()
a['approved_flag'] = a['approval_status'].eq('Approved').astype(int)
out = a.groupby('branch_id',as_index=False).agg(
    applications=('application_id','size'),
    approved=('approved_flag','sum')
)
out['approval_rate_pct'] = 100*out['approved']/out['applications']
```

copy是为了不意外改动外部共用表。eq相当于==，astype(int)把True/False变1/0。agg里的“新列名=(旧列名,函数)”叫命名聚合。as_index=False让branch_id仍是普通列，后面merge更方便。

size数行，count数非空，区别对应COUNT(*)与COUNT(字段)。缺失支行要保留时写dropna=False。不要把支行通过率直接平均得到全局通过率，应加总通过数再除总申请数。

## P03 合并：merge与LEFT JOIN

```python
out = d['loans'].merge(
    d['customers'][['customer_id','credit_score']],
    on='customer_id',how='left',validate='many_to_one',indicator=True
)
assert len(out) == len(d['loans'])
assert out['_merge'].eq('both').all()
```

左边是一笔一行的贷款，右边是一人一行的客户，所以many_to_one。validate会在右表客户ID重复时直接报错，避免静默扩行。indicator生成_merge列，both表示匹配上，left_only表示左边有、右边找不到。不能只看行数不变就认为都匹配：LEFT JOIN即使一个都没匹配，行数也能不变。

pandas会匹配两边都为空的连接键，这与通常SQL不同。业务ID为空时应先显式处理，再合并，不能默认结果等价。

## P04 先聚合再合并：避免一对多扩行

```python
r = d['repayments'].groupby('loan_id',as_index=False).agg(total_paid=('amount_paid','sum'))
out = d['loans'][['loan_id','loan_amount']].merge(r,on='loan_id',how='left',validate='one_to_one')
out['total_paid'] = out['total_paid'].fillna(0)
```

每笔贷款可能几十期还款，先合成一笔一个total_paid，再与贷款连接。本题缺少还款匹配才以0作为教学口径；真实数据必须区别“确实未支付”与“流水系统缺数”。全空组的sum默认可能给0，若要保持缺失需min_count=1并设计后续处理。

验收：贷款行数6112不变，本金总额不变，总实还与原repayments的SUM一致。输出贷款本金不应重复乘以分期数。

## P05 最新一条：排序后drop_duplicates

```python
a = d['loan_applications'].copy()
a['application_date'] = pd.to_datetime(a['application_date'])
a = a.sort_values(['customer_id','application_date','application_id'],ascending=[True,False,False])
out = a.drop_duplicates('customer_id',keep='first')
```

先把时间转成真正日期。按客户正序、时间倒序、ID倒序排列，再每人留第一行。drop_duplicates不是“智能取最新”，它只按当前排列保留第一条；漏掉排序就会答错。

SQL写法是ROW_NUMBER PARTITION BY customer_id ORDER BY application_date DESC,application_id DESC。口试：如果同一天申请两次怎么办？解释第二排序键。

## P06 transform：保留明细的“窗口”

```python
out = d['loans'][['loan_id','branch_id','loan_amount']].copy()
out['branch_total'] = out.groupby('branch_id')['loan_amount'].transform('sum')
out['share'] = out['loan_amount']/out['branch_total']
```

agg返回每组一个结果，transform把每组结果广播回组内每一行。三笔100、200、300，其中前两笔在A、后一笔在B，则branch_total三行都是300，share分别1/3、2/3、1。

面试答法：算支行总额用agg；算“每笔贷款占所在支行的比重”还要保留loan_id，用transform更直接。也可agg后merge，但多一步连接与校验。

## P07 pivot和pivot_table：行列交叉汇总

```python
out = d['loans'].pivot_table(index='branch_id',columns='loan_type',
                            values='loan_amount',aggfunc='sum',fill_value=0)
```

index决定每行是哪个支行，columns决定每列是哪种产品，values是格子里统计的变量，aggfunc='sum'指定求和。默认聚合常为mean，省略会变成平均贷款额。fill_value=0只用于这里“该支行没有该类型贷款”的空格。

pivot只改变布局，要求同一行键+列键唯一。一个支行有多笔同类贷款时会冲突；pivot_table可以先聚合。所以不要用drop_duplicates强行消灭重复，否则丢钱。

练习：手写三行(A,home,100)、(A,home,200)、(A,auto,50)，透视后home=300、auto=50。SQL等价写法为SUM(CASE WHEN loan_type='home' THEN loan_amount ELSE 0 END)。

## P08 分箱再聚合

```python
out['band'] = pd.cut(out['credit_score'],
    bins=[float('-inf'),600,700,750,800,float('inf')],right=False,
    labels=['<600','600-699','700-749','750-799','800+'])
out['bad'] = out['status'].eq('defaulted').astype(int)
g = out.groupby('band',observed=True,as_index=False).agg(loans=('loan_id','size'),defaults=('bad','sum'))
g['default_rate_pct'] = 100*g['defaults']/g['loans']
```

right=False表示左闭右开，所以600属于[600,700)，不是<600。observed=True只输出实际出现的类别，显式设置避免版本默认变化。若有缺失分数，不能默默丢掉，要另设缺失组。

## 额外口试：pandas与SQL怎么选

“我倾向在数据库用SQL过滤、聚合和关联，把数据降到需要的规模，再用pandas做进一步检查、可视化和建模。pandas受单机内存限制，SQL由数据库优化器决定执行计划。两者都要注意粒度、空键、重复和排序，语法互译不代表口径自动等价。”

别回答“pandas一定更快”或“SQL一定更好”。数据位置、规模、索引、连接基数和任务类型决定方案。大表先读必要列，不要for循环逐行修改能向量化的简单运算。

参考：pandas官方merge、groupby、pivot_table文档。完整代码在pandas_lab.py；此章示例与说明为独立编写。
