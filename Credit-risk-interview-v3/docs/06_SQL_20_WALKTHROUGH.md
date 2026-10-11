# CreditCheck五表SQL：20题逐题拆解

这套是作者合成数据的关系型练习，不是UCI真实主项目。先学SQL基础，再做此处练习。答案中的每个CTE都表示一次粒度变化。

## Q01 组合规模与状态违约占比

### 思路拆解

1. 从loans每笔一行起步，不接分期表。
2. COUNT计贷款，CASE把状态转成0/1，SUM得到各状态数。
3. 违约笔数除贷款总数。原始本金与当前余额分开命名。

### 参考代码

```sql
-- 组合规模与状态违约占比
-- 一行一笔贷款。原始本金之和不是当前贷款余额；违约依赖源status定义。
SELECT COUNT(*) AS total_loans, SUM(loan_amount) AS original_principal,
 SUM(CASE WHEN status='active' THEN 1 ELSE 0 END) AS active_loans,
 SUM(CASE WHEN status='defaulted' THEN 1 ELSE 0 END) AS defaulted_loans,
 ROUND(100.0*SUM(CASE WHEN status='defaulted' THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0),2) AS default_rate_pct
FROM loans;

```

### 自检和面试追问

一行一笔贷款。原始本金之和不是当前贷款余额；违约依赖源status定义。

先独立写入sql/exercises/Q01.sql，再运行python practice.py Q01。变式：如果关联键重复、分母为0或排序并列，你会怎样检查？

## Q02 申请审批通过率

### 思路拆解

1. 从全申请表起步，不先过滤通过申请。
2. 用CASE在同一张表同时计分子与分母。
3. 乘100.0防整数除法，NULLIF处理分母0。

### 参考代码

```sql
-- 申请审批通过率
-- 分母用全部申请，不能从仅有放款的loans表算通过率。
SELECT COUNT(*) AS applications,
 SUM(CASE WHEN approval_status='Approved' THEN 1 ELSE 0 END) AS approved,
 ROUND(100.0*SUM(CASE WHEN approval_status='Approved' THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0),2) AS approval_rate_pct
FROM loan_applications;

```

### 自检和面试追问

分母用全部申请，不能从仅有放款的loans表算通过率。

先独立写入sql/exercises/Q02.sql，再运行python practice.py Q02。变式：如果关联键重复、分母为0或排序并列，你会怎样检查？

## Q03 快照内分期金额偿付比

### 思路拆解

1. 在分期粒度分别加总应还和实还金额。
2. 总实还除总应还，不能先算每期率再简单平均。
3. 无统一观察日，仅描述文件快照，不称某日实时回收率。

### 参考代码

```sql
-- 快照内分期金额偿付比
-- 不是平均每期偿付率，也不是含时间截断的某日催收回款率。
SELECT ROUND(SUM(amount_due),2) AS scheduled_amount,
 ROUND(SUM(amount_paid),2) AS paid_amount,
 ROUND(100.0*SUM(amount_paid)/NULLIF(SUM(amount_due),0),2) AS paid_ratio_pct
FROM repayments;

```

### 自检和面试追问

不是平均每期偿付率，也不是含时间截断的某日催收回款率。

先独立写入sql/exercises/Q03.sql，再运行python practice.py Q03。变式：如果关联键重复、分母为0或排序并列，你会怎样检查？

## Q04 每月放款量

### 思路拆解

1. ISO日期前7个字符取年月。
2. 按月份GROUP BY统计贷款数和本金。
3. ORDER BY确保按年月展示；结果缺月需要看业务是否补0。

### 参考代码

```sql
-- 每月放款量
-- SQLite ISO日期取年月用SUBSTR；MySQL可用DATE_FORMAT。
SELECT SUBSTR(disbursement_date,1,7) AS disb_month,COUNT(*) AS loans,
 SUM(loan_amount) AS original_principal FROM loans GROUP BY disb_month ORDER BY disb_month;

```

### 自检和面试追问

SQLite ISO日期取年月用SUBSTR；MySQL可用DATE_FORMAT。

先独立写入sql/exercises/Q04.sql，再运行python practice.py Q04。变式：如果关联键重复、分母为0或排序并列，你会怎样检查？

## Q05 累计放款额

### 思路拆解

1. 先把贷款压成每月一行。
2. SUM OVER保留月度行数，附累计金额。
3. ROWS从首行到当前，显式窗口避免重复排序值的RANGE差异。

### 参考代码

```sql
-- 累计放款额
-- 先月度聚合再开窗；解释ROWS与RANGE。
WITH m AS (SELECT SUBSTR(disbursement_date,1,7) AS disb_month,SUM(loan_amount) AS amount
 FROM loans GROUP BY disb_month)
SELECT disb_month,amount,SUM(amount) OVER(ORDER BY disb_month ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_amount
FROM m ORDER BY disb_month;

```

### 自检和面试追问

先月度聚合再开窗；解释ROWS与RANGE。

先独立写入sql/exercises/Q05.sql，再运行python practice.py Q05。变式：如果关联键重复、分母为0或排序并列，你会怎样检查？

## Q06 产品类型风险分布

### 思路拆解

1. 产品类型作为分组键。
2. 同组统计笔数、金额、违约数。
3. 违约率分母是笔数，不是本金。

### 参考代码

```sql
-- 产品类型风险分布
-- 笔数率与金额率不同；没有剩余本金与回收台账，不能推算真实损失。
SELECT loan_type,COUNT(*) AS loans,SUM(loan_amount) AS original_principal,
 SUM(CASE WHEN status='defaulted' THEN 1 ELSE 0 END) AS defaults,
 ROUND(100.0*SUM(CASE WHEN status='defaulted' THEN 1 ELSE 0 END)/COUNT(*),2) AS default_rate_pct
FROM loans GROUP BY loan_type ORDER BY original_principal DESC,loan_type;

```

### 自检和面试追问

笔数率与金额率不同；没有剩余本金与回收台账，不能推算真实损失。

先独立写入sql/exercises/Q06.sql，再运行python practice.py Q06。变式：如果关联键重复、分母为0或排序并列，你会怎样检查？

## Q07 复核原作者DPD桶

### 思路拆解

1. 先从分期表按贷款聚合最大missed状态dpd。
2. LEFT JOIN保留没有missed的贷款并按约定给0。
3. 过滤closed后分桶，最后按桶汇总原始本金代理。

### 参考代码

```sql
-- 复核原作者DPD桶
-- 复现missed中最大dpd。原文90+标签实际是>90；此处修正为91+。未提供统一快照日期，不称为独立重算当前DPD。
WITH w AS (SELECT loan_id,MAX(CASE WHEN payment_status='missed' THEN dpd ELSE 0 END) AS reported_dpd
 FROM repayments GROUP BY loan_id), b AS (
 SELECT l.loan_id,l.loan_amount,
 CASE WHEN COALESCE(w.reported_dpd,0)=0 THEN '0'
 WHEN w.reported_dpd<=30 THEN '1-30' WHEN w.reported_dpd<=60 THEN '31-60'
 WHEN w.reported_dpd<=90 THEN '61-90' ELSE '91+' END AS bucket,
 COALESCE(w.reported_dpd,0) AS dpd
 FROM loans l LEFT JOIN w USING(loan_id) WHERE l.status<>'closed')
SELECT bucket,COUNT(*) AS loans,SUM(loan_amount) AS original_principal_proxy
FROM b GROUP BY bucket ORDER BY MIN(dpd);

```

### 自检和面试追问

复现missed中最大dpd。原文90+标签实际是>90；此处修正为91+。未提供统一快照日期，不称为独立重算当前DPD。

先独立写入sql/exercises/Q07.sql，再运行python practice.py Q07。变式：如果关联键重复、分母为0或排序并列，你会怎样检查？

## Q08 信用分分层

### 思路拆解

1. 将客户分数多对一接到贷款。
2. CASE按由低到高的上界分段。
3. 分段数贷款与违约，排序用实际最小分数而非标签字典序。

### 参考代码

```sql
-- 信用分分层
-- 同一客户可能多笔贷款；分母是贷款。当前分数没有历史可用时间，不能直接当申请时特征回测。
WITH s AS (SELECT l.status,c.credit_score,
 CASE WHEN c.credit_score<600 THEN '<600' WHEN c.credit_score<700 THEN '600-699'
 WHEN c.credit_score<750 THEN '700-749' WHEN c.credit_score<800 THEN '750-799' ELSE '800+' END AS band
 FROM loans l JOIN customers c USING(customer_id))
SELECT band,COUNT(*) AS loans,SUM(CASE WHEN status='defaulted' THEN 1 ELSE 0 END) AS defaults,
 ROUND(100.0*SUM(CASE WHEN status='defaulted' THEN 1 ELSE 0 END)/COUNT(*),2) AS default_rate_pct
FROM s GROUP BY band ORDER BY MIN(credit_score);

```

### 自检和面试追问

同一客户可能多笔贷款；分母是贷款。当前分数没有历史可用时间，不能直接当申请时特征回测。

先独立写入sql/exercises/Q08.sql，再运行python practice.py Q08。变式：如果关联键重复、分母为0或排序并列，你会怎样检查？

## Q09 收入分层

### 思路拆解

1. 贷款接客户收入，确保一人一个收入值。
2. CASE定义收入区间，不捏造收入周期。
3. 先计每组好坏笔数，再算比例。

### 参考代码

```sql
-- 收入分层
-- 收入周期和单位的业务定义未获确认，不擅自构造DTI。
WITH s AS (SELECT l.status,c.income,
 CASE WHEN income<30000 THEN '<30k' WHEN income<50000 THEN '30k-50k'
 WHEN income<75000 THEN '50k-75k' WHEN income<100000 THEN '75k-100k' ELSE '100k+' END AS band
 FROM loans l JOIN customers c USING(customer_id))
SELECT band,COUNT(*) AS loans,SUM(CASE WHEN status='defaulted' THEN 1 ELSE 0 END) AS defaults,
 ROUND(100.0*SUM(CASE WHEN status='defaulted' THEN 1 ELSE 0 END)/COUNT(*),2) AS default_rate_pct
FROM s GROUP BY band ORDER BY MIN(income);

```

### 自检和面试追问

收入周期和单位的业务定义未获确认，不擅自构造DTI。

先独立写入sql/exercises/Q09.sql，再运行python practice.py Q09。变式：如果关联键重复、分母为0或排序并列，你会怎样检查？

## Q10 就业类型分层

### 思路拆解

1. 把客户就业类型接到贷款。
2. 每个就业类型聚合贷款和违约笔数。
3. 排序按率和稳定次键；不能解释为就业类型造成违约。

### 参考代码

```sql
-- 就业类型分层
-- 相关性不能证明因果；合成机制也可能预先制造这种相关。
SELECT c.employment_type,COUNT(*) AS loans,
 SUM(CASE WHEN l.status='defaulted' THEN 1 ELSE 0 END) AS defaults,
 ROUND(100.0*SUM(CASE WHEN l.status='defaulted' THEN 1 ELSE 0 END)/COUNT(*),2) AS default_rate_pct
FROM loans l JOIN customers c USING(customer_id) GROUP BY c.employment_type ORDER BY default_rate_pct DESC,c.employment_type;

```

### 自检和面试追问

相关性不能证明因果；合成机制也可能预先制造这种相关。

先独立写入sql/exercises/Q10.sql，再运行python practice.py Q10。变式：如果关联键重复、分母为0或排序并列，你会怎样检查？

## Q11 地区风险分布

### 思路拆解

1. 从支行维表接地区，branch_id必须唯一。
2. 地区内求笔数、本金与违约数。
3. 检查各地区笔数总和与全组合一致。

### 参考代码

```sql
-- 地区风险分布
-- 检查branch_id唯一，维表重复会放大分母和金额。
SELECT b.region,COUNT(*) AS loans,SUM(l.loan_amount) AS original_principal,
 SUM(CASE WHEN l.status='defaulted' THEN 1 ELSE 0 END) AS defaults,
 ROUND(100.0*SUM(CASE WHEN l.status='defaulted' THEN 1 ELSE 0 END)/COUNT(*),2) AS default_rate_pct
FROM loans l JOIN branches b USING(branch_id) GROUP BY b.region ORDER BY original_principal DESC,b.region;

```

### 自检和面试追问

检查branch_id唯一，维表重复会放大分母和金额。

先独立写入sql/exercises/Q11.sql，再运行python practice.py Q11。变式：如果关联键重复、分母为0或排序并列，你会怎样检查？

## Q12 支行风险排名

### 思路拆解

1. 先按支行主键与名称聚合风险率。
2. 在支行结果上RANK，窗口保留一支行一行。
3. 按未舍入率排名，显示时再ROUND，避免假并列。

### 参考代码

```sql
-- 支行风险排名
-- 按未四舍五入率排名，减少假并列；小支行须结合样本量。原作者按四舍五入率排名。
WITH s AS (SELECT b.branch_id,b.branch_name,COUNT(*) AS loans,
 SUM(CASE WHEN l.status='defaulted' THEN 1 ELSE 0 END) AS defaults,
 100.0*SUM(CASE WHEN l.status='defaulted' THEN 1 ELSE 0 END)/COUNT(*) AS rate
 FROM loans l JOIN branches b USING(branch_id) GROUP BY b.branch_id,b.branch_name)
SELECT branch_id,branch_name,loans,defaults,ROUND(rate,2) AS default_rate_pct,
 RANK() OVER(ORDER BY rate DESC) AS risk_rank FROM s ORDER BY risk_rank,branch_id;

```

### 自检和面试追问

按未四舍五入率排名，减少假并列；小支行须结合样本量。原作者按四舍五入率排名。

先独立写入sql/exercises/Q12.sql，再运行python practice.py Q12。变式：如果关联键重复、分母为0或排序并列，你会怎样检查？

## Q13 放款月份分组与LAG

### 思路拆解

1. 把贷款按放款年月分组，计算快照违约率。
2. LAG取上一组月份的率。
3. 两组相减单位是百分点，但账龄不同，不能叫策略改善。

### 参考代码

```sql
-- 放款月份分组与LAG
-- 是不同放款cohort在此快照的状态比例，不是当月新增违约率；账龄未对齐，不能断言策略改善。LAG默认上一行，缺月要补日历。
WITH m AS (SELECT SUBSTR(disbursement_date,1,7) AS disb_month,COUNT(*) AS loans,
 SUM(CASE WHEN status='defaulted' THEN 1 ELSE 0 END) AS defaults,
 100.0*SUM(CASE WHEN status='defaulted' THEN 1 ELSE 0 END)/COUNT(*) AS rate
 FROM loans GROUP BY disb_month)
SELECT disb_month,loans,defaults,ROUND(rate,2) AS snapshot_default_rate_pct,
 ROUND(rate-LAG(rate) OVER(ORDER BY disb_month),2) AS previous_cohort_delta_pp
FROM m ORDER BY disb_month;

```

### 自检和面试追问

是不同放款cohort在此快照的状态比例，不是当月新增违约率；账龄未对齐，不能断言策略改善。LAG默认上一行，缺月要补日历。

先独立写入sql/exercises/Q13.sql，再运行python practice.py Q13。变式：如果关联键重复、分母为0或排序并列，你会怎样检查？

## Q14 支行审批排名

### 思路拆解

1. 先在申请表按支行算通过率。
2. DENSE_RANK对汇总结果排名，并列后不跳号。
3. 按branch_id作为展示次键，输出可重复。

### 参考代码

```sql
-- 支行审批排名
-- RANK并列跳号，DENSE_RANK不跳号，ROW_NUMBER每行唯一序号。
WITH s AS (SELECT b.branch_id,b.branch_name,COUNT(*) AS applications,
 SUM(CASE WHEN a.approval_status='Approved' THEN 1 ELSE 0 END) AS approved,
 100.0*SUM(CASE WHEN a.approval_status='Approved' THEN 1 ELSE 0 END)/COUNT(*) AS rate
 FROM loan_applications a JOIN branches b USING(branch_id) GROUP BY b.branch_id,b.branch_name)
SELECT branch_id,branch_name,applications,approved,ROUND(rate,2) AS approval_rate_pct,
 DENSE_RANK() OVER(ORDER BY rate DESC) AS approval_rank FROM s ORDER BY approval_rank,branch_id;

```

### 自检和面试追问

RANK并列跳号，DENSE_RANK不跳号，ROW_NUMBER每行唯一序号。

先独立写入sql/exercises/Q14.sql，再运行python practice.py Q14。变式：如果关联键重复、分母为0或排序并列，你会怎样检查？

## Q15 多笔借款客户

### 思路拆解

1. 只在loans表按customer_id计历史贷款。
2. HAVING筛至少两笔的客户。
3. 总原始本金包括已结清，不能直接称当前敞口。

### 参考代码

```sql
-- 多笔借款客户
-- 包含已结清历史贷款，所以是历史累计原始本金，不是当前集中度敞口。
SELECT customer_id,COUNT(*) AS loans,SUM(loan_amount) AS cumulative_original_principal
FROM loans GROUP BY customer_id HAVING COUNT(*)>1 ORDER BY cumulative_original_principal DESC,customer_id;

```

### 自检和面试追问

包含已结清历史贷款，所以是历史累计原始本金，不是当前集中度敞口。

先独立写入sql/exercises/Q15.sql，再运行python practice.py Q15。变式：如果关联键重复、分母为0或排序并列，你会怎样检查？

## Q16 申请放款漏斗核对

### 思路拆解

1. 先把loans压到application_id，每申请一个loan_count。
2. 全申请LEFT JOIN已放款汇总。
3. 分审批状态计申请、放款申请与贷款，区别三个单位。

### 参考代码

```sql
-- 申请放款漏斗核对
-- 先将loans聚合到申请粒度，防止一次审批对应多笔放款时重复计数。
WITH funded AS (SELECT application_id,COUNT(*) AS loan_count FROM loans GROUP BY application_id)
SELECT a.approval_status,COUNT(*) AS applications,
 SUM(CASE WHEN f.loan_count>0 THEN 1 ELSE 0 END) AS funded_applications,
 SUM(COALESCE(f.loan_count,0)) AS loans
FROM loan_applications a LEFT JOIN funded f USING(application_id)
GROUP BY a.approval_status ORDER BY a.approval_status;

```

### 自检和面试追问

先将loans聚合到申请粒度，防止一次审批对应多笔放款时重复计数。

先独立写入sql/exercises/Q16.sql，再运行python practice.py Q16。变式：如果关联键重复、分母为0或排序并列，你会怎样检查？

## Q17 每个客户最新申请

### 思路拆解

1. 每客户单独ROW_NUMBER。
2. 日期倒序、同日ID倒序保证第一条定义明确。
3. 外层rn=1，每客户仅一条，再按客户排序展示。

### 参考代码

```sql
-- 每个客户最新申请
-- 日期并列用application_id稳定排序；MAX日期再JOIN可能一人多行。
WITH s AS (SELECT application_id,customer_id,application_date,approval_status,
 ROW_NUMBER() OVER(PARTITION BY customer_id ORDER BY application_date DESC,application_id DESC) AS rn
 FROM loan_applications)
SELECT customer_id,application_id,application_date,approval_status FROM s WHERE rn=1 ORDER BY customer_id;

```

### 自检和面试追问

日期并列用application_id稳定排序；MAX日期再JOIN可能一人多行。

先独立写入sql/exercises/Q17.sql，再运行python practice.py Q17。变式：如果关联键重复、分母为0或排序并列，你会怎样检查？

## Q18 每支行原始本金前三名

### 思路拆解

1. 每个支行单独对贷款金额倒序排号。
2. 同额时loan_id确定次序。
3. 外层rn<=3保证每支行至多3条。

### 参考代码

```sql
-- 每支行原始本金前三名
-- 要恰好3条用ROW_NUMBER，要保留金额并列用DENSE_RANK并可能超过3条。
WITH s AS (SELECT branch_id,loan_id,loan_amount,
 ROW_NUMBER() OVER(PARTITION BY branch_id ORDER BY loan_amount DESC,loan_id) AS rn FROM loans)
SELECT branch_id,loan_id,loan_amount,rn FROM s WHERE rn<=3 ORDER BY branch_id,rn;

```

### 自检和面试追问

要恰好3条用ROW_NUMBER，要保留金额并列用DENSE_RANK并可能超过3条。

先独立写入sql/exercises/Q18.sql，再运行python practice.py Q18。变式：如果关联键重复、分母为0或排序并列，你会怎样检查？

## Q19 JOIN本金重复放大

### 思路拆解

1. 故意直接JOIN贷款与分期，记录错误金额。
2. 独立从贷款表算正确金额。
3. 两者相除得到放大倍数；这题展示错误为何发生，不让你沿用错误聚合。

### 参考代码

```sql
-- JOIN本金重复放大
-- SUM(DISTINCT loan_amount)不是修复方案：不同贷款可能金额相同。先按loan_id聚合。
WITH x AS (SELECT COUNT(*) AS rows_after_join,SUM(l.loan_amount) AS inflated_amount
 FROM loans l JOIN repayments r USING(loan_id)), y AS (SELECT COUNT(*) AS loans,SUM(loan_amount) AS correct_amount FROM loans)
SELECT x.rows_after_join,y.loans,x.inflated_amount,y.correct_amount,
 ROUND(1.0*x.inflated_amount/y.correct_amount,2) AS inflation_multiple FROM x CROSS JOIN y;

```

### 自检和面试追问

SUM(DISTINCT loan_amount)不是修复方案：不同贷款可能金额相同。先按loan_id聚合。

先独立写入sql/exercises/Q19.sql，再运行python practice.py Q19。变式：如果关联键重复、分母为0或排序并列，你会怎样检查？

## Q20 状态与DPD交叉核对

### 思路拆解

1. 按源payment_status聚合。
2. 查看dpd范围、正dpd笔数和缺失付款日期数。
3. 交叉检查标签含义；出现不一致先记录，别强行改为自己假设的口径。

### 参考代码

```sql
-- 状态与DPD交叉核对
-- on_time允许dpd为1或2，可能存在宽限口径，但作者未说明。记录差异，不自动改为late。
SELECT payment_status,COUNT(*) AS installments,MIN(dpd) AS min_dpd,MAX(dpd) AS max_dpd,
 SUM(CASE WHEN dpd>0 THEN 1 ELSE 0 END) AS positive_dpd_rows,
 SUM(CASE WHEN paid_date IS NULL THEN 1 ELSE 0 END) AS missing_paid_date_rows
FROM repayments GROUP BY payment_status ORDER BY payment_status;

```

### 自检和面试追问

on_time允许dpd为1或2，可能存在宽限口径，但作者未说明。记录差异，不自动改为late。

先独立写入sql/exercises/Q20.sql，再运行python practice.py Q20。变式：如果关联键重复、分母为0或排序并列，你会怎样检查？
