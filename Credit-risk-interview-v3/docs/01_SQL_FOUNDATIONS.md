# SQL从读懂到能写：四个核心积木

## 0. 一张表到底是什么

把表看成Excel。一列是字段，一行是一次观察。最先问“这一行是什么”。customers每行一个客户；loan_applications每行一次申请；loans每行一笔借款；repayments每行一期应还和汇总支付。一个客户可以多次申请，一笔贷款可以分很多期，所以这些表不能随意拼起来求和。

记住一句话：先确定最终一行代表什么，再决定如何汇总和连接。

## 1. SELECT、WHERE和CASE WHEN

```sql
SELECT application_id, customer_id,
       CASE WHEN approval_status='Approved' THEN 1 ELSE 0 END AS approved_flag
FROM loan_applications
WHERE requested_amount >= 100000;
```

逐行翻译：FROM选择申请表；WHERE只留下申请金额至少10万的记录；SELECT选出要看的列；CASE给每行做判断，通过写1、否则0；AS给新列命名。此处10万只是筛选练习，不是推荐准入阈值。

CASE像Excel的IF，但能写多层。条件按先后顺序匹配，命中一项后不再继续。分数600应进入600–699，所以第一项写 `<600`，第二项写 `<700`。不要在两项都写包含600的闭区间然后忽略顺序。

NULL是未知，不是0，也不是空字符串。用 `IS NULL`，不要写 `= NULL`。`COALESCE(x,0)`是在明确“缺失应当按0处理”时才用，不能把缺失信用分改成0后声称他高风险。

## 2. GROUP BY：把很多行压成每组一行

小表：A支行两笔金额100和200，B支行一笔300。

```sql
SELECT branch_id, COUNT(*) AS loan_count, SUM(loan_amount) AS total_amount
FROM loans
GROUP BY branch_id;
```

结果只有两行：A→2笔/300，B→1笔/300。原来具体哪笔贷款金额100的信息不在这个结果里了。

COUNT(*)数行，包括某些列为空的行；COUNT(paid_date)只数非空日期。COUNT(DISTINCT customer_id)数不同客户，不能拿来替代贷款数。SUM求总额，AVG求平均。整数除法可能截断，比例用100.0或1.0确保小数运算。

WHERE在分组前筛原始行，HAVING在分组后筛整组：

```sql
SELECT customer_id, COUNT(*) AS n
FROM loans
WHERE status <> 'closed'
GROUP BY customer_id
HAVING COUNT(*) >= 2;
```

这问的是“至少两笔未结清贷款的客户”，与Q15“历史至少两笔贷款”不同。面试中改一个WHERE就换了业务口径。

## 3. CASE加聚合：一张表里同时算多个指标

```sql
SELECT COUNT(*) AS applications,
       SUM(CASE WHEN approval_status='Approved' THEN 1 ELSE 0 END) AS approved,
       100.0 * SUM(CASE WHEN approval_status='Approved' THEN 1 ELSE 0 END)
       / NULLIF(COUNT(*),0) AS approval_rate_pct
FROM loan_applications;
```

CASE先给每行打0/1，SUM再数有几个1。NULLIF(x,0)把0分母改成NULL，避免除零。7500次申请6112次通过，率为81.4933…%，四舍五入81.49%。如果先WHERE Approved，分母也只剩通过申请，结果将永远100%。这是面试最常见的陷阱之一。

平均比例的陷阱：甲组1/2=50%，乙组9/10=90%，总率是10/12=83.33%，不是两个率平均得到70%。先加分子与分母，再相除。

## 4. JOIN：按键把字段接过来

```sql
SELECT l.loan_id, l.loan_amount, c.credit_score
FROM loans l
LEFT JOIN customers c ON l.customer_id=c.customer_id;
```

l、c是别名，让表达简洁。ON规定配对方式。客户表customer_id唯一时，这是多对一：许多贷款对应一个客户，连接后应仍是6112行。

INNER JOIN只保留匹配上的。LEFT JOIN保留全部左表，右表找不到时填NULL。若LEFT JOIN之后在WHERE写 `c.credit_score>=600`，没有客户匹配的NULL行也会被滤掉。想保留左表且仅接符合条件右表，把条件放ON并单独标记缺失；想只看分数达标贷款，WHERE才是目标。不能死记“所有条件都放ON”。

### 为什么本金会凭空变多

贷款L1本金100，有两期还款30、40。直接JOIN后变两行：(L1,100,30)、(L1,100,40)。SUM本金成200，而真实本金还是100。

```sql
WITH paid AS (
  SELECT loan_id, SUM(amount_paid) AS total_paid
  FROM repayments GROUP BY loan_id
)
SELECT l.loan_id, l.loan_amount, COALESCE(p.total_paid,0) AS total_paid
FROM loans l LEFT JOIN paid p ON l.loan_id=p.loan_id;
```

先把分期压成每笔贷款一行，再JOIN。`SUM(DISTINCT loan_amount)`不是修复：两笔不同贷款本金都是100，DISTINCT会错误只算一次。正确去重单位是业务主键，不是金额。

面试写完JOIN立刻口述三项检查：右表键是否唯一；连接前后主表行数是否改变；关键金额是否对账。

## 5. 窗口函数：保留明细，额外挂上组内计算结果

仍用100、200、300的三行小表：

```sql
SELECT loan_id, branch_id, loan_amount,
       SUM(loan_amount) OVER(PARTITION BY branch_id) AS branch_total
FROM loans;
```

结果仍三行。A的两行分别是100/组总300、200/组总300，B是300/组总300。GROUP BY像“收起明细只看汇总”；窗口像“保留每张账单，在旁边写所属支行汇总”。

### 面试60秒：GROUP BY和窗口的区别

GROUP BY通常将每组压成一行，适合支行总额、分组违约率。窗口函数在保留当前结果粒度的同时进行分区计算，适合组内排名、累计值、上一期值。窗口不是更高级的GROUP BY，两者常配合：先按月聚合，再对月度结果做累计。窗口也能用于已经GROUP BY过的结果，保留的是该阶段的行数。

### OVER里的三个组成部分

- PARTITION BY：各算各的。例如按客户分开排名。省略时整个结果一个区。
- ORDER BY：计算顺序，例如申请日期倒序；它不保证最终展示顺序，外层仍需ORDER BY。
- ROWS/RANGE：当前行计算时能看哪些邻居。累计用ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW，即从开头到当前。

### 三个排名函数

金额100、100、80倒序。ROW_NUMBER得到1、2、3；RANK得到1、1、3；DENSE_RANK得到1、1、2。取恰好3条用ROW_NUMBER加稳定排序键；取前三种不同金额用DENSE_RANK，可能超过3行。

### 每客户最新申请

```sql
WITH ranked AS (
 SELECT application_id,customer_id,application_date,
        ROW_NUMBER() OVER(
          PARTITION BY customer_id
          ORDER BY application_date DESC, application_id DESC
        ) AS rn
 FROM loan_applications
)
SELECT * FROM ranked WHERE rn=1;
```

先每位客户单独排队，再挑第一条。为什么不能本层WHERE rn=1？按SQL逻辑处理顺序，本层WHERE发生在窗口求值之前，所以需要CTE/子查询外包一层。为什么多排application_id？同日多次申请需要明确谁在前，避免每次结果不稳定。

LAG(x)取排序后前一行的值，首行没有前项得到NULL。不是自动找前一个自然日；若缺日期，前一行可能是三天前。累计、滚动与环比都必须先检查时间连续性。

## 6. 7行不等于7天

若只在1号、3号、10号有订单，ROWS BETWEEN 6 PRECEDING AND CURRENT ROW表示最多7条已有记录，不是近7个自然日。解决方案：先生成每日日期序列并补零，再用7行窗口；或者使用数据库支持的时间RANGE。本文I03用补日历方式，跨数据库更容易解释。今天是否纳入、退款如何处理、支付时间还是创建时间，先问清再写。

## 7. SQL与pandas互译速查

|需求|SQL|pandas|
|---|---|---|
|选列|SELECT a,b|df[['a','b']]|
|筛行|WHERE x>0|df.loc[df['x']>0]|
|多条件|x>0 AND y=1|(df['x']>0) & (df['y']==1)|
|数每组行数|GROUP BY k; COUNT(*)|groupby('k').size()|
|数非空值|COUNT(v)|groupby('k')['v'].count()|
|聚合|SUM/AVG GROUP BY|groupby(...).agg(...)|
|连接|LEFT JOIN|merge(...,how='left')|
|保留明细组总额|SUM(v) OVER(PARTITION BY k)|groupby('k')['v'].transform('sum')|
|累计|SUM(v) OVER(ORDER BY t ROWS...)|sort_values(...).groupby(...).cumsum()|
|上一行|LAG(v)|groupby(...).shift(1)|
|条件分桶|CASE WHEN|pd.cut或布尔赋值|
|行转列|条件聚合|pivot_table(...,aggfunc='sum')|

最后两个差异要会说：pandas merge可能把两边的空键匹配，普通SQL等值JOIN不会NULL=NULL；pandas groupby默认常会丢弃NA组，SQL通常将NULL分为一组，要一致需显式dropna=False。生产数据库适合大规模就地筛选聚合，pandas适合能装入内存的结果探索、画图和建模，常见分工是SQL先压缩，pandas再分析。

## 8. 通关口试

不看上文回答：①通过率为什么不能先筛Approved？②LEFT JOIN为何可能少行或多行？③ROW_NUMBER为什么加第二排序键？④窗口是否能代替全部GROUP BY？⑤7日GMV为什么要补0？⑥0与NULL什么时候不能互换？

参考：SQLite窗口函数官方文档 https://www.sqlite.org/windowfunctions.html ；pandas与SQL对照 https://pandas.pydata.org/docs/getting_started/comparison/comparison_with_sql.html 。例子与中文教学解释为本训练包独立编写。
