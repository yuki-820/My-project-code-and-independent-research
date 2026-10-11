# 20道SQL通关题

Q01–Q15对应原作者15项分析，Q16–Q20为独立补充。先独立写，再看答案。

每题口述：业务问题→表粒度→关联键→分母→NULL和时间边界→如何对账。结果一致仅为最低门槛。

## Q01 组合规模与状态违约占比

任务：复现 `reports/Q01.csv` 中的列与指标，按同样排序输出。字段结构见 DATA_DICTIONARY.md。

自检：一行一笔贷款。原始本金之和不是当前贷款余额；违约依赖源status定义。

运行 `python practice.py Q01`。

## Q02 申请审批通过率

任务：复现 `reports/Q02.csv` 中的列与指标，按同样排序输出。字段结构见 DATA_DICTIONARY.md。

自检：分母用全部申请，不能从仅有放款的loans表算通过率。

运行 `python practice.py Q02`。

## Q03 快照内分期金额偿付比

任务：复现 `reports/Q03.csv` 中的列与指标，按同样排序输出。字段结构见 DATA_DICTIONARY.md。

自检：不是平均每期偿付率，也不是含时间截断的某日催收回款率。

运行 `python practice.py Q03`。

## Q04 每月放款量

任务：复现 `reports/Q04.csv` 中的列与指标，按同样排序输出。字段结构见 DATA_DICTIONARY.md。

自检：SQLite ISO日期取年月用SUBSTR；MySQL可用DATE_FORMAT。

运行 `python practice.py Q04`。

## Q05 累计放款额

任务：复现 `reports/Q05.csv` 中的列与指标，按同样排序输出。字段结构见 DATA_DICTIONARY.md。

自检：先月度聚合再开窗；解释ROWS与RANGE。

运行 `python practice.py Q05`。

## Q06 产品类型风险分布

任务：复现 `reports/Q06.csv` 中的列与指标，按同样排序输出。字段结构见 DATA_DICTIONARY.md。

自检：笔数率与金额率不同；没有剩余本金与回收台账，不能推算真实损失。

运行 `python practice.py Q06`。

## Q07 复核原作者DPD桶

任务：复现 `reports/Q07.csv` 中的列与指标，按同样排序输出。字段结构见 DATA_DICTIONARY.md。

自检：复现missed中最大dpd。原文90+标签实际是>90；此处修正为91+。未提供统一快照日期，不称为独立重算当前DPD。

运行 `python practice.py Q07`。

## Q08 信用分分层

任务：复现 `reports/Q08.csv` 中的列与指标，按同样排序输出。字段结构见 DATA_DICTIONARY.md。

自检：同一客户可能多笔贷款；分母是贷款。当前分数没有历史可用时间，不能直接当申请时特征回测。

运行 `python practice.py Q08`。

## Q09 收入分层

任务：复现 `reports/Q09.csv` 中的列与指标，按同样排序输出。字段结构见 DATA_DICTIONARY.md。

自检：收入周期和单位的业务定义未获确认，不擅自构造DTI。

运行 `python practice.py Q09`。

## Q10 就业类型分层

任务：复现 `reports/Q10.csv` 中的列与指标，按同样排序输出。字段结构见 DATA_DICTIONARY.md。

自检：相关性不能证明因果；合成机制也可能预先制造这种相关。

运行 `python practice.py Q10`。

## Q11 地区风险分布

任务：复现 `reports/Q11.csv` 中的列与指标，按同样排序输出。字段结构见 DATA_DICTIONARY.md。

自检：检查branch_id唯一，维表重复会放大分母和金额。

运行 `python practice.py Q11`。

## Q12 支行风险排名

任务：复现 `reports/Q12.csv` 中的列与指标，按同样排序输出。字段结构见 DATA_DICTIONARY.md。

自检：按未四舍五入率排名，减少假并列；小支行须结合样本量。原作者按四舍五入率排名。

运行 `python practice.py Q12`。

## Q13 放款月份分组与LAG

任务：复现 `reports/Q13.csv` 中的列与指标，按同样排序输出。字段结构见 DATA_DICTIONARY.md。

自检：是不同放款cohort在此快照的状态比例，不是当月新增违约率；账龄未对齐，不能断言策略改善。LAG默认上一行，缺月要补日历。

运行 `python practice.py Q13`。

## Q14 支行审批排名

任务：复现 `reports/Q14.csv` 中的列与指标，按同样排序输出。字段结构见 DATA_DICTIONARY.md。

自检：RANK并列跳号，DENSE_RANK不跳号，ROW_NUMBER每行唯一序号。

运行 `python practice.py Q14`。

## Q15 多笔借款客户

任务：复现 `reports/Q15.csv` 中的列与指标，按同样排序输出。字段结构见 DATA_DICTIONARY.md。

自检：包含已结清历史贷款，所以是历史累计原始本金，不是当前集中度敞口。

运行 `python practice.py Q15`。

## Q16 申请放款漏斗核对

任务：复现 `reports/Q16.csv` 中的列与指标，按同样排序输出。字段结构见 DATA_DICTIONARY.md。

自检：先将loans聚合到申请粒度，防止一次审批对应多笔放款时重复计数。

运行 `python practice.py Q16`。

## Q17 每个客户最新申请

任务：复现 `reports/Q17.csv` 中的列与指标，按同样排序输出。字段结构见 DATA_DICTIONARY.md。

自检：日期并列用application_id稳定排序；MAX日期再JOIN可能一人多行。

运行 `python practice.py Q17`。

## Q18 每支行原始本金前三名

任务：复现 `reports/Q18.csv` 中的列与指标，按同样排序输出。字段结构见 DATA_DICTIONARY.md。

自检：要恰好3条用ROW_NUMBER，要保留金额并列用DENSE_RANK并可能超过3条。

运行 `python practice.py Q18`。

## Q19 JOIN本金重复放大

任务：复现 `reports/Q19.csv` 中的列与指标，按同样排序输出。字段结构见 DATA_DICTIONARY.md。

自检：SUM(DISTINCT loan_amount)不是修复方案：不同贷款可能金额相同。先按loan_id聚合。

运行 `python practice.py Q19`。

## Q20 状态与DPD交叉核对

任务：复现 `reports/Q20.csv` 中的列与指标，按同样排序输出。字段结构见 DATA_DICTIONARY.md。

自检：on_time允许dpd为1或2，可能存在宽限口径，但作者未说明。记录差异，不自动改为late。

运行 `python practice.py Q20`。
