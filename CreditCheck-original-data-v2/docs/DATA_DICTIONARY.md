# 数据字典

来源为作者PBIX模型，原CSV文件未公开。保留模型附加的Credit Band、Income Band、Current DPD、DPD Bucket，不将其视为原始可用特征。

## branches（25行）

`branch_id`, `branch_name`, `city`, `region`, `manager_name`

## customers（5,000行）

`customer_id`, `name`, `age`, `gender`, `city`, `income`, `employment_type`, `credit_score`, `join_date`, `Credit Band`, `Income Band`

## loan_applications（7,500行）

`application_id`, `customer_id`, `branch_id`, `application_date`, `requested_amount`, `approval_status`, `risk_score`

## loans（6,112行）

`loan_id`, `application_id`, `customer_id`, `branch_id`, `loan_type`, `loan_amount`, `interest_rate`, `tenure_months`, `disbursement_date`, `status`, `Current DPD`, `DPD Bucket`

## repayments（153,170行）

`payment_id`, `loan_id`, `due_date`, `paid_date`, `amount_due`, `amount_paid`, `payment_status`, `dpd`

## 业务与时间口径

branches一行一支行；customers一行一客户；loan_applications一行一申请；loans一行一已放款贷款；repayments一行一期应还及其支付汇总，不是一笔支付流水。

金额单位沿用作者印度卢比INR；收入周期未确认。逾期天数dpd采用原作者存储值，源快照统一截止日未明确。due_date最大2025-12-31，paid_date最大2026-01-27；不能把2025-12-31擅自当成所有结果的截断日。

status=defaulted是本项目违约标签定义，作者未提供触发标准，不能自行等同90+或中国监管不良率。没有累计本金余额、核销回收明细、申请时分数快照与特征可用时间。不能可靠做真实损失、生产准入回测、统一账龄FPD7或因果评估。
