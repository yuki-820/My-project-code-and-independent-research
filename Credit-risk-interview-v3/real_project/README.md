# UCI真实数据主项目

来源与许可：Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H 。CC BY 4.0 https://creativecommons.org/licenses/by/4.0/ 。保留出处，数据处理变更为读取原XLS第二行标题、导出CSV、规范化长表和派生特征，未补造观测。

从训练包根目录：python real_project/download_source.py（代码版首次）与 python real_project/run_real.py。

feature.sql是无标签的特征查询。run_real.py完成同源拆表、pandas独立对账、三集划分、基准/候选训练与评估。reports/PROJECT_REPORT.md给出当前实际结果；results.json是机器可读结果。原始数据年代早，存在来源字典未完整说明的编码，不能代替当前业务的OOT或真实干预评估。
