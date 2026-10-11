# 信贷风控面试训练包 v3：真实数据主项目 + 从零代码教学

**先打开 START.html**：完整中文手册、每章复制GPT求助、进度勾选与全文查找。无账号读取、无自动发送，复制后在网页版GPT自行粘贴。

## 真实主项目

UCI公开30,000个台湾历史信用卡客户记录，CC BY 4.0。将半年行为特征加入逻辑回归，用SQL/pandas复算、训练/验证/测试隔离和固定阈值进行风险分层。结果与讲述见real_project/reports/PROJECT_REPORT.md、docs/05_REAL_PROJECT_WALKTHROUGH.md。

## 快速运行

```bash
python -m pip install -r requirements.txt
python real_project/run_real.py
```

从GitHub取得的代码没有原始数据，先运行 `python real_project/download_source.py`。完整学习ZIP已附官方原始XLS与CSV、报告和PPT。

## 训练内容

- SQL四核心：JOIN、窗口、聚合、CASE。CreditCheck原作者合成数据作五表练习，共20题，含逐步讲解与结果自检。
- pandas八题：筛选、groupby、merge、先聚合再合并、最新记录、transform、pivot_table、分箱。
- 面经三SQL：完播率、连续登录、7自然日GMV，独立教学小表，非真实客户记录。
- Python三算法：三数之和、快排、给定数字组成小于n的最大数，含暴力对拍测试。
- 业务：准入/额度/贷中、七个面经问题、评分卡/树模型、样本偏差、策略收益与误伤、项目讲述及模拟面试。

```bash
python run.py
python -m unittest discover -s tests -v
python pandas_lab.py
python interview_cases.py
python algorithms.py
python practice.py Q02
python check_pandas.py P01
```

CreditCheck练习如果缺CSV，先 `python -m pip install -r requirements-extract.txt` 然后 `python extract_source.py`。它是作者合成数据，不与UCI真实数据拼接。两套数据来源与边界分别记录，不能混用指标。

学习顺序与北京时间10月11日晚至14日18:30安排见docs/00_START.md。主项目是历史数据离线分析，不是快手上线、真实降损或信贷实习经历。
