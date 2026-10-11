# CreditCheck 原作者数据复现版（v2）

使用 [MaizRajput/CreditCheck](https://github.com/MaizRajput/CreditCheck) 的 **PBIX内嵌合成数据**，没有重新生成数据。目标：复现信贷审批、组合风险与分期偿付分析，并用于SQL面试实操。

## 开始运行

下载学习ZIP后解压，在本目录运行（Python 3.10+，SQLite 3.25+）：

```bash
python run.py
python -m unittest discover -s tests -v
```

如果从GitHub下载代码版，先获取作者数据：

```bash
pip install -r requirements-extract.txt
python extract_source.py
python run.py
```

然后填写 `sql/exercises/Q01.sql`，执行 `python practice.py Q01`。输出到reports/Q01.csv至Q20.csv。

## 数据与内容

25个支行、5,000位客户、7,500次申请、6,112笔贷款、153,170期应还记录。源文件锁定提交和SHA256在 `data/source_manifest.json`。

原作者15个分析场景已独立改写为可本地运行的SQL，另加5题JOIN、最新记录、TopN和质量核对。`docs/EXERCISES.md`是题目与自检，`docs/START_HERE.md`是新版学习顺序。

## 实际发现

通过率81.49%，源status违约占比6.18%，原始本金73.68亿印度卢比。README贷款数与内嵌模型差1笔，42,320条on_time的dpd>0，需要确认口径。详情看 `reports/FINDINGS.md`。

这是作者合成数据的离线描述分析。不能称为快手业务、真实银行数据、上线风控策略或实现了真实降损。没有统一截止日和历史特征快照，不能直接沿用v1的策略收益与FPD7结果。

上游归原作者；公开仓库不附完整数据，个人复现包附CSV。见 `docs/SOURCE_AND_LIMITS.md`。
