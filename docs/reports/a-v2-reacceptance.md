# 新版A再次验收与接续交接

日期：2026-10-27。结论：**三项修复复验通过，A工具层可交接；真实语义数据与研究实验阶段尚未通过验收。** 本轮未修改算法、真实标注或审核身份。

## 实际复验

| 检查 | 本轮结果 |
| --- | --- |
| A-V2-01源图身份 | 回归覆盖图片/审核摘要同时替换、inventory绑定/覆盖/错配、冻结上下文；通过 |
| A-V2-02排名差配对 | 无交集返回null、部分交集只用共有sample/variant、先按源图宏平均；通过 |
| A-V2-03金标/评分统一 | 重复规范化实体、属性、有向关系冻结前拒绝，预测/金标错误区分；通过 |
| 语义回归 | 36项全部通过 |
| 完整回归 | 133项，132通过、1项Windows符号链接权限跳过 |
| 七个CLI注册 | 每个命令的`--help`均正常退出 |
| 合成端到端 | evaluate与compare重新执行，各100次bootstrap，保留synthetic标记 |
| 真实工作包 | 30张val源图字节与原inventory一致；原始/辅助工作包重建摘要均相同 |
| 未审数据门槛 | 真实草稿用于eval、train、freeze均被拒绝；拒绝冻结未留下输出目录 |

本轮审阅修复入口、运行自动测试和数据审计；未重新进行浏览器交互、人工图像语义复核或模型训练。测试通过只证明本轮检查范围，不能替代真实语言质量、人工身份或run-v2制品来源验收。当前结构校验不会强制每组具有全部现象；冻结前仍须检查有效标注与覆盖，见A-V2-04。

## 新证据

本机目录 `artifacts/a-v2-reacceptance-20261027-161218/`（Git忽略，换机需复制或重跑）：

- `semantic-tests.log`、`all-tests.log`：本轮新测试输出。
- `acceptance.json`：真实审计、拒绝原因、分配计数、工作文件与实现SHA-256。
- `commands.json`：执行环境、Python路径与时间。
- `rebuilt-pilot/`、`rebuilt-assisted/`：新目录重建，未覆盖原文件。
- `synthetic-evaluation/`、`synthetic-comparison/`：接口验证输出，不是模型成绩。

重跑入口：

```powershell
python -m unittest discover -s tests -p test_semantics.py -v
python -m unittest discover -s tests -v
python -m fgir validate-semantics --samples data/semantic/pilot-val-assisted-v1/samples.work.jsonl --data-root data
```

源工作文件SHA-256：

- 原始：`89cd73664d5022a87b8ad24154740d57eae19affbe180a094ea14654742bad31`。
- 辅助：`33d44c2bd2c8e142e768bd6544fa0d664a83b1bd4b9ddd7b6dd59cedfbc49beb`。

三项旧问题的历史反例保留在[修复前验收](a-v2-acceptance-review.md)，修复细节见[修复记录](a-v2-fixes.md)。历史脚本断言的是旧错误，不拿它的预期失败冒充当前回归失败。

## 未完成项与接手入口

30组中8组只有助手原句预审、22组draft；全部缺实体/变体/组件，0组独立人工金标。8组预审为4 supported、4 uncertain，全部30组的source标签为4 supported、26 uncertain。无真实N系列结果、正式test成绩及12例模型错误分析。

已建立根目录[AGENTS.md](../../AGENTS.md)与[接续任务板](../research-v2-tasks.md)。任务板逐项给出负责人、READY/BLOCKED/WAITING_HUMAN状态、前置输入、输出路径、验收标准与下游解锁条件：B负责语言候选/train核验数据；C负责LoRA/受控实验；D负责制品/导出/服务；全体真人负责初标互审；A在依赖到位后冻结和正式评测。接手者可直接按角色执行，不需要再从聊天推断剩余工作。
