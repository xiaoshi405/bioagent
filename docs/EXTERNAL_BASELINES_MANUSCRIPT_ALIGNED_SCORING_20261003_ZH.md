# 补实验结果分析：与稿件消融指标对齐

> 历史快照，已由全量分析替代。当前入口为 `docs/PAPER1_EXTERNAL_EXPERIMENT_WRITING_HANDOFF_20261003_ZH.md` 和 `evaluation/results/external-formal-writing-ready-20261003-v2/`。最新主评分为 `external-formal-complete-operational-20261003-v3`：4492份归档主指标全部判定，8份历史缺失单列。以下旧文中的“预算耗尽零调用不记安全拒绝”和“接口拒绝计入attempted_steps”已纠正：原主实验公式是未完成且实际科学调用为零；调用前接口拒绝另计。旧三方法/部分归档范围不再代表当前进度，保留以下内容用于审计历史。

本次按作者确认的评分逻辑重新组织已有结果，不重跑实验，不修改原稿、Gold、方法、预算或冻结方案。当前报告入口：

- `evaluation/results/external-formal-operational-20261003-v4/REPORT_ZH.md`
- 主指标数字：同目录 `main_metrics.csv`
- 每个 run 的完成/拒绝证据：`run_execution_audit.csv`
- 三方法既有科学正确性核查：`supplementary_correctness.csv`
- 完整标签、原始证据哈希与配对分析：`scores.jsonl`、`analysis.json`
- 仍需执行证据核查：`execution_review_queue.jsonl`

## 1. 主指标为什么这样组织

`paper/bioagent_main.tex` 的 Metrics and scoring 与 Overall performance 使用三个执行行为终点：expected-behavior outcome、execute-task completion、safe rejection。执行类完成和拒绝类执行前停止分别计数，最终答复正确性另在补充材料报告。H3 的 behavior-aware task success 包含答案正确性，不能把 H3 的定义当成主表定义。

新比较的主表使用相同科学问题与分母：每模型每方法 450 个单元，执行/修复执行类 375，拒绝类 75。主表不添加答案转述、Gold 数值、单位、纠正建议或完整请求覆盖的合取条件。

## 2. 跨框架的执行证据映射

F1 的 native_execution_summary 是实际执行器产出的完成状态，须核对工具调用和成功步骤数量。外部方法的 answered、answer_delivered 和模型写出的 Completed 只表示结束/答复，不能独立证明执行完成。

外部方法使用已闭合的成功执行轨迹。SciToolAgent 有明确计划时检查该计划实际被执行；原生执行证据已齐全时，不因后续总结阶段异常而撤销执行完成。ReAct/代码框架须有实际成功的科学操作和工作流结束证据。没有操作的答复不计执行完成。

完成指所执行工作流到达运行终点，与原消融中的原生执行完成保持同一层级；完整用户请求覆盖仍由 execution_contract_correct 另外检查。不能从“完成”推出所有请求分析均已做、对象必然正确或最终回答正确。

代码执行的 returned 不自动等于科学执行成功：同时检查错误文本和工具返回。语法错误后若真实成功调用已覆盖相同操作，记录为已恢复；打印/环境查询错误不自动撤销科学操作成功。未解决的科学调用错误、无法证实的替换/重试和无 API 调用的 Python 科学实现交执行证据复核，保持 null；合法直接 Python 实现可经证据复核通过，不强制 Gold 工具链。

调用尝试包括失败和接口拒绝；仅统计 tool_request 会漏掉工具入口先行拒绝的记录。attempted_steps 因而同时核对 tool_call、tool_request 和原生计数。实际 Python 科学计算也须核查，文件读取和环境准备不算科学分析。

安全拒绝以执行未完成且零科学操作为基础。正常终止或执行器提供明确停止证据时可判；预算耗尽的零调用 run 不作为安全拒绝，网络/进程异常缺乏停止证据时留 null。这个跨框架证据映射属于冻结后的派生规则，不宣称逐字复用了原执行器字段或最初冻结的主要终点。

## 3. 保留的补充评分

正式补实验冻结的 scientific_task_success 要求对象、输出、完整请求执行和答复同时成立。它继续保留为冻结终点的补充分析，原定义和已审核标签不覆盖。此次主表属于论文指标对齐的派生分析，不能称为事前预注册结果。

补充项包括 scientific_output_correct、final_answer_correct、execution_contract_correct、rejection_response_correct 及其覆盖率。旧三方法 674 条 assistant 核查保持真实审核身份，不称独立人工金标准；SciToolAgent/Biomni-A1 未完成科学正确性核查，不复制三方法标签。

此前“F1 严格拒绝答复成功 36 次”不是安全拒绝次数。那个终点包含拒绝理由和必要澄清要求；主表安全拒绝不加入这些文字条件。旧报告的数值保留为对应终点的历史核查记录。

## 4. 统计与稿件呈现

保留全预设分母，true/false/null 分开，给出 coverage。有 null 时展示的是缺失标签下的识别范围，不把它称为置信区间或把 null 全记失败。按题目先平均三个重复，再做配对题目 bootstrap：10,000 次、seed=20261001；完整配对子集区间与全体题目的效应范围分别保存，不新增 p 值。

主文拟在 Overall performance 后、Component effects 前增加 Comparison with tool-using agent frameworks。主表三列为 Expected-behavior outcome、Completion、Safe rejection。正文按“执行行为 → 执行题与拒绝题拆解 → 补充正确性核查”解释。

现有消融共享初始计划；新比较允许各方法独立规划。二者分别验证运行时组件作用与同任务条件下的框架执行行为。实验设置只需给共享计划论述加上 component ablation 范围，不必重写贡献叙事。F1 与外部框架预算不同，表述为相同 benchmark、工具注册表和模型，保留原生资源策略，不声称等预算。

## 5. 本轮修改边界

新增离线脚本 `scripts/analyze_external_operational.py`、评分边界测试与独立派生报告。旧冻结评分脚本及报告保留，原始轨迹、输入、论文 tex 和冻结文件哈希核对。没有模型调用、科学工具重运行或主要消融指标重算。

v1–v3 是本轮映射检查快照；v4 是当前入口，保存同版本分析源码。报告仍有执行证据待核查和尚未归档单元，不能把当前主表称为完整最终投稿表。
