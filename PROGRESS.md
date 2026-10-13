# 阶段性进度

## Phase 1 — 工具链集成（完成）

- Python Pipeline、Adapter、CLI 骨架。
- PDF/EPUB/MOBI/AZW3 识别。
- Marker CLI Adapter：文字层/OCR 选项通过真实 CLI 参数传递。
- Calibre Adapter：`ebook-convert` 输入/输出转换。
- EPUB 规范化、资源保留、内置结构校验。

## Phase 2 — 本地翻译闭环（完成）

- 内置语言检测与命令行覆盖。
- Ollama `/api/chat` Adapter，模型、地址和推理参数配置化。
- BabelEbook CLI Adapter（可选）；默认 native Ollama 路径用于段落级缓存和恢复。
- translated/bilingual/original 三种模式。
- EPUB/AZW3 输出链路（AZW3 需要本机 Calibre）。

## Phase 3 — 质量与可靠性（完成）

- 段落稳定 ID、缓存键包含文本/语言/目标/模型/prompt/glossary/关键参数。
- `state.json` 增量写入、任务恢复、失败段落查询。
- 确定性 QA：缺译、重复 ID、数字变化。
- 可选 Ollama JSON 审校接口。
- 阶段日志目录和明确外部工具错误。
- 长篇一致性记忆：持久化术语、章节摘要、最近译文上下文和记忆指纹。
- 术语抽取后的混合语言校正；无法可靠校正的条目不进入锁定术语库。
- 审校发现问题时支持按段落自动重译，并保留审校 JSON 记录。

## 实际验证记录（2026-10-10）

- 运行环境：macOS 26.6.2、arm64、Python 3.14.6。
- `ollama --version`：客户端 0.40.2；本机服务已通过 `127.0.0.1:11434` 运行。
- 已下载并验证 `translategemma:12b`（8.1 GB）与 `qwen3.5:latest`（6.6 GB）；TranslateGemma 下载时的本机代理超时已通过 Ollama 本地直连和断点续传修复。
- 安装前 `marker_single`、`ebook-convert`、`epubcheck` 未在 PATH；安装后对应 Adapter 的失败提示仍由测试覆盖，真实命令均已通过版本/帮助检查。
- `python3 -m unittest discover -s tests -v`：17 个测试全部通过（含假的 Ollama 闭环、缓存恢复、EPUB 资源保留、Marker/Calibre/BabelEbook CLI 合约、TranslationMemory 持久化、术语响应解析、数字 QA 和残留重试）。
- `python3 -m bookflow doctor --config config.example.yaml`：Ollama、Marker、Calibre、BabelEbook 四项均已发现。
- `.venv/bin/bookflow --version`：0.1.0；`.venv/bin/python -m pytest -q`：17 passed。
- 实际端到端样本：Project Gutenberg *Frankenstein* Chapter 1，12 个段落、10,129 个英文字符；两个模型均完成 12/12。Qwen3.5 用时 417.593 秒，TranslateGemma 12B 最终用时 201.209 秒（含 2 段局部重试）。
- 评估报告：[artifacts/evaluation/frankenstein/FINAL-EVALUATION.md](artifacts/evaluation/frankenstein/FINAL-EVALUATION.md)。结论：Qwen3.5 当前质量更适合作为默认文学翻译模型；TranslateGemma 更快，英文残留通过自动局部重试修复，最终 QA 通过但文学措辞略逊。
- 长篇评估：使用风格不同的 Project Gutenberg *Alice's Adventures in Wonderland*，连续翻译 30 个段落（12,226 个英文字符），跨越第一章和第二章，Qwen3.5 用时 268.864 秒；术语库 20 条、章节摘要 2 条，自动 QA 通过，仅 1 条术语变体警告；结果见 `artifacts/evaluation/alice/long-qwen/`。

## 外部依赖安装与真实集成验证（2026-10-10）

- Calibre 9.16.0：通过 Homebrew Cask 安装，使用公开 EPUB 完成 EPUB→AZW3→EPUB 往返转换；`unzip -t` 通过，EPUBCheck 5.4.0 报告 0 errors / 0 warnings。
- EPUBCheck 5.4.0：已安装并作为最终 EPUB 的独立质量门禁。
- Marker 2.0.0：安装到项目 `.venv`，导入验证成功；PyTorch 2.14.1 报告 `mps_built=True`、`mps_available=True`，可在 M4 上使用 Apple GPU。
- BabelEbook 0.5.0：从上游 Rust 源码构建到 `.tools/bin/babel-ebook`；当前 Adapter 已适配其 `translate SOURCE -o OUTPUT --config CONFIG` 接口，并通过 CLI 帮助与版本检查。
- `bookflow doctor --config config.example.yaml`：四项外部工具均已发现（Ollama、Marker、Calibre、BabelEbook）。
- 备注：BabelEbook 当前 `pdf-to-epub` 子命令依赖外部视觉 API；为保持默认离线，BookFlow 的 PDF 入口使用本地 Marker 抽取文字，再走统一 EPUB 翻译链路。
- 管理学实际闭环：对 LibreTexts *Principles of Management*（公开许可、531 页）使用 Marker `--page_range 0-7` 抽取前 8 页，得到 55 个段落；PDF→EPUB 原始产物通过 `unzip -t` 和 EPUBCheck 5.4.0。
- EPUB→双语 EPUB：同一外文 EPUB 由 Qwen3.5 完成 55/55 段翻译；术语库 19 条、章节摘要 1 条，中文对照节点 55 个；数字 QA 通过，最终 EPUBCheck 0 errors / 0 warnings。
- Calibre 输出：最终双语 EPUB→AZW3→EPUB 往返成功，往返 EPUBCheck 0 errors / 0 warnings。真实产物在 `artifacts/evaluation/management/epub-bilingual/`。
- 本轮审校请求验证了本地 Qwen3.5 JSON 接口，但 55 段逐段审校耗时超过本次验证窗口，最终交付样本使用已完成翻译缓存和确定性 QA 生成；审校仍保留为默认能力，并新增逐段 `review_checkpoint.json`，避免中断后重复审校。
- 修复了两个真实质量问题：日期中的前导零不再触发数字 QA 误报；双语译文节点不再复制原文标题 `id`，严格 EPUBCheck 通过。
- 本轮产物修复：双语 EPUB 改为原文段落后紧邻对应译文；Marker 原文和模型译文中的 Markdown 粗体、斜体、链接、图片均转换为 XHTML；PDF 目录解析为章节标题和子项目层级列表，恢复页锚点与分页，并过滤分段抽取时不存在的远端页链接，避免目录格式崩坏或 Licensing 错位。管理学样例现有 152 个可翻译节点，规范化 EPUB 和最终双语 EPUB 均通过 EPUBCheck 5.4.0；视觉复核确认目录结构及 Licensing 独立分页；AZW3 使用 Calibre `--no-inline-toc` 生成，往返 EPUBCheck 通过，测试为 21 passed。
- 脚注处理修复：识别常见 EPUB 脚注引用及返回链接，翻译输入使用 `[[BOOKFLOW_FOOTNOTE_N]]` 占位符，输出时恢复可点击的上标链接并避免脚注正文/编号重复；同时修复 ZIP 顺序与 OPF spine 顺序不一致导致的段落 ID 错配。脚注回归测试和全套自动化测试共 23 项通过；Superforecasting 任务将从全新 `runs-footnotes` 工作目录重新开始。
- Ollama 稳定性修复：请求层新增可配置的指数退避重试（默认额外重试 3 次，退避 2/4/8 秒），覆盖翻译、术语抽取、章节摘要、审校及审校触发的修订；对 HTTP 408/429/5xx、连接错误、超时和非 JSON 响应重试，持续失败仍由 checkpoint 支持安全恢复。自动化测试更新为 24 项通过。
- 审校协议修复：合法 JSON 才允许解释 `ok`；模型返回截断 JSON、代码块或其他结构异常时按同一重试上限再次请求，耗尽后记录 `ok: false`、`review_error: invalid_json`，不把原始响应当作重译建议；旧 checkpoint 中误记为 `ok: true` 的协议失败记录会在恢复时重新审校。相关回归测试后全套测试为 26 项通过。
- 审校截断与稳定性修复：实测发现单纯把 `num_predict` 从 256/512 无限放大会导致完整译文建议过长，甚至诱发 Ollama HTTP 500；审校改用 Ollama JSON Schema，限制 issues 最多 3 条、每条及 suggestion 最多 160 字符，基础输出 1024，格式重试按 1024→2048→4096 逐级扩大，上下文保持 4096。翻译请求当前没有设置 `num_predict`，不会被此审校上限影响。
- 审校协议进一步收紧：模型不再生成 suggestion 或完整修订段落，只返回 `ok` 和最多 3 条 issues；BookFlow 在本地将 issues 组合成简短 suggestion 供重译使用。这样失败重试不会因模型重复输出整段译文而截断。

## 整段修订与可追溯审校修复（2026-10-11）

- 根因确认：旧修订输入使用 `Complete review feedback history`，而 Adapter 只识别 `Reviewer feedback history:`，造成反馈被截为前 1500 字符；同时整段编辑复用了 `chunk_size=1200` 的初译分块，可能拼接出重复内容。选择器还将每个候选截为前 3500 字符，并沿用过时审校意见；选择理由没有独立验证。
- 独立 `OllamaTranslator.revise` 一次请求处理完整段落，保留起始译文、上一候选和全部经核验的历史反馈，不再切块或切掉正文。无进展时指示重新组织句子。审校、修订的本地上下文设为可配置的 16384；初译配置不变，避免作废全书翻译缓存。
- 当前候选独立审校；否决意见再核验证据，允许等义表达、语序调整及中文标点，严重等级不随失败次数上升。严重问题修好后按当前 minor 上限收尾。候选去重、完整文字两两评选，选中后重新审校；选择理由不能当作 `ok=true`。
- `reviews/segments/*.json` 保存原文、起始译文、全部候选、逐轮审校、初步意见及核验、最终选择和复审结果；每步保存，恢复沿用剩余轮次。检查点带版本与译文哈希。旧流程修订过的记录重新审校；新版非严格模式已复核的限次记录不在每次恢复时无限循环。脚注数量不一致不能因模型通过而被忽略；模型报告输出截断时不接受部分译文。
- 修复数字 QA 的真实误报：1.3 billion 与 13 亿按数量等值比较；数量级错误（1.3 亿、13 万、13）仍失败。修订上限保持 minor 2、major/critical 5，默认及当前任务均非严格。
- `python3 -m unittest discover -s tests -q`：41 项通过；新增覆盖实际 Adapter 长输入、一次整段修订、完整候选选择、错误选择理由、否决核验、恢复轮次、脚注保护、严重降级预算、截断重试及数字单位换算。
- 本地真实回归：`tools/evaluate_revision.py`，配置中的 Qwen3.5，think=false。从旧流程最终保留的译文开始复核（不是同一起始初译的严格 A/B 实验）。结果：`d0f920aab1af2a13c8e1` 原 5 轮 critical 未通过 → 2 轮后只剩 minor，英文残留和重复句消失；`e1c7fbe92e24e1637736` 原 5 轮 critical 未通过 → 1 轮通过，修复重复/截断引语；`789f111f27c6c4769667` 原 5 轮 critical → 0 轮通过，原稿当前已为“鲍尔默”及正确的“13 亿”，旧意见不再适用；`d0ee80857eb863138b84` 原 2 轮 minor → 0 轮通过，排除可接受破折号的误报。
- 四条模型处理分别耗时 148.00、58.83、17.08、9.39 秒；数字 QA 重验全部通过。完整对比和候选在 `artifacts/evaluation/revision-v2-final/`。四条已校验结果写回整书；旧检查点与译文备份在任务目录 `reviews-before-revision-v2/` 和 `translations/segments-before-revision-v2.json`。整书任务已恢复，仍需观察大样本收敛情况。


## 自动准备、全书一致性与自动验收（2026-10-13）

- 全书分批扫描术语并复核，复用原文分析生成章节/全书概要；本地模型自动推断风格，人工术语和风格字段优先，无需先手工配置。
- 初译、审校和修订共享规则及相邻原文；仅发送当前文本命中的术语，移除旧上下文 1500 字截断。章节检查进入有限、可恢复的修订闭环。
- 快速审校提出问题后自动进入默认推理模式核验证据；模型不可用时保留诊断并回退，避免每段都支付慢速推理成本。
- 增强数字证据、脚注和图片硬性检查、EPUB 内容覆盖及旧段落 ID 迁移；加入 EPUBCheck 与自动验收，区分运行完成、待审、阻塞和定稿，产物随附质量记录和哈希。
- 70 项回归检查通过。真实本地 Qwen3.5 无人工配置短样本完成 3/3 节点，段落/章节/术语检查通过，EPUBCheck 零错误零警告，自动 finalized；再次恢复成功。
- 未重跑现有整书，短样本不代表整书质量和耗时已经验证；复杂诊断样本曾中断，未计为通过。详见 `artifacts/evaluation/autonomous-workflow-report.md`。
