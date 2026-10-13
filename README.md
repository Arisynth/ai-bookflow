# AI BookFlow

AI BookFlow 是一个面向 macOS Apple Silicon 的本地电子书处理与翻译 CLI。它把 Marker、Calibre、Ollama 和可选的 BabelEbook 放在一个可恢复的 Python Pipeline 后面，默认不访问云端 API，也不尝试绕过 DRM。

本项目代码使用 MIT License。Marker、Calibre、Ollama、BabelEbook、PyYAML 和 pytest 等外部组件不随项目源码分发，分别受各自许可证和使用条款约束，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

当前实现覆盖规格文档中的 Phase 1–3：

- 输入识别：PDF、EPUB、无 DRM 的 MOBI/AZW/AZW3。
- PDF：通过 Marker CLI 做文字层/布局/OCR 处理，再规范化成 EPUB。
- MOBI/AZW3：通过 Calibre `ebook-convert` 转 EPUB。
- EPUB：保留原 ZIP 资源、OPF、spine、图片和 CSS；双语模式按段落上下插入译文，Marker/模型返回的常见 Markdown 行内标记会转换为 EPUB XHTML，并识别常见 EPUB 脚注引用/返回链接，翻译时用占位符保护、输出时恢复可点击脚注。
- 语言识别：内置轻量检测，支持 `--source` 覆盖。
- 翻译：默认直接调用 Ollama `/api/chat`；配置 `translation.engine: babelebook` 时调用 BabelEbook CLI Adapter。
- TranslateGemma 12B 使用官方单用户提示格式；短章节标题由结构层处理，避免短输入触发续写幻觉。
- 质量与可靠性：段落级缓存、原文/译文稳定 ID、增量 state、恢复、数字检查、可选 Ollama 审校、EPUB 结构验证；审校发现问题后进入“修订→复审”闭环，非严格模式允许在限次后采用复核过的最佳候选并明确记录遗留问题。
- Ollama 稳定性：每次请求支持配置化的指数退避重试；翻译段落、术语抽取、章节摘要和审校均复用同一重试策略，阶段 checkpoint 继续保留。
- 长篇一致性：本地模型逐批扫描全书并复核术语，自动设定文体与风格；人工配置优先。初译、审校、修订和章节审校共享规则，最终全书检查术语与跨段语义。
- 输出：EPUB；配置包含 `azw3` 时由 Calibre 再生成 AZW3，并关闭重复的 inline ToC，保留 EPUB 内已有的结构化目录。

## 安装

项目运行时只依赖 Python 标准库；建议安装 YAML 解析器并以 editable 方式安装：

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -e '.[yaml,dev]'
cp config.example.yaml config.yaml
```

macOS 外部工具（Apple Silicon 已在本机实际验证）：

1. 安装并启动 Ollama，下载配置中的模型，例如 `ollama pull <translation-model>`。
2. 安装 Marker（建议使用项目虚拟环境）：

   ```bash
   python3 -m venv .venv
   ./.venv/bin/python -m pip install -i https://pypi.org/simple marker-pdf
   ```

   `marker-pdf 2.0.0` 已在本机 M4 上验证，PyTorch 可使用 MPS；配置示例直接使用 `.venv/bin/marker_single`。
3. 安装 Calibre 和 EPUBCheck：

   ```bash
   brew install --cask calibre
   brew install epubcheck
   ```

   它们分别提供 `ebook-convert`（MOBI/AZW3 输入与 AZW3 输出）和 EPUB 规范校验。
4. 若使用 BabelEbook：安装 Rust 后从上游源码构建 CLI，并把二进制放到 `.tools/bin/babel-ebook`。本项目 Adapter 使用 BabelEbook 0.5 的 JSON 配置接口；其 `pdf-to-epub` 子命令目前需要外部视觉 API，因此离线 PDF 流程使用 Marker 完成 PDF→Markdown，再由 BookFlow 规范化成 EPUB。

检查工具：

```bash
bookflow doctor --config config.yaml
```

## 使用

```bash
bookflow convert book.epub --config config.yaml --output ./out
bookflow convert book.pdf --config config.yaml --source en --mode bilingual
bookflow convert book.azw3 --config config.yaml --target zh-CN
bookflow resume <job-id> --config config.yaml
bookflow status <job-id> --config config.yaml
bookflow failures <job-id> --config config.yaml

# 用 Qwen3.5 做一段可恢复的长篇一致性评估
PYTHONPATH=. python3 -m tools.evaluate_book book.epub \
  --output artifacts/evaluation/book-long \
  --model qwen3.5:latest --start 0 --limit 30
```

`--mode` 为 `original`、`translated` 或 `bilingual`。翻译过程的每本书工作目录默认在 `~/.bookflow/<job-id>/`：`source/`、`normalized/`、`metadata/`、`translations/`、`reviews/`、`output/` 和 `state.json`。输入文件只读，不会被覆盖。

审校配置中的 `review.minor_revision_rounds` 和 `review.serious_revision_rounds` 分别控制轻微问题与严重问题的“修订后复审”轮数，默认是 2 和 5。修订使用独立的整段接口，一次请求输出一份完整译文，不再复用初译分块，也不按字符截断反馈。输入包括原文、本次修订流程的起始译文、上一候选和所有历次问题及建议；重复失败时明确要求重新组织段落。`review.num_ctx` 同时用于审校和修订，示例为 16384，初译的上下文配置独立。

审校以当前原文/译文为准，不把旧意见当作仍然存在的问题；否决结果再经过一次证据核验，过滤同义表达、语序差异和不适用于中文的英文语法要求。严重问题修好后，若只剩 minor，按当前轻微问题上限收尾，不沿用旧 major/critical 的五轮预算。达到上限后，审校模型通过完整候选两两比较选出较佳版本，再对选中的实际文字独立复审；选择理由不等于通过证明。模型复审不能覆盖缺失或重复脚注占位符等确定性错误。默认非严格模式 `review.strict: false`，语义问题未解决的最佳候选可作为待审稿输出；数字缺失、空译和脚注/图片占位符错误始终阻止输出。`true` 要求全书准备、段落与章节审校以及 EPUBCheck 全部通过。旧配置中的 `max_revision_rounds` 仍兼容，并作为轻微问题轮数的回退值。

`reviews/segments/<segment_id>.json` 保存原文、起始译文、全部候选正文、逐轮意见、候选索引、最终选择和复审结果；每轮持久化，支持剩余轮次恢复。检查点记录规则、相邻原文、审校配置版本和实际译文哈希；升级后的旧审校记录会重新核验，旧候选档案保存在 `reviews/segments/history/`。新版非严格模式中已完成复核的限次结果不会在每次恢复时无限重跑；协议错误仍重试。历史运行未保存的初译和候选正文不能凭空还原。

PDF 测试或大书分段处理可以在 `pdf.page_range` 配置 Marker 页码范围，例如 `"0-7"`；留空表示整本 PDF。该选项只限制 Marker 抽取页，不改变 EPUB 输入、缓存和翻译接口。

若 Ollama 尚未启动，任务会失败并在 `state.json` 记录原因；修复环境后执行 `resume`，已完成的段落不会重复请求模型。模型、地址、生成参数和术语表都来自配置文件或 CLI 覆盖。

## 无需人工填写的全书流程

默认无需先填术语或选择风格。流程为：全书分批扫描 → 本地模型复核术语 → 推断文体/风格 → 初译 → 段落审校及修订 → 带相邻原文的章节审校及局部修订 → 全书术语和确定性检查 → EPUBCheck → 自动验收。`models.review` 未填写时使用翻译模型。审校有独立的生成预算；普通审校默认 `review.think: false`。出现否决或章节问题时，自动用 `review.dispute_think: null`（遵循本地模型默认推理模式）再次核验证据，仅确认的问题进入修订。推理模式不可用时保留错误记录并自动回退，不要求人工选择。争议核验可能明显增加耗时。此完整流程使用 `translation.engine: ollama`；BabelEbook 路径不能静默绕过已启用的规则及审校，必须明确关闭这些功能后才可使用。

- `metadata/preparation.json` 逐批保存扫描、术语裁定和风格，失败后恢复会重试未完成批次，不把失败标为已完成。扫描覆盖全部章节，同时生成各段概要，复用这些分析合并为章节及全书概要，不另行重复扫描整本书；保留合理的缩写、品牌和原文书目。
- 人工 `translation.glossary` 和 `translation.terminology` 始终优先；自动术语记录来源、释义、别名、允许变体和原文证据。普通词或不可靠候选可被模型剔除。人工译法不允许模型以“同义表达”为由豁免。
- 人工 `translation.style` 可以是说明文字或字段对象；只填写部分字段时，其他字段由本地模型补齐。自动风格识别使用跨全书样本，不固定套用文学翻译风格。
- 长段初译在句子/词边界分块，脚注和图片占位符保持完整。审校修订仍一次输出完整段落。
- 章节审校逐窗覆盖全章，检查指代、论证、术语、风格和数字对应关系；每个问题须提供可在原文/译文中定位的证据。问题进入现有段落修订闭环，随后对实际文本重新做章节检查。最多 `review.book_revision_rounds` 轮，跨中断保留预算。
- 修改术语会触发相关段落复审，修改全书风格会触发全书复审。章节摘要基于完整原文分批生成；译文变化会刷新摘要关联的版本，原文概要可直接复用，避免从错误译文归纳事实。审校的例外和未解决问题保留在报告中。

例如，以下字段完全可选；不填写时无需人工介入：

```yaml
translation:
  glossary:
    Brier score: 布里尔评分
  terminology:
    superforecaster:
      target: 超级预测者
      aliases: ["superforecasters"]
      allowed_variants: []
      meaning: 本书对高准确率预测者的专称
  style:
    tone: 准确、自然的中文科普说明
    rules: 保留原文的论证力度，不额外解释；书目中的英文书名可保留。
```

### 完成与定稿的区别

`status: completed` 表示任务运行结束。请同时看 `quality_status`：

| 质量状态 | 含义 |
| --- | --- |
| `draft` | 待审稿：有遗留问题，或必要检查未启用/EPUBCheck 未安装 |
| `finalized` | 本地模型及自动检查全部通过，已冻结本次产物哈希 |
| `blocked` | 硬性质量错误或执行失败，不能视为本次定稿 |
| `not_evaluated` | 原文输出/跳过翻译等未执行翻译验收的路径 |

`reviews/quality.json` 保存段落、章节和术语问题；`reviews/acceptance.json` 保存有效规则、模型配置、译文和产物哈希。每段新生成的初译上下文和命中术语另存于 `translations/contexts/`，审校档案也保留参考上下文。每个输出旁也生成 `.quality.json`，使待审状态随文件交付。`finalized` 是自动验收，不代表人工批准或已经在实体 Kindle 上检查过排版；报告明确标记未执行的视觉检查。默认不要求人工批准才能完成。

自动终审仍是概率性的：数字归属、事实和隐喻依赖语义审校，数字集合检查不能代替理解。PDF/OCR 上游漏识别的内容也不能仅凭抽取文本证明完整；复杂页面仍可能需要按报告排查。内置结构检查始终检查输出，安装 EPUBCheck 后自动执行完整规范校验。

小规模真实本地模型回归（独立目录，不修改现有整书任务）：

```bash
python3 -m tools.evaluate_autonomous --model qwen3.5:latest
# 更短的全流程验证，术语和风格均由模型自动生成：
python3 -m tools.evaluate_autonomous --quick --automatic-only
```

## 第三方核验记录

本实现选择并核验了以下公开接口：

- [Marker](https://github.com/datalab-to/marker)：Apache-2.0；当前 CLI 为 `marker_single INPUT`，支持 `--output_format markdown`、`--output_dir`、`--force_ocr`、`--disable_ocr`。其文档说明默认会按设备在 CPU/MPS 使用 fast 模式，并提示复杂表格/表单可能需要人工检查。
- [Calibre macOS](https://calibre-ebook.com/download_osx)：官方 macOS 页面当前标注 macOS 14 Sonoma 及以上，提供桌面版及 `ebook-convert` 命令。本机实际安装并验证 Calibre 9.16.0。
- [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md)：使用本地 `POST /api/chat`，`stream: false`，模型和 `options` 均配置化。本机已验证 arm64 Ollama 服务和 `qwen3.5:latest`。
- [BabelEbook](https://github.com/nevertiree/babel-ebook)：MIT；其 CLI 支持 Ollama、`bilingual`/`translation_only` 等模式。上游没有 macOS 官方桌面包，但本机已从 Rust 源码构建并验证 BabelEbook CLI 0.5.0；默认仍使用原生 Ollama Adapter，以实现 Python 段落缓存、恢复和审校。

## 实际模型评估

已在 Apple Silicon Mac 上下载并验证 `translategemma:12b` 与 `qwen3.5:latest`，并使用 Project Gutenberg 的 *Frankenstein* Chapter 1 完成相同 12 段的真实本地翻译。评估产物属于本地验证数据，不随公开源码仓库分发。

当前评估结论是 `qwen3.5:latest` 更适合作为文学翻译默认模型；TranslateGemma 12B 速度更快，首次生成的英文残留已由自动重试修复，最终 QA 通过，但文学措辞仍略逊。

长篇翻译时，BookFlow 在 `metadata/translation_memory.json` 中维护术语、风格和完整章节摘要。当前段落相关规则、全书内容概要、当前章节概要和相邻原文进入所有翻译/审校阶段，初译另外获得最近译文及前面章节的简要摘要。术语在本地按词边界、常见复数和别名匹配，只注入相关条目；不会每段发送整份术语库。请求不会按固定 1,500 字符截断，过大的完整请求会明确报错。术语或风格改变后按段落规则指纹复审，未受影响的段落保留；若关闭审校，则重译受影响段落。修改译文后刷新对应章节摘要。

第二次长篇实测使用风格不同的 [Alice's Adventures in Wonderland](https://www.gutenberg.org/ebooks/11)：连续 30 段、12,226 个英文字符，跨越两章，自动 QA 通过；相关评估产物保留在本地，不随公开源码仓库分发。

管理学端到端样本使用 [LibreTexts Principles of Management](https://biz.libretexts.org/Bookshelves/Management/Principles_of_Management) 的公开许可 PDF 前 8 页：Marker 抽取为 EPUB 后，再由 Qwen3.5 翻译；这些样本只用于本地验证，不随公开源码仓库分发。继续处理完整 531 页时把 `pdf.page_range` 留空即可。

## 测试与验证

单元测试不依赖网络、Ollama、Marker 或 Calibre，使用本地临时 EPUB 和假的 Ollama HTTP 服务验证闭环：

```bash
python3 -m unittest discover -s tests -v
python3 -m bookflow --help
```

已覆盖：输入识别、语言检测、EPUB 资源保留、双语/译文输出、翻译缓存、任务恢复、中文跳过翻译、数字 QA、Ollama 请求协议、缺少外部工具的错误路径，以及 CLI 状态查询。真实外部工具验证记录见下方阶段进度；本地模型评估与格式校验均保留了实际产物。LLM 审校会逐段写入 `reviews/review_checkpoint.json`，中断后可恢复；审校响应不是合法 JSON 时会按 `translation.request_retries` 重试，耗尽后记录为协议失败并在恢复时重新审校，不会把模型原始输出直接当作重译建议。合法的 `ok: false` 结果经证据核验后触发有限次数的修订，每次修订后立即复审；非严格模式下达到限次的候选仍可输出，遗留问题保持 `ok: false`。最终质量报告写入 `reviews/quality.json`，检查点与逐段候选档案保留供复核。

## 已知边界

Marker 输出的复杂表格、公式、嵌套布局仍需人工检查；本地 EPUB 适配器保留原生 XHTML 节点，并将译文中的常见粗体、斜体、链接等 Markdown 行内标记转成安全 XHTML，不会自动重建复杂布局。EPUB 内含 DRM/加密标记时明确拒绝处理。
