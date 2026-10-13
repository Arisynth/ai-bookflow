# AI BookFlow

AI BookFlow 是一个面向 macOS Apple Silicon 的本地电子书处理与翻译 CLI。它将 PDF、EPUB、MOBI 和 AZW3 处理成可恢复、可校验的本地翻译任务，默认通过 Ollama 在本机完成模型调用，不访问云端 API，也不尝试绕过 DRM。

项目代码采用 MIT License。Marker、Calibre、Ollama、BabelEbook、PyYAML 和 pytest 等外部组件不随项目源码分发，许可证与使用说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 功能概览

- 输入：PDF、EPUB、无 DRM 的 MOBI/AZW/AZW3。
- PDF：通过 Marker 完成文字层、布局和 OCR 处理，再转换为 EPUB。
- MOBI/AZW3：通过 Calibre `ebook-convert` 转换为 EPUB。
- 翻译：默认使用 Ollama `/api/chat`；支持 `original`、`translated` 和 `bilingual` 三种输出模式。
- EPUB：保留原有资源、OPF、spine、图片和 CSS；支持常见 Markdown 行内标记、脚注链接和图片占位符。
- 可靠性：段落级缓存、稳定 ID、增量 checkpoint、失败恢复和可配置指数退避重试。
- 一致性：全书分批扫描术语和原文摘要，自动推断文体；人工术语、词汇表和风格配置优先。
- 质量控制：数字、空译、脚注/图片占位符和 EPUB 结构检查；可选的段落审校、章节审校、有限修订和 EPUBCheck。
- 输出：EPUB；配置 `azw3` 后可通过 Calibre 额外生成 AZW3。

## 环境要求

- macOS Apple Silicon
- Python 3.11 或更高版本
- Ollama 及至少一个本地翻译模型
- PDF 输入需要 Marker
- MOBI/AZW3 输入或 AZW3 输出需要 Calibre
- 完整 EPUB 规范校验需要 EPUBCheck

运行时核心代码只依赖 Python 标准库；YAML 配置和开发测试建议安装项目可选依赖。

## 安装

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -e '.[yaml,dev]'
cp config.example.yaml config.yaml
```

安装并启动 Ollama，然后下载配置中的模型：

```bash
ollama pull qwen3.5:latest
```

安装外部工具：

```bash
./.venv/bin/python -m pip install -i https://pypi.org/simple marker-pdf
brew install --cask calibre
brew install epubcheck
```

`config.example.yaml` 默认使用 `.venv/bin/marker_single`、`ebook-convert` 和 `epubcheck`。安装完成后可检查本机工具：

```bash
bookflow doctor --config config.yaml
```

BabelEbook 是可选适配器。它需要单独安装 CLI 并在配置中设置 `translation.engine: babelebook`；该路径不提供 BookFlow 的全书一致性、统一审校和自动验收能力，适合明确关闭这些功能的场景。

## 使用

```bash
# EPUB → 双语 EPUB
bookflow convert book.epub --config config.yaml --output ./out

# PDF → 双语 EPUB；--source 用于覆盖自动语言识别
bookflow convert book.pdf --config config.yaml --source en --mode bilingual

# MOBI/AZW3 → 译文 EPUB/AZW3
bookflow convert book.azw3 --config config.yaml --target zh-CN

# 查看、恢复任务
bookflow status <job-id> --config config.yaml
bookflow failures <job-id> --config config.yaml
bookflow resume <job-id> --config config.yaml
```

常用选项：

- `--model`、`--review-model`：覆盖翻译或审校模型。
- `--source`、`--target`：覆盖源语言或目标语言。
- `--mode original|translated|bilingual`：选择输出模式。
- `--no-review`：关闭模型审校。
- `--no-validate`：关闭内置 EPUB 校验。
- `--workspace`：指定任务工作目录。

任务默认写入 `~/.bookflow/<job-id>/`，输入文件保持只读。任务目录通常包含：

```text
source/                 原始输入
normalized/             规范化后的 EPUB
metadata/               书籍信息、术语、摘要和准备 checkpoint
translations/           翻译缓存、上下文和段落结果
reviews/                审校记录、质量报告和验收结果
output/                 最终 EPUB/AZW3
state.json              任务状态
```

如果 Ollama 或其他外部工具中断，修复环境后运行 `resume`。已完成的阶段和段落会复用 checkpoint，不会无条件重复请求模型。

## 配置要点

完整配置见 [config.example.yaml](config.example.yaml)。常用配置如下：

```yaml
models:
  translation: qwen3.5:latest
  review: qwen3.5:latest

processing:
  target_language: zh-CN
  output_mode: bilingual

translation:
  engine: ollama
  consistency_enabled: true
  term_extraction_enabled: true
  chapter_summary_enabled: true
  glossary: {}
  terminology: {}
  style: {}

review:
  enabled: true
  chapter_enabled: true
  strict: false

output:
  formats: ["epub"]
  validate_epub: true
```

全书准备阶段无需人工填写术语或风格。模型会按章节和语义边界扫描整本书，并将每个批次原子写入 `metadata/preparation.json`。人工 `glossary`、`terminology` 和 `style` 始终优先。

术语决策必须能回溯到原文证据：无法验证的普通词、虚构来源和不可靠译法会被丢弃；多候选术语的缺失决策会采用已扫描候选并记录警告。模型返回的空响应、截断 JSON 或夹带说明文字会按 `translation.request_retries` 重试。

审校配置中的 `review.minor_revision_rounds`、`review.serious_revision_rounds` 和 `review.book_revision_rounds` 分别控制段落问题的分级修订次数与章节问题的修订次数。`review.strict: true` 时，未通过全书准备、段落/章节审校或 EPUBCheck 的任务不会作为定稿完成。

## 任务状态与质量报告

`status: completed` 只表示任务运行结束，还需查看 `quality_status`：

| 状态 | 含义 |
| --- | --- |
| `finalized` | 模型审校和自动检查通过，并记录了产物哈希 |
| `draft` | 任务完成但仍需人工审阅，或部分质量门禁未启用 |
| `blocked` | 发生硬性质量错误或执行失败 |
| `not_evaluated` | 原文输出或跳过翻译，未执行翻译验收 |

主要报告：

- `reviews/quality.json`：确定性检查、段落审校、章节审校和术语问题。
- `reviews/acceptance.json`：质量状态、规则、模型配置和产物哈希。
- `reviews/segments/<segment-id>.json`：逐段候选、审校意见、修订历史和恢复信息。
- `output/*.quality.json`：随输出文件交付的质量摘要。

`finalized` 表示自动验收通过，不等同于人工批准，也不代表已经在实体 Kindle 上完成排版检查。复杂 PDF 页面、表格、公式和 OCR 结果仍应结合实际产物人工复核。

## 测试

测试不依赖网络、Ollama、Marker 或 Calibre，使用临时 EPUB 和模拟 Ollama 服务验证主要流程：

```bash
python3 -m unittest discover -s tests -q
python3 -m bookflow --help
```

当前回归套件包含 70 项测试，覆盖输入识别、语言检测、EPUB 资源保留、翻译缓存、任务恢复、术语与风格一致性、数字和占位符检查、审校修订、质量验收以及 CLI 状态查询。

## 已知边界

- 不处理 DRM 或加密电子书。
- Marker 对复杂表格、公式和嵌套布局的还原仍可能需要人工检查。
- 自动质量检查不能替代对事实、语义、隐喻和版式的人工判断。
- 完整流程的耗时取决于书籍规模、模型、上下文配置和本机硬件。
