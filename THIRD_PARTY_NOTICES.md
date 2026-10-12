# Third-party software

AI BookFlow is distributed under the MIT License. The project does not bundle
the external programs or model weights listed below; users install and run
them separately under their own licenses and terms.

| Component | How AI BookFlow uses it | License / notice |
| --- | --- | --- |
| [Marker](https://github.com/datalab-to/marker) | Optional external CLI for PDF extraction and OCR | Apache-2.0; see the upstream repository for its notices and dependencies. |
| [Calibre](https://github.com/kovidgoyal/calibre) | Optional external `ebook-convert` CLI for MOBI/AZW3 conversion | GPLv3; see the upstream repository for its complete copyright and license information. |
| [Ollama](https://github.com/ollama/ollama) | Local HTTP API used for translation and review | The Ollama server is a separate program; see its upstream MIT license. Model licenses are separate. |
| [BabelEbook](https://github.com/nevertiree/babel-ebook) | Optional external CLI translation adapter | MIT; see the upstream repository for its notices and dependencies. |
| [PyYAML](https://github.com/yaml/pyyaml) | Optional YAML configuration parser | MIT; used only when installed. |
| [pytest](https://github.com/pytest-dev/pytest) | Development-only test dependency | MIT; not required at runtime. |

Only the BookFlow source files in this repository are licensed by the
top-level `LICENSE` file. Input books, generated translations, model weights,
and other user-provided or generated content are not part of this repository
and are not relicensed by it.
