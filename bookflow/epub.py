from __future__ import annotations

import hashlib
import io
import mimetypes
import posixpath
import re
import zipfile
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

from .errors import BookFlowError, QualityError

XHTML_NS = "http://www.w3.org/1999/xhtml"
CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"
OPF_NS = "http://www.idpf.org/2007/opf"
DC_NS = "http://purl.org/dc/elements/1.1/"
EPUB_NS = "http://www.idpf.org/2007/ops"
ET.register_namespace("", XHTML_NS)
ET.register_namespace("opf", OPF_NS)
ET.register_namespace("dc", DC_NS)

BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "figcaption", "caption", "dt", "dd", "td", "th"}
SKIP_TAGS = {"script", "style", "code", "pre", "svg"}
FOOTNOTE_FRAGMENT_RE = re.compile(r"(?:^|[-_])nts\d+[a-z]?$", re.IGNORECASE)
FOOTNOTE_TOKEN_TEMPLATE = "[[BOOKFLOW_FOOTNOTE_{}]]"


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def text_content(element: ET.Element) -> str:
    return re.sub(r"\s+", " ", "".join(element.itertext())).strip()


def _is_footnote_anchor(element: ET.Element) -> bool:
    """Recognize common EPUB footnote references and backlinks.

    The source EPUBs we support use either EPUB's ``epub:type=noteref`` or
    the common ``*-nts1``/``*-nts1a`` id convention.  Ordinary web links are
    deliberately excluded, even when they contain a fragment.
    """
    if local_name(element.tag) != "a":
        return False
    epub_type = element.attrib.get(f"{{{EPUB_NS}}}type", "").casefold()
    href = element.attrib.get("href", "")
    fragment = href.rsplit("#", 1)[-1] if "#" in href else ""
    visible = re.sub(r"\s+", " ", "".join(element.itertext())).strip()
    numbered = bool(re.fullmatch(r"\d+(?:\.\s*)?", visible))
    return epub_type == "noteref" or (numbered and bool(FOOTNOTE_FRAGMENT_RE.search(fragment)))


def _footnote_refs(element: ET.Element) -> list[tuple[str, ET.Element]]:
    refs: list[tuple[str, ET.Element]] = []
    for child in element.iter():
        if _is_footnote_anchor(child):
            refs.append((FOOTNOTE_TOKEN_TEMPLATE.format(len(refs) + 1), child))
    image_number = 0
    for child in element.iter():
        if local_name(child.tag) == "img":
            image_number += 1
            refs.append((f"[[BOOKFLOW_IMAGE_{image_number}]]", child))
    return refs


def _segment_text(element: ET.Element) -> str:
    """Extract translatable text while replacing footnote markup by tokens."""
    refs = {id(anchor): token for token, anchor in _footnote_refs(element)}
    parts: list[str] = []

    def visit(node: ET.Element) -> None:
        token = refs.get(id(node))
        if token is not None:
            parts.append(token)
            return
        if node.text:
            parts.append(node.text)
        for child in list(node):
            visit(child)
            if child.tail:
                parts.append(child.tail)

    visit(element)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def has_block_descendant(element: ET.Element) -> bool:
    return any(local_name(child.tag) in BLOCK_TAGS or "bookflow-source-run" in child.get("class", "").split()
               for child in list(element.iter())[1:])


def _prepare_text_blocks(root: ET.Element) -> None:
    """Expose loose/mixed container text without flattening nested paragraphs.

    The same deterministic normalization runs for inspection and rebuilding,
    so stable IDs describe precisely the text that is replaced in the output.
    """
    containers = {"body", "div", "section", "article", "aside", "header", "footer", "main", "nav", "figure"}
    for node in list(root.iter())[::-1]:
        tag = local_name(node.tag)
        if tag not in containers and not (tag in BLOCK_TAGS and has_block_descendant(node)):
            continue
        children = list(node)
        if not (node.text and node.text.strip()) and all(
                (local_name(child.tag) in BLOCK_TAGS | containers | SKIP_TAGS or "bookflow-source-run" in child.get("class", "").split()) and not (child.tail and child.tail.strip())
                for child in children):
            continue
        leading = node.text
        node.text = None
        for child in children:
            node.remove(child)
        run = ET.Element(f"{{{XHTML_NS}}}span", {"class": "bookflow-source-run"})
        run.text = leading

        def flush() -> None:
            nonlocal run
            if run.text or len(run):
                node.append(run)
            run = ET.Element(f"{{{XHTML_NS}}}span", {"class": "bookflow-source-run"})

        for child in children:
            if local_name(child.tag) in BLOCK_TAGS | containers | SKIP_TAGS or has_block_descendant(child) or "bookflow-source-run" in child.get("class", "").split():
                flush()
                tail = child.tail
                child.tail = None
                node.append(child)
                run.text = tail
            else:
                run.append(child)
        flush()


def _translatable(element: ET.Element) -> bool:
    return (local_name(element.tag) in BLOCK_TAGS or "bookflow-source-run" in element.get("class", "").split()) and not has_block_descendant(element)


@dataclass
class Segment:
    segment_id: str
    chapter: str
    text: str
    file_name: str
    ordinal: int


@dataclass
class EpubBook:
    path: Path
    opf_path: str
    spine_files: list[str]
    title: str
    segments: list[Segment]
    legacy_ids: dict[str, str] = field(default_factory=dict)


def _read_xml(zf: zipfile.ZipFile, name: str) -> ET.Element:
    try:
        return ET.fromstring(zf.read(name))
    except KeyError as exc:
        raise BookFlowError(f"EPUB 缺少必要文件: {name}") from exc
    except ET.ParseError as exc:
        raise BookFlowError(f"EPUB XML 损坏: {name}: {exc}") from exc


def _resolve(base: str, href: str) -> str:
    return posixpath.normpath(posixpath.join(posixpath.dirname(base), href.split("#", 1)[0]))


def _opf_details(zf: zipfile.ZipFile) -> tuple[str, str, list[str]]:
    container = _read_xml(zf, "META-INF/container.xml")
    rootfile = container.find(f".//{{{CONTAINER_NS}}}rootfile")
    if rootfile is None or not rootfile.attrib.get("full-path"):
        raise BookFlowError("EPUB 无法定位 OPF 包文件")
    opf_path = rootfile.attrib["full-path"]
    opf = _read_xml(zf, opf_path)
    manifest: dict[str, str] = {}
    for item in opf.findall(f".//{{{OPF_NS}}}item"):
        if item.attrib.get("id") and item.attrib.get("href"):
            manifest[item.attrib["id"]] = _resolve(opf_path, item.attrib["href"])
    spine_files = []
    for itemref in opf.findall(f".//{{{OPF_NS}}}itemref"):
        href = manifest.get(itemref.attrib.get("idref", ""))
        if href:
            spine_files.append(href)
        else:
            raise BookFlowError("EPUB spine 引用了不存在的 manifest 条目")
    title = ""
    title_el = opf.find(f".//{{{DC_NS}}}title")
    if title_el is not None and title_el.text:
        title = title_el.text.strip()
    return opf_path, title or "BookFlow book", spine_files


def assert_not_encrypted(zf: zipfile.ZipFile) -> None:
    names = set(zf.namelist())
    if "META-INF/rights.xml" in names or "META-INF/encryption.xml" in names:
        raise BookFlowError("检测到可能的 DRM/加密 EPUB；BookFlow 不破解 DRM")


def inspect_epub(path: Path) -> EpubBook:
    with zipfile.ZipFile(path) as zf:
        assert_not_encrypted(zf)
        opf_path, title, spine_files = _opf_details(zf)
        segments: list[Segment] = []
        ordinal = 0
        legacy_ordinal = 0
        legacy_ids = {}
        legacy_tags = BLOCK_TAGS - {"caption", "dt", "dd"}
        for chapter_index, file_name in enumerate(spine_files, start=1):
            try:
                root = ET.fromstring(zf.read(file_name))
            except KeyError:
                raise BookFlowError(f"EPUB spine 指向缺失的章节: {file_name}")
            except ET.ParseError as exc:
                raise BookFlowError(f"章节 XML 损坏: {file_name}: {exc}") from exc
            old_ids = {}
            for node in root.iter():
                if local_name(node.tag) not in legacy_tags or any(local_name(c.tag) in legacy_tags for c in list(node.iter())[1:]):
                    continue
                old_text = re.sub(r"\[\[BOOKFLOW_IMAGE_\d+\]\]", "", _segment_text(node))
                old_text = re.sub(r"\s+", " ", old_text).strip()
                if len(old_text) < 2:
                    continue
                legacy_ordinal += 1
                old_ids[id(node)] = hashlib.sha256(f"{file_name}\0{legacy_ordinal}\0{old_text}".encode()).hexdigest()[:20]
            _prepare_text_blocks(root)
            skipped = {id(child) for node in root.iter() if local_name(node.tag) in SKIP_TAGS for child in node.iter()}
            for element in root.iter():
                tag = local_name(element.tag)
                if not _translatable(element) or id(element) in skipped:
                    continue
                text = _segment_text(element)
                if not text:
                    continue
                ordinal += 1
                segment_id = hashlib.sha256(f"{file_name}\0{ordinal}\0{text}".encode()).hexdigest()[:20]
                if id(element) in old_ids:
                    legacy_ids[segment_id] = old_ids[id(element)]
                segments.append(Segment(segment_id, f"chapter-{chapter_index}", text, file_name, ordinal))
    if not spine_files:
        raise BookFlowError("EPUB 没有可处理的 spine 章节")
    return EpubBook(path, opf_path, spine_files, title, segments, legacy_ids)


def book_text(book: EpubBook) -> str:
    return "\n".join(segment.text for segment in book.segments)


def _set_inline_markdown(
    element: ET.Element,
    value: str,
    footnote_refs: list[tuple[str, ET.Element]] | None = None,
) -> None:
    """Set XHTML from model Markdown and restore original footnote links."""
    footnote_markup: dict[str, str] = {}
    for token, anchor in footnote_refs or []:
        cloned = deepcopy(anchor)
        # The source and translated paragraphs share one XHTML document. Do
        # not duplicate the source anchor's ID in the inserted translation.
        cloned.attrib.pop("id", None)
        # ElementTree serializes an element's tail by default; the tail must
        # remain in the translated paragraph around the restored anchor,
        # otherwise it would be inserted twice.
        cloned.tail = None
        footnote_markup[token] = ET.tostring(cloned, encoding="unicode")
    fragment = _inline_markdown(value, Path("."), {})
    for token, markup in footnote_markup.items():
        fragment = fragment.replace(_xml_escape(token), markup)
    try:
        wrapper = ET.fromstring(f'<span xmlns="{XHTML_NS}">{fragment}</span>')
    except ET.ParseError as exc:
        raise QualityError(f"译文包含无法解析的内联标记: {value[:120]}") from exc
    element.text = wrapper.text
    for child in list(wrapper):
        wrapper.remove(child)
        element.append(child)


def _make_translation_element(original: ET.Element, translated: str) -> ET.Element:
    tag = original.tag if isinstance(original.tag, str) else f"{{{XHTML_NS}}}p"
    attributes = {key: value for key, value in original.attrib.items() if key != "id"}
    element = ET.Element(tag, attributes)
    element.set("class", ((element.get("class") or "") + " bookflow-translation").strip())
    _set_inline_markdown(element, translated, _footnote_refs(original))
    return element


def _replace_text(element: ET.Element, translated: str) -> None:
    footnote_refs = _footnote_refs(element)
    for child in list(element):
        element.remove(child)
    _set_inline_markdown(element, translated, footnote_refs)


def build_translated_epub(
    source: Path,
    destination: Path,
    translations: dict[str, str],
    mode: str,
) -> None:
    book = inspect_epub(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source, "r") as zin, zipfile.ZipFile(destination, "w") as zout:
        assert_not_encrypted(zin)
        # ZIP member order is not required to match the OPF spine order. The
        # segment IDs, however, are assigned in spine order by inspect_epub.
        # Record each file's global ordinal offset so output reconstruction
        # remains stable for books whose archive order is scrambled.
        file_offsets: dict[str, int] = {}
        for segment in book.segments:
            file_offsets.setdefault(segment.file_name, segment.ordinal - 1)
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename in book.spine_files:
                root = ET.fromstring(data)
                _prepare_text_blocks(root)
                skipped = {id(child) for node in root.iter() if local_name(node.tag) in SKIP_TAGS for child in node.iter()}
                parents = {child: parent for parent in root.iter() for child in list(parent)}
                elements = list(root.iter())
                local_ordinal = 0
                for element in elements:
                    if "bookflow-translation" in element.attrib.get("class", "").split():
                        continue
                    tag = local_name(element.tag)
                    if not _translatable(element) or id(element) in skipped:
                        continue
                    text = _segment_text(element)
                    if not text:
                        continue
                    local_ordinal += 1
                    ordinal = file_offsets.get(info.filename, 0) + local_ordinal
                    segment_id = hashlib.sha256(f"{info.filename}\0{ordinal}\0{text}".encode()).hexdigest()[:20]
                    translated = translations.get(segment_id)
                    if translated is None:
                        raise QualityError(f"缺少段落译文: {segment_id}")
                    if mode == "translated":
                        _replace_text(element, translated)
                    elif mode == "bilingual":
                        parent = parents[element]
                        index = list(parent).index(element)
                        parent.insert(index + 1, _make_translation_element(element, translated))
                data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
            zout.writestr(info, data)


def validate_epub(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"输出不存在: {path}"]
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            if bad:
                errors.append(f"ZIP 校验失败: {bad}")
            names = set(zf.namelist())
            for required in ("mimetype", "META-INF/container.xml"):
                if required not in names:
                    errors.append(f"缺少 {required}")
            if "mimetype" in names and zf.read("mimetype") != b"application/epub+zip":
                errors.append("mimetype 内容不正确")
            _opf_details(zf)
    except (zipfile.BadZipFile, BookFlowError) as exc:
        errors.append(str(exc))
    return errors


def markdown_to_epub(markdown_path: Path, destination: Path, title: str | None = None) -> Path:
    """Create a small valid EPUB from Marker markdown output.

    This is intentionally only the PDF normalization bridge; Marker remains
    responsible for OCR, reading order, images, and layout analysis.
    """
    title = title or markdown_path.stem
    lines = markdown_path.read_text(encoding="utf-8", errors="replace").splitlines()
    blocks: list[str] = []
    paragraph: list[str] = []
    image_assets: dict[str, Path] = {}
    chapter_no = 0
    toc_entries: list[dict[str, object]] = []
    in_toc = False
    toc_marker = "__BOOKFLOW_TABLE_OF_CONTENTS__"
    page_anchor_re = re.compile(
        r'^(?P<heading>#{0,6}\s*)?<span\s+id=["\'](?P<anchor>page-[^"\']+)["\']></span>\s*(?P<rest>.*)$'
    )
    heading_re = re.compile(r"^(#{1,6})\s+(.+)$")
    available_page_anchors = {
        match.group("anchor")
        for raw_line in lines
        if (match := page_anchor_re.match(raw_line.strip()))
    }

    def flush() -> None:
        if paragraph:
            value = re.sub(r"\s+", " ", " ".join(paragraph)).strip()
            if value:
                blocks.append(f"<p>{_inline_markdown(value, markdown_path.parent, image_assets)}</p>")
            paragraph.clear()

    for line in lines:
        value = line.strip()
        if not value:
            if in_toc:
                continue
            flush()
            continue

        page_match = page_anchor_re.match(value)
        if page_match:
            in_toc = False
            flush()
            anchor = page_match.group("anchor")
            page_class = "bookflow-page-break" if blocks else "bookflow-page-anchor"
            blocks.append(f'<div id="{_xml_escape(anchor)}" class="{page_class}"></div>')
            heading_prefix = page_match.group("heading") or ""
            value = f"{heading_prefix}{page_match.group('rest')}".strip()
            if not value:
                continue

        match = heading_re.match(value)
        if match:
            level = min(len(match.group(1)), 6)
            heading_text = match.group(2)
            plain_heading = _markdown_plain_text(heading_text)
            if plain_heading.upper() == "TABLE OF CONTENTS":
                flush()
                in_toc = True
                toc_entries = []
                blocks.append(toc_marker)
                continue
            if in_toc:
                links = _markdown_links(heading_text)
                label, href = (links[0] if links else (_markdown_plain_text(heading_text), ""))
                entry: dict[str, object] = {"label": label, "href": href, "children": []}
                toc_entries.append(entry)
                continue
            flush()
            chapter_no += 1
            blocks.append(f'<h{level} id="chapter-{chapter_no}">{_inline_markdown(heading_text, markdown_path.parent, image_assets)}</h{level}>')
        elif in_toc:
            links = _markdown_links(value)
            if links and toc_entries:
                children = toc_entries[-1]["children"]
                assert isinstance(children, list)
                children.extend({"label": label, "href": href} for label, href in links)
        elif re.match(r"^[-*+]\s+", value):
            flush()
            bullet = re.sub(r"^[-*+]\s+", "", value)
            blocks.append(f"<p>{_inline_markdown(bullet, markdown_path.parent, image_assets)}</p>")
        else:
            paragraph.append(value)
    flush()
    if not blocks:
        raise BookFlowError("Marker 未提取到可读文本，请检查 OCR/版面结果")

    destination.parent.mkdir(parents=True, exist_ok=True)
    toc_html = _render_toc(toc_entries, markdown_path.parent, image_assets, available_page_anchors) if toc_entries else ""
    rendered_blocks = "".join(toc_html if block == toc_marker else block for block in blocks)
    chapter = "<?xml version=\"1.0\" encoding=\"utf-8\"?>" + f"""
<html xmlns="{XHTML_NS}" lang="zh-CN"><head><title>{_xml_escape(title)}</title>
<style>
body {{ margin: 5%; line-height: 1.6; font-family: serif; }}
h1 {{ margin: 0 0 1.5em; }} h2 {{ margin: 1.8em 0 0.8em; }} h3 {{ margin: 1.2em 0 0.4em; }}
p {{ margin: 0 0 1em; }}
.bookflow-page-break {{ break-before: page; page-break-before: always; height: 0; margin: 0; padding: 0; }}
.bookflow-page-anchor {{ height: 0; margin: 0; padding: 0; }}
.bookflow-toc {{ margin: 1em 0 2em; }}
.bookflow-toc .toc-title {{ background: #187bb8; color: white; padding: 0.35em 0.5em; }}
.bookflow-toc h3 {{ color: #145d91; font-size: 1.05em; margin: 0.8em 0 0.2em; }}
.bookflow-toc ul {{ margin: 0 0 0.5em 1.4em; padding: 0; list-style: none; }}
.bookflow-toc a {{ color: #145d91; text-decoration: none; }}
.bookflow-toc li {{ margin: 0.2em 0; }}
.bookflow-translation {{ color: #333; }}
</style>
</head><body><h1 id="bookflow-title">{_xml_escape(title)}</h1>{rendered_blocks}</body></html>"""
    modified = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    image_manifest = "".join(
        f'<item id="img-{index}" href="images/{_xml_escape(name)}" media-type="{mimetypes.guess_type(name)[0] or "application/octet-stream"}"/>'
        for index, name in enumerate(image_assets, start=1)
    )
    opf = f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="{OPF_NS}" unique-identifier="book-id" version="3.0">
  <metadata xmlns:dc="{DC_NS}" xmlns:dcterms="http://purl.org/dc/terms/"><dc:identifier id="book-id">bookflow-{hashlib.sha256(title.encode()).hexdigest()[:16]}</dc:identifier><dc:title>{_xml_escape(title)}</dc:title><dc:language>und</dc:language><meta property="dcterms:modified">{modified}</meta></metadata>
  <manifest><item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/><item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/><item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/><item id="css" href="style.css" media-type="text/css"/>{image_manifest}</manifest>
  <spine toc="ncx"><itemref idref="chapter"/></spine>
</package>"""
    nav = f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="{XHTML_NS}" xmlns:epub="http://www.idpf.org/2007/ops"><head><title>Contents</title></head><body><nav epub:type="toc"><ol><li><a href="chapter.xhtml#bookflow-title">{_xml_escape(title)}</a>{_render_nav_toc(toc_entries, available_page_anchors)}</li></ol></nav></body></html>"""
    ncx = _render_ncx_toc(title, toc_entries, available_page_anchors)
    container = f"""<?xml version="1.0" encoding="UTF-8"?><container version="1.0" xmlns="{CONTAINER_NS}"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>"""
    with zipfile.ZipFile(destination, "w") as zout:
        zout.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zout.writestr("META-INF/container.xml", container)
        zout.writestr("OEBPS/content.opf", opf)
        zout.writestr("OEBPS/chapter.xhtml", chapter)
        zout.writestr("OEBPS/nav.xhtml", nav)
        zout.writestr("OEBPS/toc.ncx", ncx)
        zout.writestr("OEBPS/style.css", "body { font-family: serif; }")
        for name, asset in image_assets.items():
            zout.write(asset, f"OEBPS/images/{name}")
    return destination


def _markdown_links(value: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")
    return [(match.group(1), match.group(2).replace("\\)", ")").strip()) for match in pattern.finditer(value)]


def _markdown_plain_text(value: str) -> str:
    value = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", value)
    value = re.sub(r"(?<!!)\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"[*_`]+", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _toc_anchor(label: str, href: str, base_dir: Path, image_assets: dict[str, Path], available_page_anchors: set[str]) -> str:
    rendered = _inline_markdown(label, base_dir, image_assets)
    if href and (not href.startswith("#page-") or href[1:] in available_page_anchors):
        return f'<a href="{_xml_escape(href)}">{rendered}</a>'
    return rendered


def _render_toc(entries: list[dict[str, object]], base_dir: Path, image_assets: dict[str, Path], available_page_anchors: set[str]) -> str:
    parts = ['<div id="table-of-contents" class="bookflow-toc"><h2 class="toc-title">TABLE OF CONTENTS</h2>']
    for entry in entries:
        label = str(entry["label"])
        href = str(entry.get("href", ""))
        parts.append(f'<h3 class="toc-entry">{_toc_anchor(label, href, base_dir, image_assets, available_page_anchors)}</h3>')
        children = entry.get("children", [])
        if children:
            parts.append('<ul class="toc-sections">')
            for child in children:
                parts.append(f'<li>{_toc_anchor(str(child["label"]), str(child.get("href", "")), base_dir, image_assets, available_page_anchors)}</li>')
            parts.append('</ul>')
    parts.append('</div>')
    return "".join(parts)


def _render_nav_toc(entries: list[dict[str, object]], available_page_anchors: set[str]) -> str:
    if not entries:
        return ""
    parts = ["<ol>"]
    for entry in entries:
        label = _xml_escape(str(entry["label"]))
        href = str(entry.get("href", ""))
        valid_href = href and (not href.startswith("#page-") or href[1:] in available_page_anchors)
        children = entry.get("children", [])
        valid_children = [
            child
            for child in children
            if (child_href := str(child.get("href", "")))
            and (not child_href.startswith("#page-") or child_href[1:] in available_page_anchors)
        ]
        if not valid_href and not valid_children:
            continue
        if not valid_href:
            continue
        parts.append(f'<li><a href="{_xml_escape(f"chapter.xhtml{href}")}">{label}</a>')
        if valid_children:
            parts.append("<ol>")
            for child in valid_children:
                child_label = _xml_escape(str(child["label"]))
                child_href = str(child.get("href", ""))
                parts.append(f'<li><a href="{_xml_escape(f"chapter.xhtml{child_href}")}">{child_label}</a></li>')
            parts.append("</ol>")
        parts.append("</li>")
    parts.append("</ol>")
    return "".join(parts)


def _render_ncx_toc(title: str, entries: list[dict[str, object]], available_page_anchors: set[str]) -> str:
    nav_points: list[tuple[str, str]] = [(title, "chapter.xhtml#bookflow-title")]
    for entry in entries:
        href = str(entry.get("href", ""))
        if not href or (href.startswith("#page-") and href[1:] not in available_page_anchors):
            continue
        nav_points.append((str(entry["label"]), f"chapter.xhtml{href}"))
        for child in entry.get("children", []):
            child_href = str(child.get("href", ""))
            if child_href and (not child_href.startswith("#page-") or child_href[1:] in available_page_anchors):
                nav_points.append((str(child["label"]), f"chapter.xhtml{child_href}"))
    points = []
    for index, (label, href) in enumerate(nav_points, start=1):
        points.append(
            f'<navPoint id="navPoint-{index}" playOrder="{index}"><navLabel><text>{_xml_escape(label)}</text></navLabel><content src="{_xml_escape(href)}"/></navPoint>'
        )
    return f'''<?xml version="1.0" encoding="utf-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1"><head><meta name="dtb:uid" content="bookflow-{hashlib.sha256(title.encode()).hexdigest()[:16]}"/></head><docTitle><text>{_xml_escape(title)}</text></docTitle><navMap>{"".join(points)}</navMap></ncx>'''


def _inline_markdown(value: str, base_dir: Path, image_assets: dict[str, Path]) -> str:
    """Render the small Markdown subset emitted by Marker as XHTML inline markup."""
    value = re.sub(r"\\([\\`*{}\[\]()#+.!_-])", r"\1", value)
    placeholders: dict[str, str] = {}

    def keep(rendered: str) -> str:
        token = f"BOOKFLOWTOKEN{len(placeholders)}X"
        placeholders[token] = rendered
        return token

    image_re = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+['\"][^'\"]*['\"])?\)")

    def image(match: re.Match[str]) -> str:
        raw_path = match.group(2).replace("\\", "")
        asset = (base_dir / raw_path).resolve()
        try:
            asset.relative_to(base_dir.resolve())
        except ValueError:
            return _xml_escape(match.group(0))
        if not asset.is_file():
            return _xml_escape(match.group(0))
        name = asset.name
        image_assets[name] = asset
        return keep(f'<img src="images/{_xml_escape(name)}" alt="{_xml_escape(match.group(1))}" />')

    value = image_re.sub(image, value)

    link_re = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")

    def link(match: re.Match[str]) -> str:
        label = _xml_escape(match.group(1))
        href = match.group(2).replace("\\)", ")").strip()
        # Marker emits page-local anchors such as #page-7-0. The normalized
        # EPUB is a single XHTML flow and does not recreate every PDF page
        # anchor, so keep the visible label instead of emitting broken links.
        if not (href.startswith(("https://", "http://", "mailto:"))):
            return label
        return keep(f'<a href="{_xml_escape(href)}">{label}</a>')

    value = link_re.sub(link, value)
    rendered = _xml_escape(value)
    rendered = re.sub(r"`([^`]+)`", r"<code>\1</code>", rendered)
    rendered = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", rendered)
    rendered = re.sub(r"__([^_]+)__", r"<strong>\1</strong>", rendered)
    rendered = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", rendered)
    rendered = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"<em>\1</em>", rendered)
    for token, markup in placeholders.items():
        rendered = rendered.replace(token, markup)
    return rendered


def _xml_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
