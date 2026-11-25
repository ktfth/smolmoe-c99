/* Safe, DOM-only Markdown parser & pretty-printer for the web UI.
 * Never uses innerHTML on model output: every node is created with
 * document.createElement and textContent. */

export type InlineNode =
  | { type: "text"; value: string }
  | { type: "bold"; value: string }
  | { type: "italic"; value: string }
  | { type: "strike"; value: string }
  | { type: "code"; value: string }
  | { type: "link"; label: string; href: string };

export type BlockNode =
  | { type: "heading"; level: number; spans: InlineNode[] }
  | { type: "code"; lang: string; code: string }
  | { type: "hr" }
  | { type: "quote"; lines: InlineNode[][] }
  | { type: "ul"; items: InlineNode[][] }
  | { type: "ol"; items: InlineNode[][] }
  | { type: "table"; headers: InlineNode[][]; rows: InlineNode[][][] }
  | { type: "p"; lines: InlineNode[][] };

export function parseInline(s: string): InlineNode[] {
  const out: InlineNode[] = [];
  let buf = "";
  const flush = () => {
    if (buf) {
      out.push({ type: "text", value: buf });
      buf = "";
    }
  };
  let i = 0;
  while (i < s.length) {
    if (s[i] === "`") {
      const end = s.indexOf("`", i + 1);
      if (end > i + 1) {
        flush();
        out.push({ type: "code", value: s.slice(i + 1, end) });
        i = end + 1;
        continue;
      }
    } else if ((s[i] === "*" && s[i + 1] === "*") || (s[i] === "_" && s[i + 1] === "_")) {
      const delim = s.slice(i, i + 2);
      const end = s.indexOf(delim, i + 2);
      if (end > i + 2) {
        flush();
        out.push({ type: "bold", value: s.slice(i + 2, end) });
        i = end + 2;
        continue;
      }
    } else if (s[i] === "~" && s[i + 1] === "~") {
      const end = s.indexOf("~~", i + 2);
      if (end > i + 2) {
        flush();
        out.push({ type: "strike", value: s.slice(i + 2, end) });
        i = end + 2;
        continue;
      }
    } else if (s[i] === "*" && i + 1 < s.length && s[i + 1] !== " " && s[i + 1] !== "*") {
      const end = s.indexOf("*", i + 1);
      if (end > i + 1 && s[end - 1] !== " ") {
        flush();
        out.push({ type: "italic", value: s.slice(i + 1, end) });
        i = end + 1;
        continue;
      }
    } else if (s[i] === "[") {
      const closeB = s.indexOf("]", i + 1);
      if (closeB > i + 1 && s[closeB + 1] === "(") {
        const closeP = s.indexOf(")", closeB + 2);
        if (closeP > closeB + 2) {
          flush();
          out.push({ type: "link", label: s.slice(i + 1, closeB), href: s.slice(closeB + 2, closeP).trim() });
          i = closeP + 1;
          continue;
        }
      }
    }
    buf += s[i];
    ++i;
  }
  flush();
  return out;
}

function isHr(line: string): boolean {
  const t = line.trim();
  return /^(-{3,}|\*{3,}|_{3,})$/.test(t.replace(/\s+/g, ""));
}

function isTableSep(line: string): boolean {
  return line.includes("|") && line.includes("-") && /^[\s|:-]+$/.test(line);
}

function splitTableRow(line: string): string[] {
  let t = line.trim();
  if (t.startsWith("|")) t = t.slice(1);
  if (t.endsWith("|")) t = t.slice(0, -1);
  return t.split("|").map((c) => c.trim());
}

export function parseMarkdown(text: string): BlockNode[] {
  const lines = text.replace(/\r\n?/g, "\n").split("\n");
  const blocks: BlockNode[] = [];
  let i = 0;

  while (i < lines.length) {
    const ln = lines[i]!;
    const trimmed = ln.trim();
    if (!trimmed) {
      ++i;
      continue;
    }

    // Fenced code block
    const fence = ln.match(/^\s{0,3}```(.*)$/);
    if (fence) {
      const lang = (fence[1] ?? "").trim();
      const codeLines: string[] = [];
      ++i;
      while (i < lines.length && !/^\s{0,3}```\s*$/.test(lines[i]!)) {
        codeLines.push(lines[i]!);
        ++i;
      }
      if (i < lines.length) ++i;
      blocks.push({ type: "code", lang, code: codeLines.join("\n") });
      continue;
    }

    // Heading
    const heading = ln.match(/^\s{0,3}(#{1,4})\s+(.*)$/);
    if (heading) {
      blocks.push({ type: "heading", level: heading[1]!.length, spans: parseInline(heading[2]!) });
      ++i;
      continue;
    }

    // Horizontal rule
    if (isHr(ln)) {
      blocks.push({ type: "hr" });
      ++i;
      continue;
    }

    // Table
    if (ln.includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1]!)) {
      const headers = splitTableRow(ln).map(parseInline);
      const rows: InlineNode[][][] = [];
      i += 2;
      while (i < lines.length && lines[i]!.includes("|") && lines[i]!.trim()) {
        rows.push(splitTableRow(lines[i]!).map(parseInline));
        ++i;
      }
      blocks.push({ type: "table", headers, rows });
      continue;
    }

    // Blockquote
    if (/^\s{0,3}>\s?/.test(ln)) {
      const qLines: InlineNode[][] = [];
      while (i < lines.length && /^\s{0,3}>\s?/.test(lines[i]!)) {
        qLines.push(parseInline(lines[i]!.replace(/^\s{0,3}>\s?/, "")));
        ++i;
      }
      blocks.push({ type: "quote", lines: qLines });
      continue;
    }

    // Unordered list
    if (/^\s*[-*]\s+/.test(ln)) {
      const items: InlineNode[][] = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i]!)) {
        items.push(parseInline(lines[i]!.replace(/^\s*[-*]\s+/, "")));
        ++i;
      }
      blocks.push({ type: "ul", items });
      continue;
    }

    // Ordered list
    if (/^\s*\d+[.)]\s+/.test(ln)) {
      const items: InlineNode[][] = [];
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i]!)) {
        items.push(parseInline(lines[i]!.replace(/^\s*\d+[.)]\s+/, "")));
        ++i;
      }
      blocks.push({ type: "ol", items });
      continue;
    }

    // Paragraph (consecutive plain lines)
    const pLines: InlineNode[][] = [];
    while (
      i < lines.length &&
      lines[i]!.trim() &&
      !/^\s{0,3}```/.test(lines[i]!) &&
      !/^\s{0,3}#{1,4}\s+/.test(lines[i]!) &&
      !isHr(lines[i]!) &&
      !/^\s{0,3}>\s?/.test(lines[i]!) &&
      !/^\s*[-*]\s+/.test(lines[i]!) &&
      !/^\s*\d+[.)]\s+/.test(lines[i]!) &&
      !(lines[i]!.includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1]!))
    ) {
      pLines.push(parseInline(lines[i]!));
      ++i;
    }
    if (pLines.length) blocks.push({ type: "p", lines: pLines });
  }

  return blocks;
}

function appendInline(parent: Node, spans: InlineNode[]): void {
  for (const sp of spans) {
    switch (sp.type) {
      case "text":
        parent.appendChild(document.createTextNode(sp.value));
        break;
      case "bold": {
        const el = document.createElement("strong");
        el.textContent = sp.value;
        parent.appendChild(el);
        break;
      }
      case "italic": {
        const el = document.createElement("em");
        el.textContent = sp.value;
        parent.appendChild(el);
        break;
      }
      case "strike": {
        const el = document.createElement("del");
        el.textContent = sp.value;
        parent.appendChild(el);
        break;
      }
      case "code": {
        const el = document.createElement("code");
        el.className = "md-inline-code";
        el.textContent = sp.value;
        parent.appendChild(el);
        break;
      }
      case "link": {
        const safe = /^https?:\/\//i.test(sp.href);
        if (safe) {
          const a = document.createElement("a");
          a.href = sp.href;
          a.target = "_blank";
          a.rel = "noopener noreferrer";
          a.textContent = sp.label;
          parent.appendChild(a);
        } else {
          parent.appendChild(document.createTextNode(sp.label));
        }
        break;
      }
    }
  }
}

export function renderMarkdown(
  text: string,
  onCopy?: (value: string, button: HTMLButtonElement) => void,
): DocumentFragment {
  const frag = document.createDocumentFragment();
  for (const block of parseMarkdown(text)) {
    switch (block.type) {
      case "heading": {
        const tag = `h${Math.min(4, Math.max(1, block.level))}` as "h1" | "h2" | "h3" | "h4";
        const h = document.createElement(tag);
        appendInline(h, block.spans);
        frag.appendChild(h);
        break;
      }
      case "code": {
        const wrap = document.createElement("div");
        wrap.className = "md-code-block";
        const head = document.createElement("div");
        head.className = "md-code-head";
        const lang = document.createElement("span");
        lang.textContent = block.lang || "código";
        head.appendChild(lang);
        if (onCopy) {
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "copy-btn";
          btn.textContent = "Copiar código";
          btn.onclick = () => onCopy(block.code, btn);
          head.appendChild(btn);
        }
        const pre = document.createElement("pre");
        const code = document.createElement("code");
        code.textContent = block.code;
        pre.appendChild(code);
        wrap.append(head, pre);
        frag.appendChild(wrap);
        break;
      }
      case "hr":
        frag.appendChild(document.createElement("hr"));
        break;
      case "quote": {
        const bq = document.createElement("blockquote");
        block.lines.forEach((ln, idx) => {
          if (idx > 0) bq.appendChild(document.createElement("br"));
          appendInline(bq, ln);
        });
        frag.appendChild(bq);
        break;
      }
      case "ul":
      case "ol": {
        const list = document.createElement(block.type);
        for (const item of block.items) {
          const li = document.createElement("li");
          appendInline(li, item);
          list.appendChild(li);
        }
        frag.appendChild(list);
        break;
      }
      case "table": {
        const table = document.createElement("table");
        const thead = document.createElement("thead");
        const trH = document.createElement("tr");
        for (const h of block.headers) {
          const th = document.createElement("th");
          appendInline(th, h);
          trH.appendChild(th);
        }
        thead.appendChild(trH);
        table.appendChild(thead);
        const tbody = document.createElement("tbody");
        for (const row of block.rows) {
          const tr = document.createElement("tr");
          for (let c = 0; c < block.headers.length; ++c) {
            const td = document.createElement("td");
            if (row[c]) appendInline(td, row[c]!);
            tr.appendChild(td);
          }
          tbody.appendChild(tr);
        }
        table.appendChild(tbody);
        frag.appendChild(table);
        break;
      }
      case "p": {
        const p = document.createElement("p");
        block.lines.forEach((ln, idx) => {
          if (idx > 0) p.appendChild(document.createElement("br"));
          appendInline(p, ln);
        });
        frag.appendChild(p);
        break;
      }
    }
  }
  return frag;
}
