#include "mdterm.h"

#include <string.h>

static void emit_raw(Output *out, const char *s, size_t n) {
    if (n) output_bytes(out, (const unsigned char *)s, (uint32_t)n);
}

static void emit_str(Output *out, const char *s) { emit_raw(out, s, strlen(s)); }

static int is_hr(const char *s, size_t len) {
    size_t i = 0, count = 0;
    char ch = 0;
    while (i < len && s[i] == ' ') ++i;
    for (; i < len; ++i) {
        if (s[i] == ' ' || s[i] == '\t') continue;
        if (s[i] != '-' && s[i] != '*' && s[i] != '_') return 0;
        if (!ch) ch = s[i];
        else if (s[i] != ch) return 0;
        ++count;
    }
    return count >= 3;
}

static int find_closing(const char *s, size_t len, size_t from, const char *needle, size_t nlen, size_t *pos) {
    size_t i;
    if (nlen == 0 || from + nlen > len) return 0;
    for (i = from; i + nlen <= len; ++i) {
        if (memcmp(s + i, needle, nlen) == 0) {
            *pos = i;
            return 1;
        }
    }
    return 0;
}

static void emit_inline(Output *out, const char *s, size_t len, const char *restore) {
    size_t i = 0;
    while (i < len) {
        size_t end;
        if (s[i] == '`') {
            if (find_closing(s, len, i + 1, "`", 1, &end) && end > i + 1) {
                emit_str(out, "\033[33m");
                emit_raw(out, s + i + 1, end - (i + 1));
                emit_str(out, "\033[0m");
                if (restore) emit_str(out, restore);
                i = end + 1;
                continue;
            }
        } else if (i + 1 < len && s[i] == '*' && s[i + 1] == '*') {
            if (find_closing(s, len, i + 2, "**", 2, &end) && end > i + 2) {
                emit_str(out, "\033[1m");
                emit_raw(out, s + i + 2, end - (i + 2));
                emit_str(out, "\033[22m");
                if (restore) emit_str(out, restore);
                i = end + 2;
                continue;
            }
        } else if (s[i] == '*' && i + 1 < len && s[i + 1] != ' ' && s[i + 1] != '*') {
            if (find_closing(s, len, i + 1, "*", 1, &end) && end > i + 1 && s[end - 1] != ' ') {
                emit_str(out, "\033[3m");
                emit_raw(out, s + i + 1, end - (i + 1));
                emit_str(out, "\033[23m");
                if (restore) emit_str(out, restore);
                i = end + 1;
                continue;
            }
        }
        emit_raw(out, s + i, 1);
        ++i;
    }
}

static void render_line(MdTerm *m, const char *s, size_t len, int has_nl) {
    size_t indent = 0;
    if (len && s[len - 1] == '\r') --len;
    while (indent < len && indent < 3 && s[indent] == ' ') ++indent;

    /* Fenced code block delimiter: ```[lang] */
    if (len - indent >= 3 && memcmp(s + indent, "```", 3) == 0) {
        if (!m->in_code) {
            size_t ls = indent + 3, le = len;
            while (ls < le && (s[ls] == ' ' || s[ls] == '\t')) ++ls;
            while (le > ls && (s[le - 1] == ' ' || s[le - 1] == '\t')) --le;
            m->in_code = 1;
            emit_str(m->out, "\033[2m───");
            if (le > ls) {
                emit_str(m->out, " ");
                emit_raw(m->out, s + ls, le - ls);
                emit_str(m->out, " ───");
            } else {
                emit_str(m->out, "───────────");
            }
            emit_str(m->out, "\033[0m");
        } else {
            m->in_code = 0;
            emit_str(m->out, "\033[2m──────────────\033[0m");
        }
        if (has_nl) emit_str(m->out, "\n");
        return;
    }

    if (m->in_code) {
        emit_str(m->out, "  \033[36m");
        emit_raw(m->out, s, len);
        emit_str(m->out, "\033[0m");
        if (has_nl) emit_str(m->out, "\n");
        return;
    }

    /* Heading: # ... ###### */
    if (indent < len && s[indent] == '#') {
        size_t h = indent;
        while (h < len && h < indent + 6 && s[h] == '#') ++h;
        if (h < len && s[h] == ' ') {
            emit_str(m->out, "\033[1;36m");
            emit_inline(m->out, s + h + 1, len - (h + 1), "\033[1;36m");
            emit_str(m->out, "\033[0m");
            if (has_nl) emit_str(m->out, "\n");
            return;
        }
    }

    /* Horizontal rule */
    if (is_hr(s, len)) {
        emit_str(m->out, "\033[2m────────────────────────\033[0m");
        if (has_nl) emit_str(m->out, "\n");
        return;
    }

    /* Blockquote */
    if (indent < len && s[indent] == '>') {
        size_t off = indent + 1;
        if (off < len && s[off] == ' ') ++off;
        emit_str(m->out, "\033[2m│ \033[3m");
        emit_inline(m->out, s + off, len - off, "\033[2;3m");
        emit_str(m->out, "\033[0m");
        if (has_nl) emit_str(m->out, "\n");
        return;
    }

    /* Unordered bullet list */
    {
        size_t sp = 0;
        while (sp < len && s[sp] == ' ') ++sp;
        if (sp + 1 < len && (s[sp] == '-' || s[sp] == '*') && s[sp + 1] == ' ') {
            emit_raw(m->out, s, sp);
            emit_str(m->out, "\033[36m• \033[0m");
            emit_inline(m->out, s + sp + 2, len - (sp + 2), NULL);
            if (has_nl) emit_str(m->out, "\n");
            return;
        }
    }

    emit_inline(m->out, s, len, NULL);
    if (has_nl) emit_str(m->out, "\n");
}

void mdterm_init(MdTerm *m, Output *out, int enabled) {
    memset(m, 0, sizeof(*m));
    m->out = out;
    m->enabled = enabled;
}

void mdterm_bytes(MdTerm *m, const unsigned char *p, uint32_t n) {
    uint32_t i;
    if (!m->enabled) {
        output_bytes(m->out, p, n);
        return;
    }
    for (i = 0; i < n; ++i) {
        char c = (char)p[i];
        if (c == '\n') {
            render_line(m, m->line, m->len, 1);
            m->len = 0;
        } else {
            if (m->len >= sizeof(m->line) - 1) {
                render_line(m, m->line, m->len, 0);
                m->len = 0;
            }
            m->line[m->len++] = c;
        }
    }
}

void mdterm_finish(MdTerm *m) {
    if (m->enabled) {
        if (m->len) {
            render_line(m, m->line, m->len, 0);
            m->len = 0;
        }
        if (m->in_code) {
            emit_str(m->out, "\033[0m");
            m->in_code = 0;
        }
    }
    output_finish(m->out);
}
