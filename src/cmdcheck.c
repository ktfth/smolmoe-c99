#include "cmdcheck.h"
#include "process.h"
#include "util.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RS '\036'               /* separates the snippets of a request */
#define REQUEST_MAX 24576       /* snippet bytes per reply: far more than a 512-token reply holds */
#define FINDINGS_MAX 8192
#define FIRST_ANSWER_MS 20000u  /* PowerShell starting and warming up, on a slow or busy machine */
#define ANSWER_MS 6000u
#define TAG_MAX 16
#define FIELD_MAX 256

/* ---- finding the commands in a reply ------------------------------------ */

/* Get-Verb, the verbs PowerShell approves: prose only counts Verb-Noun words
 * that start with one, so "Wi-Fi" and "Plug-and-Play" are left alone. */
static const char *const VERBS[] = {
    "Add", "Approve", "Assert", "Backup", "Block", "Build", "Checkpoint", "Clear", "Close", "Compare", "Complete",
    "Compress", "Confirm", "Connect", "Convert", "ConvertFrom", "ConvertTo", "Copy", "Debug", "Deny", "Deploy",
    "Disable", "Disconnect", "Dismount", "Edit", "Enable", "Enter", "Exit", "Expand", "Export", "Find", "Format",
    "Get", "Grant", "Group", "Hide", "Import", "Initialize", "Install", "Invoke", "Join", "Limit", "Lock", "Measure",
    "Merge", "Mount", "Move", "New", "Open", "Optimize", "Out", "Ping", "Pop", "Protect", "Publish", "Push", "Read",
    "Receive", "Redo", "Register", "Remove", "Rename", "Repair", "Request", "Reset", "Resize", "Resolve", "Restart",
    "Restore", "Resume", "Revoke", "Save", "Search", "Select", "Send", "Set", "Show", "Skip", "Split", "Start", "Step",
    "Stop", "Submit", "Suspend", "Switch", "Sync", "Test", "Trace", "Unblock", "Undo", "Uninstall", "Unlock",
    "Unprotect", "Unpublish", "Unregister", "Update", "Use", "Wait", "Watch", "Write"};

/* Fence tags: checked by either kind, by one kind (the other names them as
 * not checked), or by neither (named). Other languages are skipped. */
static const char *const BOTH_TAGS[] = {"", "shell", "console", "terminal"};
static const char *const POWERSHELL_TAGS[] = {"powershell", "posh", "ps", "ps1", "psm1", "pwsh"};
static const char *const POSIX_TAGS[] = {"bash", "sh", "zsh", "ksh", "dash", "ash"};
static const char *const OTHER_SHELL_TAGS[] = {"fish", "cmd", "bat", "batch"};
#define COUNT(a) (sizeof(a) / sizeof((a)[0]))

static int is_alpha(char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }
static int is_upper(char c) { return c >= 'A' && c <= 'Z'; }
static int is_digit(char c) { return c >= '0' && c <= '9'; }
static int is_blank(char c) { return c == ' ' || c == '\t'; }

static int in_list(const char *s, size_t n, const char *const *list, size_t count) {
    size_t i;
    for (i = 0; i < count; ++i)
        if (strlen(list[i]) == n && !memcmp(list[i], s, n)) return 1;
    return 0;
}

/* Length of the Verb-Noun word at s[i..e), or 0. `approved`: the verb must be
 * one of VERBS and the noun capitalized (prose); else any letters (code). */
static size_t verb_noun(const char *s, size_t i, size_t e, int approved) {
    size_t v = i, k;
    while (v < e && is_alpha(s[v])) ++v;
    if (v == i || v + 1 >= e || s[v] != '-') return 0;
    if (approved ? !is_upper(s[v + 1]) : !is_alpha(s[v + 1])) return 0;
    if (approved && !in_list(s + i, v - i, VERBS, sizeof(VERBS) / sizeof(VERBS[0]))) return 0;
    for (k = v + 1; k < e && (is_alpha(s[k]) || is_digit(s[k])); ++k) {}
    return k - i;
}

/* Prose may start a command at s[i]: the text start, or after a character
 * that cannot be part of a word, path or variable. */
static int word_start(const char *s, size_t line, size_t i) {
    char p;
    if (i == line) return 1;
    p = s[i - 1];
    return !is_alpha(p) && !is_digit(p) && !strchr("-_$\\/.:", p);
}

typedef struct Request {
    char *out;
    size_t cap, len;
    CmdExtract *x;
} Request;

static void add_snippet(Request *r, char kind, const char *text, size_t n) {
    size_t i, need = (r->len ? 1 : 0) + 1 + n;
    if (r->x->kind == CMDCHECK_POSIX) {
        r->x->checked += cmdcheck_sh_commands(text, n, r->out, r->cap, &r->len, r->x->changes, sizeof(r->x->changes));
        return;
    }
    for (i = 0; i < n && (is_blank(text[i]) || text[i] == '\n' || text[i] == '\r'); ++i) {}
    if (i == n || r->len + need >= r->cap) return;
    if (r->len) r->out[r->len++] = RS;
    r->out[r->len++] = kind;
    memcpy(r->out + r->len, text, n);
    r->len += n;
    r->out[r->len] = 0;
    ++r->x->checked;
}

static void add_foreign(CmdExtract *x, const char *tag) {
    size_t used = strlen(x->foreign), n = strlen(tag);
    const char *p;
    for (p = x->foreign; (p = strstr(p, tag)) != NULL; p += n)
        if ((p == x->foreign || p[-1] == ' ') && (p[n] == 0 || p[n] == ',')) return;
    if (used + (used ? 2 : 0) + n >= sizeof(x->foreign)) return;
    if (used) strcat(x->foreign, ", ");
    strcat(x->foreign, tag);
}

/* POSIX inline code worth reading as a command: a lowercase name, then more
 * words (`sudo systemctl restart x`), never a lone word (`config.ini`). */
static int posix_inline(const char *s, size_t n) {
    size_t i = 0;
    if (!n || s[0] < 'a' || s[0] > 'z') return 0;
    while (i < n && !is_blank(s[i])) {
        if (!is_alpha(s[i]) && !is_digit(s[i]) && !strchr("._+-", s[i])) return 0;
        ++i;
    }
    while (i < n && is_blank(s[i])) ++i;
    return i < n;
}

/* `inline code` and, for PowerShell, prose on one line s[b..e) outside any block. */
static void scan_line(Request *r, const char *s, size_t b, size_t e) {
    size_t i = b;
    int posix = r->x->kind == CMDCHECK_POSIX;
    while (i < e) {
        size_t len = 0, k;
        if (s[i] == '`') {
            const char *close = (const char *)memchr(s + i + 1, '`', e - i - 1);
            if (!close) {
                ++i;
                continue;
            }
            k = (size_t)(close - s);
            if (posix ? posix_inline(s + i + 1, k - i - 1) : verb_noun(s, i + 1, k, 0) != 0)
                add_snippet(r, 'I', s + i + 1, k - i - 1);
            i = k + 1;
            continue;
        }
        if (posix || !word_start(s, b, i) || !(len = verb_noun(s, i, e, 1))) {
            ++i;
            continue;
        }
        /* Up to the line's end, a backtick, a sentence mark before a blank, or the next command. */
        for (k = i + len; k < e; ++k) {
            if (s[k] == '`' || (strchr(".,;:!?", s[k]) && k + 1 < e && is_blank(s[k + 1]))) break;
            if (word_start(s, b, k) && verb_noun(s, k, e, 1)) break;
        }
        add_snippet(r, 'I', s + i, k - i);
        i = k;
    }
}

static size_t line_end(const char *s, size_t i, size_t n) {
    const char *nl = (const char *)memchr(s + i, '\n', n - i);
    return nl ? (size_t)(nl - s) : n;
}

/* Whether s[i..) starts a ``` fence after blanks; *after is past the backticks. */
static int fence_at(const char *s, size_t i, size_t e, size_t *after) {
    while (i < e && is_blank(s[i])) ++i;
    if (e - i < 3 || memcmp(s + i, "```", 3)) return 0;
    *after = i + 3;
    return 1;
}

static void fence_tag(const char *s, size_t i, size_t e, char tag[TAG_MAX]) {
    size_t n = 0;
    while (i < e && is_blank(s[i])) ++i;
    while (i < e && !is_blank(s[i]) && s[i] != '\r' && n + 1 < TAG_MAX) {
        char c = s[i++];
        tag[n++] = is_upper(c) ? (char)(c - 'A' + 'a') : c;
    }
    tag[n] = 0;
}

static int tag_checked(const char *tag, CmdCheckKind kind) {
    size_t n = strlen(tag);
    return in_list(tag, n, BOTH_TAGS, COUNT(BOTH_TAGS)) ||
           (kind == CMDCHECK_POWERSHELL ? in_list(tag, n, POWERSHELL_TAGS, COUNT(POWERSHELL_TAGS))
                                        : in_list(tag, n, POSIX_TAGS, COUNT(POSIX_TAGS)));
}

static int tag_foreign(const char *tag, CmdCheckKind kind) {
    size_t n = strlen(tag);
    return in_list(tag, n, OTHER_SHELL_TAGS, COUNT(OTHER_SHELL_TAGS)) ||
           (kind == CMDCHECK_POWERSHELL ? in_list(tag, n, POSIX_TAGS, COUNT(POSIX_TAGS))
                                        : in_list(tag, n, POWERSHELL_TAGS, COUNT(POWERSHELL_TAGS)));
}

size_t cmdcheck_extract(const char *reply, size_t n, CmdCheckKind kind, char *out, size_t cap, CmdExtract *x) {
    Request r;
    size_t i = 0;
    memset(x, 0, sizeof(*x));
    x->kind = kind;
    r.out = out, r.cap = cap, r.len = 0, r.x = x;
    if (cap) out[0] = 0;
    while (i < n) {
        size_t e = line_end(reply, i, n), next = e < n ? e + 1 : n, after, body, j;
        char tag[TAG_MAX];
        if (!fence_at(reply, i, e, &after)) {
            scan_line(&r, reply, i, e);
            i = next;
            continue;
        }
        fence_tag(reply, after, e, tag);
        body = j = next;
        while (j < n && !fence_at(reply, j, line_end(reply, j, n), &after)) {
            size_t le = line_end(reply, j, n);
            j = le < n ? le + 1 : n;
        }
        if (tag_checked(tag, kind)) add_snippet(&r, 'B', reply + body, j - body);
        else if (tag_foreign(tag, kind)) add_foreign(x, tag);
        if (j < n) j = line_end(reply, j, n), j = j < n ? j + 1 : n; /* past the closing fence */
        i = j;
    }
    return r.len;
}

/* ---- the notice ---------------------------------------------------------- */

typedef struct Text {
    char *p;
    size_t cap, len;
} Text;

static void put(Text *t, const char *fmt, ...) SM_PRINTF(2, 3);

static void put(Text *t, const char *fmt, ...) {
    va_list ap;
    int n;
    if (t->len + 1 >= t->cap) return;
    va_start(ap, fmt);
    n = vsnprintf(t->p + t->len, t->cap - t->len, fmt, ap);
    va_end(ap);
    if (n < 0) return;
    t->len = (size_t)n < t->cap - t->len ? t->len + (size_t)n : t->cap - 1;
}

/* Splits a findings line into up to `max` tab-separated fields, copied with
 * control bytes as '?'. Returns how many there are. */
static int split_fields(const char *line, size_t n, char fields[][FIELD_MAX], int max) {
    int count = 0;
    size_t i = 0;
    while (count < max) {
        size_t w = 0;
        while (i < n && line[i] != '\t') {
            unsigned char c = (unsigned char)line[i++];
            if (w + 1 < FIELD_MAX) fields[count][w++] = c < 0x20 || c == 0x7f ? '?' : (char)c;
        }
        fields[count++][w] = 0;
        if (i >= n) break;
        ++i;
    }
    return count;
}

static int starts_verb(const char *name) {
    size_t i = 0;
    while (is_alpha(name[i])) ++i;
    return i && name[i] == '-';
}

/* Where an option or subcommand was looked for: "lista" is the index in the binary. */
static const char *source_of(int count, char f[][FIELD_MAX], char *buf, size_t cap) {
    if (count > 3 && !strcmp(f[3], "lista")) snprintf(buf, cap, "a lista embutida (%s)", cmdcheck_index_source());
    else snprintf(buf, cap, "o manual desta máquina");
    return buf;
}

/* One finding: 1 for an item, 0 for a malformed line; an "L" line (programs
 * checked by the index) goes to `notes` instead. */
static int format_item(Text *t, Text *notes, CmdCheckKind kind, const char *line, size_t n) {
    char f[4][FIELD_MAX], source[96];
    int count = split_fields(line, n, f, 4);
    if (count < 2 || !f[1][0]) return 0;
    if (!strcmp(f[0], "N") && kind == CMDCHECK_POSIX) {
        put(t, "  - %s: não está no PATH desta máquina (se o comando é para outra máquina, ignore este aviso).", f[1]);
    } else if (!strcmp(f[0], "N")) {
        put(t, "  - %s: não achei aqui (%s).", f[1],
            starts_verb(f[1]) ? "pode ser de um módulo que esta máquina não tem"
                              : "se o comando é para outra máquina, ignore este aviso");
        if (count > 2 && f[2][0]) put(t, " Parecidos: %s.", f[2]);
    } else if (!strcmp(f[0], "P") && count >= 3 && f[2][0]) {
        put(t, "  - %s não tem o parâmetro -%s.", f[1], f[2]);
        if (count > 3 && f[3][0]) put(t, " Parecidos: %s.", f[3]);
    } else if ((!strcmp(f[0], "O") || !strcmp(f[0], "S")) && count >= 3 && f[2][0]) {
        put(t, "  - %s: %s não cita %s %s.", f[1], source_of(count, f, source, sizeof(source)),
            f[0][0] == 'O' ? "a opção" : "o subcomando", f[2]);
    } else if (!strcmp(f[0], "M")) {
        put(t, "  - %s: sem manual nesta máquina; as opções não foram conferidas.", f[1]);
    } else if (!strcmp(f[0], "L")) {
        put(notes, "%s%s", notes->len ? ", " : "", f[1]);
        return 0;
    } else if (!strcmp(f[0], "W") && count > 2 && !strcmp(f[2], "-")) {
        put(t, "  - %s muda o sistema: confira o alvo antes de rodar.", f[1]);
    } else if (!strcmp(f[0], "W")) {
        put(t, "  - %s muda o sistema%s: confira o alvo e teste antes com -WhatIf.", f[1],
            count > 2 && !strcmp(f[2], "High") ? " e o impacto é alto" : "");
    } else {
        return 0;
    }
    put(t, "\n");
    return 1;
}

/* Whether the line s[start, start+len) already appeared before start. */
static int line_seen(const char *s, size_t start, size_t len) {
    size_t k = 0;
    while (k < start) {
        const char *nl = (const char *)memchr(s + k, '\n', start - k);
        size_t e = nl ? (size_t)(nl - s) : start;
        if (e - k == len && !memcmp(s + k, s + start, len)) return 1;
        k = e + 1;
    }
    return 0;
}

static int format_lines(Text *list, Text *notes, CmdCheckKind kind, const char *s, size_t n) {
    size_t i = 0;
    int count = 0;
    while (i < n) {
        const char *nl = (const char *)memchr(s + i, '\n', n - i);
        size_t e = nl ? (size_t)(nl - s) : n;
        /* Each line of the reply is looked up apart: the same finding can come several times. */
        if (!line_seen(s, i, e - i)) count += format_item(list, notes, kind, s + i, e - i);
        i = e + 1;
    }
    return count;
}

void cmdcheck_format(const char *findings, size_t n, const CmdExtract *x, const char *shell_name, char *out,
                     size_t cap) {
    char items[CMDCHECK_NOTICE_MAX], noted[512];
    Text t, list, notes;
    int count, plural = strchr(x->foreign, ',') != NULL, posix = x->kind == CMDCHECK_POSIX;
    t.p = out, t.cap = cap, t.len = 0;
    list.p = items, list.cap = sizeof(items), list.len = 0;
    notes.p = noted, notes.cap = sizeof(noted), notes.len = 0;
    if (cap) out[0] = 0;
    items[0] = noted[0] = 0;
    count = format_lines(&list, &notes, x->kind, findings, n);
    count += format_lines(&list, &notes, x->kind, x->changes, strlen(x->changes));
    if (count) {
        put(&t, "Conferência no %s desta máquina:\n%s", shell_name, items);
        if (notes.len)
            put(&t, "  - sem manual nesta máquina para %s: conferido pela lista embutida (%s), e a versão daqui pode "
                    "ter outras opções.\n", noted, cmdcheck_index_source());
        if (x->foreign[0])
            put(&t, "  - %s %s não conferido%s: %s para outro shell.\n", plural ? "blocos" : "bloco", x->foreign,
                plural ? "s" : "", plural ? "são" : "é");
        put(&t, "Isso confere só se os nomes existem aqui; não diz se o comando faz o que você pediu.\n");
        return;
    }
    if (x->checked)
        put(&t, "(conferido no %s desta máquina: %s. Isso não diz se o comando faz o que você pediu.)\n", shell_name,
            posix ? "os programas existem e as opções aparecem no manual" : "os nomes e parâmetros existem");
    if (x->checked && notes.len)
        put(&t, "(sem manual nesta máquina para %s: conferido pela lista embutida (%s), e a versão daqui pode ter "
                "outras opções.)\n", noted, cmdcheck_index_source());
    if (x->foreign[0])
        put(&t, "(%s %s não conferido%s: %s para outro shell, e só o %s desta máquina é conferido.)\n",
            plural ? "blocos" : "bloco", x->foreign, plural ? "s" : "", plural ? "são" : "é", shell_name);
}

size_t cmdcheck_base64(const void *data, size_t n, char *out, size_t cap) {
    static const char ALPHABET[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const unsigned char *s = (const unsigned char *)data;
    size_t i, w = 0, need = (n + 2) / 3 * 4;
    if (need >= cap) {
        if (cap) out[0] = 0;
        return 0;
    }
    for (i = 0; i + 2 < n; i += 3) {
        unsigned long v = (unsigned long)s[i] << 16 | (unsigned long)s[i + 1] << 8 | s[i + 2];
        out[w++] = ALPHABET[v >> 18 & 63];
        out[w++] = ALPHABET[v >> 12 & 63];
        out[w++] = ALPHABET[v >> 6 & 63];
        out[w++] = ALPHABET[v & 63];
    }
    if (i < n) {
        unsigned long v = (unsigned long)s[i] << 16 | (i + 1 < n ? (unsigned long)s[i + 1] << 8 : 0);
        out[w++] = ALPHABET[v >> 18 & 63];
        out[w++] = ALPHABET[v >> 12 & 63];
        out[w++] = i + 1 < n ? ALPHABET[v >> 6 & 63] : '=';
        out[w++] = '=';
    }
    out[w] = 0;
    return w;
}

/* ---- the PowerShell that answers ----------------------------------------- */

/* Reads one base64 request per line and answers with findings lines and a
 * line "E". It only parses the snippets (Parser::ParseInput) and asks
 * Get-Command about each name: nothing in a reply ever runs. Written without
 * double quotes, backslashes or backticks, so it needs no escaping here; one
 * string per line keeps every literal under C99's 4095 characters. */
static const char *const SCRIPT[] = {
    "$ErrorActionPreference = 'SilentlyContinue'",
    "$ProgressPreference = 'SilentlyContinue'",
    "$T = [char]9",
    "$Out = [Console]::Out",
    "$script:Common = [System.Management.Automation.Cmdlet]::CommonParameters + [System.Management.Automation.Cmdlet]::OptionalCommonParameters",
    "$script:SessionOnly = 'ForEach-Object', 'Convert-Path', 'Clear-History'",
    "$script:SessionNouns = 'Variable', 'Alias', 'PSBreakpoint', 'PSDrive'",
    "function Dist([string]$a, [string]$b) {",
    "    $a = $a.ToLowerInvariant(); $b = $b.ToLowerInvariant()",
    "    if ([Math]::Abs($a.Length - $b.Length) -gt 2) { return 3 }",
    "    $prev = [int[]]::new($b.Length + 1)",
    "    $cur = [int[]]::new($b.Length + 1)",
    "    for ($j = 0; $j -le $b.Length; $j++) { $prev[$j] = $j }",
    "    for ($i = 1; $i -le $a.Length; $i++) {",
    "        $cur[0] = $i",
    "        for ($j = 1; $j -le $b.Length; $j++) {",
    "            $best = $prev[$j - 1] + [int]($a[$i - 1] -ne $b[$j - 1])",
    "            if ($cur[$j - 1] + 1 -lt $best) { $best = $cur[$j - 1] + 1 }",
    "            if ($prev[$j] + 1 -lt $best) { $best = $prev[$j] + 1 }",
    "            $cur[$j] = $best",
    "        }",
    "        $swap = $prev; $prev = $cur; $cur = $swap",
    "    }",
    "    $prev[$b.Length]",
    "}",
    "function Like-Command([string]$n) {",
    "    $parts = $n.Split('-', 2)",
    "    $noun = $parts[1]",
    "    $near = @()",
    "    foreach ($m in $script:ByVerb[$parts[0]]) {",
    "        if ((Dist $m.Split('-', 2)[1] $noun) -le 2) { $near += $m }",
    "    }",
    "    foreach ($k in @($noun, $noun.Substring(0, [Math]::Min(5, $noun.Length)))) {",
    "        if ($near.Count -ge 3) { break }",
    "        $pat = '*-*' + $k + '*'",
    "        $near += @($script:AllNames.Where({ $_ -like $pat }) | Sort-Object Length, { $_ })",
    "    }",
    "    @($near | Select-Object -Unique -First 3)",
    "}",
    "function Like-Parameter($params, [string]$pn) {",
    "    @($params.Keys | Where-Object { $_ -notin $script:Common } | ForEach-Object {",
    "        $d = Dist $_ $pn",
    "        if ($d -le 2 -or $_ -like ('*' + $pn + '*') -or $pn -like ('*' + $_ + '*')) { [pscustomobject]@{ N = $_; D = $d } }",
    "    } | Sort-Object D, N | Select-Object -First 3 | ForEach-Object { '-' + $_.N })",
    "}",
    "function Check([string]$text) {",
    "    $seen = @{}",
    "    foreach ($s in ($text -split [char]30)) {",
    "        if ($s.Length -lt 2) { continue }",
    "        $block = $s[0] -eq 'B'",
    "        $tokens = $null; $errors = $null",
    "        $ast = [System.Management.Automation.Language.Parser]::ParseInput($s.Substring(1), [ref]$tokens, [ref]$errors)",
    "        foreach ($c in $ast.FindAll({ $args[0] -is [System.Management.Automation.Language.CommandAst] }, $true)) {",
    "            $n = $c.GetCommandName()",
    "            if (-not $n -or $n.IndexOfAny([char[]]('/*?[]' + [char]92)) -ge 0 -or $n.StartsWith('.')) { continue }",
    "            $verbNoun = $n -match '^[A-Za-z]+-[A-Za-z][A-Za-z0-9]*$'",
    "            if (-not $block -and -not $verbNoun) { continue }",
    "            $cmd = if ($script:Known.Count -eq 0 -or $script:Known.Contains($n)) { Get-Command -Name $n | Select-Object -First 1 }",
    "            if (-not $cmd) {",
    "                if (-not $seen.ContainsKey('N' + $n)) {",
    "                    $seen['N' + $n] = 1",
    "                    $like = if ($verbNoun) { Like-Command $n } else { @() }",
    "                    $Out.WriteLine('N' + $T + $n + $T + ($like -join ', '))",
    "                }",
    "                continue",
    "            }",
    "            if ($cmd.CommandType -eq 'Alias') { $cmd = if ($cmd.Definition) { Get-Command -Name $cmd.Definition | Select-Object -First 1 } }",
    "            if (-not $cmd -or $cmd.CommandType -notin 'Cmdlet', 'Function') { continue }",
    "            $params = $cmd.Parameters",
    "            if (-not $params) { continue }",
    "            $paramNames = @($params.Keys) + @($params.Values | ForEach-Object { $_.Aliases })",
    "            foreach ($p in $c.CommandElements) {",
    "                if ($p -isnot [System.Management.Automation.Language.CommandParameterAst]) { continue }",
    "                $pn = $p.ParameterName",
    "                if (-not $pn -or $seen.ContainsKey('P' + $cmd.Name + '-' + $pn)) { continue }",
    "                $pat = [System.Management.Automation.WildcardPattern]::Escape($pn) + '*'",
    "                if ($paramNames | Where-Object { $_ -like $pat }) { continue }",
    "                $seen['P' + $cmd.Name + '-' + $pn] = 1",
    "                $Out.WriteLine('P' + $T + $cmd.Name + $T + $pn + $T + ((Like-Parameter $params $pn) -join ', '))",
    "            }",
    "            if ($seen.ContainsKey('W' + $cmd.Name) -or $cmd.Name -in $script:SessionOnly -or $cmd.Noun -in $script:SessionNouns) { continue }",
    "            $impact = $null",
    "            if ($cmd.CommandType -eq 'Cmdlet' -and $cmd.ImplementingType) {",
    "                $attr = $cmd.ImplementingType.GetCustomAttributes([System.Management.Automation.CmdletAttribute], $true) | Select-Object -First 1",
    "                if ($attr -and $attr.SupportsShouldProcess) { $impact = [string]$attr.ConfirmImpact }",
    "            } elseif ($params.ContainsKey('WhatIf')) {",
    "                $impact = ''",
    "                $binding = $cmd.ScriptBlock.Attributes | Where-Object { $_ -is [System.Management.Automation.CmdletBindingAttribute] } | Select-Object -First 1",
    "                if ($binding) { $impact = [string]$binding.ConfirmImpact }",
    "            }",
    "            if ($null -ne $impact) {",
    "                $seen['W' + $cmd.Name] = 1",
    "                $Out.WriteLine('W' + $T + $cmd.Name + $T + $impact)",
    "            }",
    "        }",
    "    }",
    "}",
    "$script:Known = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)",
    "$script:AllNames = [Collections.Generic.List[string]]::new()",
    "$script:ByVerb = @{}",
    "foreach ($c in (Get-Command -CommandType Cmdlet, Function, Alias, Filter)) {",
    "    [void]$script:Known.Add($c.Name)",
    "    if ($c.CommandType -in 'Cmdlet', 'Function' -and $c.Name -match '^[A-Za-z]+-.') {",
    "        $script:AllNames.Add($c.Name)",
    "        $verb = $c.Name.Split('-', 2)[0]",
    "        if (-not $script:ByVerb.ContainsKey($verb)) { $script:ByVerb[$verb] = [Collections.Generic.List[string]]::new() }",
    "        $script:ByVerb[$verb].Add($c.Name)",
    "    }",
    "}",
    "foreach ($a in (Get-Command -CommandType Application)) {",
    "    [void]$script:Known.Add($a.Name)",
    "    [void]$script:Known.Add([IO.Path]::GetFileNameWithoutExtension($a.Name))",
    "}",
    "foreach ($w in 'Get-ChildItem', 'Get-Process', 'Get-Service', 'Get-Content') { $null = (Get-Command -Name $w).Parameters }",
    "while ($true) {",
    "    $line = [Console]::In.ReadLine()",
    "    if ($null -eq $line) { break }",
    "    try { Check ([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($line.Trim()))) } catch { }",
    "    $Out.WriteLine('E')",
    "    $Out.Flush()",
    "}"
};

#define SH_ENV "SMOLMOE_CHECK_SCRIPT" /* Windows: the sh script travels in this variable, unquoted */
#define SH_SCRIPT_MAX 8192
#define POSIX_ANSWER_MS 15000u /* man pages of a whole reply, on a slow disk */
#define INDEX_LINE_MAX 768

struct CmdChecker {
    Shell sh;
    CmdCheckKind kind;
    Child *child; /* PowerShell, kept running; a POSIX sh lives for one reply */
    int answered; /* the running PowerShell answered once: it is warm */
};

/* sh from PATH; else the chosen shell, when it runs sh scripts (not zsh or fish). */
static int posix_shell(const Shell *chosen, Shell *out, char *err, size_t cap) {
    static const char *const RUNS_SH[] = {"sh", "bash", "dash", "ksh", "ash", "mksh"};
    if (shell_choose("sh", NULL, out, err, cap)) return 1;
    if (chosen && in_list(chosen->name, strlen(chosen->name), RUNS_SH, COUNT(RUNS_SH))) {
        *out = *chosen;
        return 1;
    }
    snprintf(err, cap, "nenhum sh no PATH para conferir os comandos");
    return 0;
}

int cmdcheck_shell(const char *choice, const Shell *chat, Shell *out, char *err, size_t cap) {
    Shell picked;
    const Shell *base = NULL;
    if (choice) {
        if (!shell_choose(choice, NULL, &picked, err, cap)) return 0;
        if (picked.kind == SHELL_CMD) {
            snprintf(err, cap, "a conferência dos comandos precisa de um PowerShell ou de um shell POSIX (sh, bash), "
                               "não do cmd");
            return 0;
        }
        base = &picked;
    } else if (chat && chat->kind != SHELL_CMD) {
        base = chat;
    }
    if (base && base->kind == SHELL_POWERSHELL) {
        *out = *base;
        return 1;
    }
    if (base) return posix_shell(base, out, err, cap);
#if defined(_WIN32)
    if (shell_choose("pwsh", NULL, out, err, cap) || shell_choose("powershell", NULL, out, err, cap)) return 1;
    snprintf(err, cap, "nenhum PowerShell (pwsh ou powershell) no PATH");
    return 0;
#else
    return posix_shell(NULL, out, err, cap);
#endif
}

/* PowerShell's -EncodedCommand: the script as UTF-16LE, in base64. */
static char *encoded_script(void) {
    size_t i, n = 0, at = 0, cap;
    unsigned char *wide;
    char *b64;
    for (i = 0; i < COUNT(SCRIPT); ++i) n += strlen(SCRIPT[i]) + 1;
    wide = (unsigned char *)malloc(n * 2);
    cap = n * 2 / 3 * 4 + 8;
    b64 = (char *)malloc(cap);
    if (!wide || !b64) {
        free(wide);
        free(b64);
        return NULL;
    }
    for (i = 0; i < COUNT(SCRIPT); ++i) {
        const char *p;
        for (p = SCRIPT[i]; *p; ++p) wide[at++] = (unsigned char)*p, wide[at++] = 0;
        wide[at++] = '\n', wide[at++] = 0;
    }
    if (!cmdcheck_base64(wide, at, b64, cap)) {
        free(b64);
        b64 = NULL;
    }
    free(wide);
    return b64;
}

static Child *start_powershell(const Shell *ps, char *err, size_t cap) {
    char *script = encoded_script(), *line = NULL;
    Child *child = NULL;
    if (!script) {
        snprintf(err, cap, "sem memória");
        return NULL;
    }
#if defined(_WIN32)
    line = (char *)malloc(strlen(ps->program) + strlen(script) + 96);
    if (line) {
        sprintf(line, "\"%s\" -NoLogo -NoProfile -NonInteractive -EncodedCommand %s", ps->program, script);
        child = child_start(NULL, line, NULL, NULL, err, cap);
    } else {
        snprintf(err, cap, "sem memória");
    }
#else
    {
        const char *argv[] = {ps->program, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", script, NULL};
        child = child_start(argv, NULL, NULL, NULL, err, cap);
    }
#endif
    free(line);
    free(script);
    return child;
}

/* The sh script goes in -c (POSIX) or in a variable (Windows), never on
 * stdin: a shell that reads its script in blocks (dash) would take request
 * lines for commands. */
static Child *start_sh(const Shell *sh, char *err, size_t cap) {
    char script[SH_SCRIPT_MAX];
    if (!cmdcheck_sh_script(script, sizeof(script))) {
        snprintf(err, cap, "o roteiro da conferência não coube");
        return NULL;
    }
#if defined(_WIN32)
    {
        char line[SHELL_PATH_MAX + 64];
        snprintf(line, sizeof(line), "\"%s\" -c \"eval \\\"$" SH_ENV "\\\"\"", sh->program);
        return child_start(NULL, line, SH_ENV, script, err, cap);
    }
#else
    {
        const char *argv[] = {sh->program, "-c", script, NULL};
        return child_start(argv, NULL, NULL, NULL, err, cap);
    }
#endif
}

CmdChecker *cmdcheck_start(const Shell *sh, int warm, char *err, size_t cap) {
    CmdChecker *c = (CmdChecker *)calloc(1, sizeof(*c));
    if (!c) {
        snprintf(err, cap, "sem memória");
        return NULL;
    }
    c->sh = *sh;
    c->kind = sh->kind == SHELL_POWERSHELL ? CMDCHECK_POWERSHELL : CMDCHECK_POSIX;
    if (warm && c->kind == CMDCHECK_POWERSHELL && !(c->child = start_powershell(sh, err, cap))) {
        free(c);
        return NULL;
    }
    return c;
}

/* Sends the request and reads the findings; 0 when PowerShell left or did not
 * answer in time (it is then stopped, and started again next time). */
static int ask_powershell(CmdChecker *c, const char *request, size_t n, char *findings, size_t cap) {
    char err[256];
    size_t b64_cap = (n + 2) / 3 * 4 + 2, len;
    char *b64 = (char *)malloc(b64_cap);
    int ok;
    if (!b64) return 0;
    if (!c->child) {
        c->answered = 0;
        c->child = start_powershell(&c->sh, err, sizeof(err));
    }
    len = cmdcheck_base64(request, n, b64, b64_cap);
    b64[len++] = '\n';
    ok = c->child && child_write(c->child, b64, len) &&
         child_read_until(c->child, "E", c->answered ? ANSWER_MS : FIRST_ANSWER_MS, findings, cap);
    free(b64);
    if (!ok) {
        child_stop(c->child);
        c->child = NULL;
        return 0;
    }
    c->answered = 1;
    return 1;
}

static int ask_sh(CmdChecker *c, const char *request, size_t n, char *findings, size_t cap) {
    char err[256];
    Child *child = start_sh(&c->sh, err, sizeof(err));
    int ok = child && child_write(child, request, n) && child_write(child, "E\n", 2) &&
             child_read_until(child, "E", POSIX_ANSWER_MS, findings, cap);
    child_stop(child);
    return ok;
}

/* Copies field `k` (tab-separated) of line[0..n) into out; 0 when absent. */
static int field(const char *line, size_t n, int k, char *out, size_t cap) {
    size_t i = 0, w = 0;
    for (; k > 0 && i < n; ++i)
        if (line[i] == '\t') --k;
    if (k > 0 || i > n) return 0;
    while (i < n && line[i] != '\t' && w + 1 < cap) out[w++] = line[i++];
    out[w] = 0;
    return w > 0 || (i < n && line[i] == '\t');
}

/* The index's verdict on one request line of a program without a manual. */
static void index_line(Text *t, const char *line, size_t n, const char *name) {
    char sub[FIELD_MAX], opt[FIELD_MAX];
    int k;
    if (field(line, n, 1, sub, sizeof(sub)) && strcmp(sub, ".") && !cmdcheck_index_sub(name, sub))
        put(t, "S\t%s\t%s\tlista\n", name, sub);
    for (k = 2; field(line, n, k, opt, sizeof(opt)); ++k)
        if (!cmdcheck_index_option(name, opt)) put(t, "O\t%s\t%s\tlista\n", name, opt);
}

/* Replaces each "M" finding (no manual here) of a program the index knows by
 * the index's verdict and an "L" line; the others stay. */
static void apply_index(const char *request, size_t request_len, char *findings, size_t cap) {
    char *out = (char *)malloc(cap), name[FIELD_MAX];
    Text t;
    size_t i = 0, flen = strlen(findings);
    if (!out) return;
    t.p = out, t.cap = cap, t.len = 0;
    out[0] = 0;
    while (i < flen) {
        const char *nl = (const char *)memchr(findings + i, '\n', flen - i);
        size_t e = nl ? (size_t)(nl - findings) : flen, j = 0;
        if (!(findings[i] == 'M' && field(findings + i, e - i, 1, name, sizeof(name)) && cmdcheck_index_has(name))) {
            put(&t, "%.*s\n", (int)(e - i), findings + i);
            i = e + 1;
            continue;
        }
        while (j < request_len) {
            const char *rn = (const char *)memchr(request + j, '\n', request_len - j);
            size_t re = rn ? (size_t)(rn - request) : request_len;
            char first[FIELD_MAX];
            if (field(request + j, re - j, 0, first, sizeof(first)) && !strcmp(first, name))
                index_line(&t, request + j, re - j, name);
            j = re + 1;
        }
        put(&t, "L\t%s\n", name);
        i = e + 1;
    }
    memcpy(findings, out, t.len + 1);
    free(out);
}

int cmdcheck_reply(CmdChecker *c, const char *reply, size_t n, char *out, size_t cap) {
    char *request = (char *)malloc(REQUEST_MAX), *findings = (char *)malloc(FINDINGS_MAX);
    CmdExtract x;
    size_t len;
    int ok = 1;
    if (cap) out[0] = 0;
    if (!request || !findings) {
        free(request);
        free(findings);
        return 0;
    }
    len = cmdcheck_extract(reply, n, c->kind, request, REQUEST_MAX, &x);
    findings[0] = 0;
    if (x.checked) {
        ok = c->kind == CMDCHECK_POWERSHELL ? ask_powershell(c, request, len, findings, FINDINGS_MAX)
                                            : ask_sh(c, request, len, findings, FINDINGS_MAX);
        if (ok && c->kind == CMDCHECK_POSIX) apply_index(request, len, findings, FINDINGS_MAX);
    }
    if (!ok)
        snprintf(out, cap, "(não consegui conferir os comandos: o %s não respondeu; tento de novo na próxima "
                           "resposta)\n", c->sh.name);
    else
        cmdcheck_format(findings, strlen(findings), &x, c->sh.name, out, cap);
    free(request);
    free(findings);
    return cap && out[0];
}

void cmdcheck_stop(CmdChecker *c) {
    if (!c) return;
    child_stop(c->child);
    free(c);
}
