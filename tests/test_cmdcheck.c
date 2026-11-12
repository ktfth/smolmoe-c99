/* Unit tests for the command check (cmdcheck.h): finding the commands in a
 * reply and turning the checker's findings into notices. Portable: nothing
 * here starts PowerShell (tests/cmdcheck.test.ts does). */
#include <stdio.h>
#include <string.h>

#include "cmdcheck.h"

static int failures;

#define CHECK(cond, ...)                                       \
    do {                                                       \
        if (!(cond)) {                                         \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                      \
            fputc('\n', stderr);                               \
            ++failures;                                        \
        }                                                      \
    } while (0)

#define RS "\036"

static void expect_extract(const char *reply, const char *want, const char *want_foreign) {
    char got[2048];
    CmdExtract x;
    size_t n = cmdcheck_extract(reply, strlen(reply), CMDCHECK_POWERSHELL, got, sizeof(got), &x);
    CHECK(n == strlen(want) && !memcmp(got, want, n), "extract(%s)\n  got  [%.*s]\n  want [%s]", reply, (int)n, got,
          want);
    CHECK(!strcmp(x.foreign, want_foreign), "foreign of (%s): got [%s] want [%s]", reply, x.foreign, want_foreign);
}

static void test_fences(void) {
    /* An untagged or PowerShell block is checked whole. */
    expect_extract("Use:\n```powershell\nGet-ChildItem -Recursive\nls -la\n```\nPronto.",
                   "BGet-ChildItem -Recursive\nls -la\n", "");
    expect_extract("```\nipconfig /all\n```", "Bipconfig /all\n", "");
    expect_extract("```PS1\nGet-Item x\n```", "BGet-Item x\n", "");
    expect_extract("```pwsh\nGet-Item x\n```\n```shell\nGet-Date\n```", "BGet-Item x\n" RS "BGet-Date\n", "");
    /* A block for another shell is named, never checked here. */
    expect_extract("```bash\nsudo apt update\n```", "", "bash");
    expect_extract("```sh\nls\n```\n```cmd\ndir /s\n```", "", "sh, cmd");
    expect_extract("```bash\nls\n```\n```bash\nls\n```", "", "bash");
    /* Other languages are neither checked nor named. */
    expect_extract("```json\n{\"a\": 1}\n```\n```python\nprint(1)\n```", "", "");
    /* An unclosed block runs to the end of the reply. */
    expect_extract("```powershell\nGet-Service -Nme x", "BGet-Service -Nme x", "");
    /* Indented fences (inside a list) count. */
    expect_extract("1. Rode:\n   ```powershell\n   Get-Item x\n   ```\n", "B   Get-Item x\n", "");
}

static void test_inline_and_prose(void) {
    /* Inline code: only when it starts with Verb-Noun. */
    expect_extract("Rode `Get-Process -Nme x` e veja.", "IGet-Process -Nme x", "");
    expect_extract("Abra `config.ini` e `C:\\Temp`.", "", "");
    expect_extract("Use `Check-Disk C:` agora.", "ICheck-Disk C:", "");
    /* Prose: an approved verb, a dash and a capitalized noun, up to the end of
     * the sentence or the next command. */
    expect_extract("Use Get-NetAdapterInfo -Name Wi-Fi para ver.", "IGet-NetAdapterInfo -Name Wi-Fi para ver.", "");
    expect_extract("Rode Get-Process, depois Stop-Process -Id 4. Fim.",
                   "IGet-Process" RS "IStop-Process -Id 4", "");
    expect_extract("Rode Get-Process e depois Stop-Process -Id 4\nok", "IGet-Process e depois " RS "IStop-Process -Id 4",
                   "");
    /* Words with dashes that are not commands. */
    expect_extract("O Wi-Fi e o e-mail; Plug-and-Play; Set-up; Check-Disk sem crase.", "", "");
    expect_extract("Mais Get-", "", "");
    /* Prose inside a fence is not read twice. */
    expect_extract("```\nGet-Item x\n```", "BGet-Item x\n", "");
}

static void test_limits(void) {
    char small[16];
    CmdExtract x;
    size_t n = cmdcheck_extract("```\nGet-ChildItem -Recurse C:\\Windows\n```", 41, CMDCHECK_POWERSHELL, small,
                                sizeof(small), &x);
    CHECK(n == 0, "a snippet that does not fit is dropped whole, got %zu", n);
    n = cmdcheck_extract("", 0, CMDCHECK_POWERSHELL, small, sizeof(small), &x);
    CHECK(n == 0 && !x.foreign[0], "empty reply");
}

/* ---- POSIX: splitting shell snippets into commands ---------------------- */

static void expect_sh(const char *text, const char *want, const char *want_changes) {
    char got[2048], changes[512];
    size_t len = 0;
    changes[0] = 0;
    got[0] = 0;
    cmdcheck_sh_commands(text, strlen(text), got, sizeof(got), &len, changes, sizeof(changes));
    CHECK(len == strlen(want) && !strcmp(got, want), "sh(%s)\n  got  [%s]\n  want [%s]", text, got, want);
    CHECK(!strcmp(changes, want_changes), "changes of (%s)\n  got  [%s]\n  want [%s]", text, changes, want_changes);
}

static void test_sh_commands(void) {
    /* Pipes, && and options, with combined short flags kept whole. */
    expect_sh("ls -la /tmp | grep -v foo && rm -rf /tmp/x\n", "ls\t.\t-la\ngrep\t.\t-v\nrm\t.\t-rf\n",
              "W\trm\t-\n");
    /* sudo is looked up on its own; its options and their values are not the command's. */
    expect_sh("sudo -u www-data apt-get instal -y nginx", "sudo\t.\napt-get\tinstal\t-y\n", "");
    expect_sh("sudo apt-get install -y nginx", "sudo\t.\napt-get\tinstall\t-y\n", "W\tapt-get\t-\n");
    /* A prompt, a subcommand typo and the real one. */
    expect_sh("$ systemctl restat nginx\n$ systemctl restart nginx\n", "systemctl\trestat\nsystemctl\trestart\n",
              "W\tsystemctl\t-\n");
    expect_sh("systemctl status nginx --no-pager", "systemctl\tstatus\t--no-pager\n", "");
    /* NAME=value, redirections, here-documents and comments. */
    expect_sh("LANG=C sort -u nomes.txt > saida.txt 2>&1", "sort\t.\t-u\n", "");
    expect_sh("dmesg -T 2>/dev/null | tail -n50\n# comentario -x\n", "dmesg\t.\t-T\ntail\t.\t-n50\n", "");
    expect_sh("cat <<EOF > /etc/x.conf\nfoo -bar\nEOF\nuname -r\n", "cat\t.\nuname\t.\t-r\n", "");
    /* Values after = go, and only safe characters stay in an option. */
    expect_sh("ls --color=auto --sort='time' -1", "ls\t.\t--color\t--sort\t-1\n", "");
    expect_sh("grep -E 'a|b' x; echo \"-n\"", "grep\t.\t-E\necho\t.\n", "");
    /* Keywords lead to the real command; paths and substitutions are skipped. */
    expect_sh("if grep -q root /etc/passwd; then echo ok; fi", "grep\t.\t-q\necho\t.\n", "");
    expect_sh("./configure --prefix=/usr && make -j4", "make\t.\t-j4\n", "");
    expect_sh("kill -9 $(pidof nginx)", "kill\t.\t-9\n", "W\tkill\t-\n");
    expect_sh("$(which ls) -la; `echo ls` -l", "", "");
    /* Continued lines are one command. */
    expect_sh("tar -czf backup.tgz \\\n  --exclude=cache /srv", "tar\t.\t-czf\t--exclude\n", "");
    /* find: single-dash long options; -delete changes the system. */
    expect_sh("find /var/log -name '*.gz' -mtime +30 -delete", "find\t.\t-name\t-mtime\t-delete\n",
              "W\tfind\t-\n");
    expect_sh("find / -name core -type f", "find\t.\t-name\t-type\n", "");
    /* Package managers: changes, and queries that are not. */
    expect_sh("pacman -Syu", "pacman\t.\t-Syu\n", "W\tpacman\t-\n");
    expect_sh("pacman -Ss vim", "pacman\t.\t-Ss\n", "");
    expect_sh("apt list --installed", "apt\tlist\t--installed\n", "");
    expect_sh("sed -i 's/a/b/' f; sed -n 1p f", "sed\t.\t-i\nsed\t.\t-n\n", "W\tsed\t-\n");
    expect_sh("service nginx restart", "service\t.\n", "W\tservice\t-\n");
    expect_sh("docker ps -a; docker rm -f web", "docker\tps\t-a\ndocker\trm\t-f\n", "W\tdocker\t-\n");
    /* bash builtins that sh lacks are not programs to look up. */
    expect_sh("history | tail -n 20; source ~/.bashrc", "tail\t.\t-n\n", "");
    /* Repeated commands once; each change once. */
    expect_sh("rm a\nrm a\nrm -r b", "rm\t.\nrm\t.\t-r\n", "W\trm\t-\n");
}

static void expect_posix(const char *reply, const char *want, const char *want_foreign) {
    char got[2048];
    CmdExtract x;
    size_t n = cmdcheck_extract(reply, strlen(reply), CMDCHECK_POSIX, got, sizeof(got), &x);
    CHECK(n == strlen(want) && !memcmp(got, want, n), "posix(%s)\n  got  [%.*s]\n  want [%s]", reply, (int)n, got,
          want);
    CHECK(!strcmp(x.foreign, want_foreign), "foreign of (%s): got [%s] want [%s]", reply, x.foreign, want_foreign);
    CHECK(x.kind == CMDCHECK_POSIX, "kind");
}

static void test_posix_extract(void) {
    expect_posix("No Debian:\n```bash\nsudo apt update\n```\n", "sudo\t.\napt\tupdate\n", "");
    expect_posix("```\nls -la\n```", "ls\t.\t-la\n", "");
    expect_posix("```console\n$ df -h\n```", "df\t.\t-h\n", "");
    /* PowerShell and cmd blocks are for another shell here. */
    expect_posix("```powershell\nGet-ChildItem -Recurse\n```\n```cmd\ndir /s\n```", "", "powershell, cmd");
    /* Inline code: a command name and more words; prose is not read. */
    expect_posix("Rode `sudo systemctl restart nginx` e veja `/etc/fstab` e `config.ini`.",
                 "sudo\t.\nsystemctl\trestart\n", "");
    expect_posix("Use Get-Process ou ls -la no texto.", "", "");
}

static void expect_format(const char *findings, int checked, const char *foreign, const char *want) {
    char got[2048];
    CmdExtract x;
    memset(&x, 0, sizeof(x));
    x.checked = checked;
    snprintf(x.foreign, sizeof(x.foreign), "%s", foreign);
    cmdcheck_format(findings, strlen(findings), &x, "PowerShell 7", got, sizeof(got));
    CHECK(!strcmp(got, want), "format(%s)\n  got  [%s]\n  want [%s]", findings, got, want);
}

static void test_format(void) {
    expect_format("", 1, "",
                  "(conferido no PowerShell 7 desta máquina: os nomes e parâmetros existem. Isso não diz se o comando "
                  "faz o que você pediu.)\n");
    expect_format("", 0, "", "");
    expect_format("", 0, "bash",
                  "(bloco bash não conferido: é para outro shell, e só o PowerShell 7 desta máquina é conferido.)\n");
    expect_format("N\tGet-NetAdapterInfo\tGet-NetAdapter, Set-NetAdapter\n"
                  "P\tGet-ChildItem\tRecursive\t-Recurse\n"
                  "P\tGet-Process\tNme\t\n"
                  "W\tRemove-Item\tMedium\n"
                  "W\tFormat-Volume\tHigh\n"
                  "N\tapt\t\n",
                  1, "sh",
                  "Conferência no PowerShell 7 desta máquina:\n"
                  "  - Get-NetAdapterInfo: não achei aqui (pode ser de um módulo que esta máquina não tem). "
                  "Parecidos: Get-NetAdapter, Set-NetAdapter.\n"
                  "  - Get-ChildItem não tem o parâmetro -Recursive. Parecidos: -Recurse.\n"
                  "  - Get-Process não tem o parâmetro -Nme.\n"
                  "  - Remove-Item muda o sistema: confira o alvo e teste antes com -WhatIf.\n"
                  "  - Format-Volume muda o sistema e o impacto é alto: confira o alvo e teste antes com -WhatIf.\n"
                  "  - apt: não achei aqui (se o comando é para outra máquina, ignore este aviso).\n"
                  "  - bloco sh não conferido: é para outro shell.\n"
                  "Isso confere só se os nomes existem aqui; não diz se o comando faz o que você pediu.\n");
    /* Control bytes from the model never reach the terminal. */
    expect_format("N\tGet-\033[31mX\t\n", 1, "",
                  "Conferência no PowerShell 7 desta máquina:\n"
                  "  - Get-?[31mX: não achei aqui (pode ser de um módulo que esta máquina não tem).\n"
                  "Isso confere só se os nomes existem aqui; não diz se o comando faz o que você pediu.\n");
    /* Lines the checker should never write are skipped. */
    expect_format("Z\twhat\nP\tonly-two\n", 1, "",
                  "(conferido no PowerShell 7 desta máquina: os nomes e parâmetros existem. Isso não diz se o comando "
                  "faz o que você pediu.)\n");
}

static void expect_format_posix(const char *findings, const char *changes, int checked, const char *foreign,
                                const char *want) {
    char got[2048];
    CmdExtract x;
    memset(&x, 0, sizeof(x));
    x.kind = CMDCHECK_POSIX;
    x.checked = checked;
    snprintf(x.foreign, sizeof(x.foreign), "%s", foreign);
    snprintf(x.changes, sizeof(x.changes), "%s", changes);
    cmdcheck_format(findings, strlen(findings), &x, "sh", got, sizeof(got));
    CHECK(!strcmp(got, want), "format posix(%s)\n  got  [%s]\n  want [%s]", findings, got, want);
}

static void test_format_posix(void) {
    expect_format_posix("", "", 1, "",
                        "(conferido no sh desta máquina: os programas existem e as opções aparecem no manual. Isso "
                        "não diz se o comando faz o que você pediu.)\n");
    expect_format_posix("", "", 0, "powershell",
                        "(bloco powershell não conferido: é para outro shell, e só o sh desta máquina é "
                        "conferido.)\n");
    expect_format_posix("N\tnetstatx\t\nO\tls\t--recursive-all\nS\tsystemctl\trestat\nM\tbusybox\n",
                        "W\trm\t-\n", 1, "",
                        "Conferência no sh desta máquina:\n"
                        "  - netstatx: não está no PATH desta máquina (se o comando é para outra máquina, ignore "
                        "este aviso).\n"
                        "  - ls: o manual desta máquina não cita a opção --recursive-all.\n"
                        "  - systemctl: o manual desta máquina não cita o subcomando restat.\n"
                        "  - busybox: sem manual nesta máquina; as opções não foram conferidas.\n"
                        "  - rm muda o sistema: confira o alvo antes de rodar.\n"
                        "Isso confere só se os nomes existem aqui; não diz se o comando faz o que você pediu.\n");
    /* A change alone is a finding: no "everything exists" line hides it. */
    expect_format_posix("", "W\tsystemctl\t-\n", 1, "",
                        "Conferência no sh desta máquina:\n"
                        "  - systemctl muda o sistema: confira o alvo antes de rodar.\n"
                        "Isso confere só se os nomes existem aqui; não diz se o comando faz o que você pediu.\n");
    /* The same finding from several lines of the reply is said once. */
    expect_format_posix("N\tufw\t\nO\tapt\t--x\nN\tufw\t\nO\tapt\t--x\n", "", 1, "",
                        "Conferência no sh desta máquina:\n"
                        "  - ufw: não está no PATH desta máquina (se o comando é para outra máquina, ignore "
                        "este aviso).\n"
                        "  - apt: o manual desta máquina não cita a opção --x.\n"
                        "Isso confere só se os nomes existem aqui; não diz se o comando faz o que você pediu.\n");
    /* A dashed program name is not read as a PowerShell Verb-Noun. */
    expect_format_posix("N\tapt-get\t\n", "", 1, "",
                        "Conferência no sh desta máquina:\n"
                        "  - apt-get: não está no PATH desta máquina (se o comando é para outra máquina, ignore "
                        "este aviso).\n"
                        "Isso confere só se os nomes existem aqui; não diz se o comando faz o que você pediu.\n");
}

static void test_sh_script(void) {
    char script[8192];
    size_t n = cmdcheck_sh_script(script, sizeof(script));
    CHECK(n > 100 && strstr(script, "man -w") && strstr(script, "command -v"), "the sh script is there (%zu)", n);
    CHECK(!strstr(script, "--help"), "the sh script never runs a program from the reply");
    CHECK(strstr(script, "Fl ") != NULL, "mdoc pages (.Fl t) are read too");
    CHECK(strstr(script, "$name-$sub") != NULL, "a subcommand's own page (docker-ps) is read");
    CHECK(strstr(script, "apt-get") != NULL, "apt's options are in apt-get's page");
    CHECK(cmdcheck_sh_script(script, 10) == 0, "a script that does not fit is refused");
}

static void test_base64(void) {
    char out[64];
    CHECK(cmdcheck_base64("", 0, out, sizeof(out)) == 0 && !out[0], "empty");
    CHECK(cmdcheck_base64("f", 1, out, sizeof(out)) == 4 && !strcmp(out, "Zg=="), "f -> %s", out);
    CHECK(cmdcheck_base64("fo", 2, out, sizeof(out)) == 4 && !strcmp(out, "Zm8="), "fo -> %s", out);
    CHECK(cmdcheck_base64("foobar", 6, out, sizeof(out)) == 8 && !strcmp(out, "Zm9vYmFy"), "foobar -> %s", out);
    CHECK(cmdcheck_base64("\xc3\xa9\x1e", 3, out, sizeof(out)) == 4 && !strcmp(out, "w6ke"), "bytes -> %s", out);
    CHECK(cmdcheck_base64("foobar", 6, out, 8) == 0, "does not fit");
}

int main(void) {
    test_fences();
    test_inline_and_prose();
    test_limits();
    test_format();
    test_sh_commands();
    test_posix_extract();
    test_format_posix();
    test_sh_script();
    test_base64();
    if (failures) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    puts("cmdcheck: ok");
    return 0;
}
