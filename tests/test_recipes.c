/* Unit tests for the chat's ready-made shell commands (recipes.h): which
 * messages get a recipe, which go to the model, and that each command is the
 * one for that shell and system. Portable: nothing here runs a command. */
#include <stdio.h>
#include <string.h>

#include "recipes.h"

static int failures;

#define CHECK(cond, ...)                                         \
    do {                                                         \
        if (!(cond)) {                                           \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                        \
            fputc('\n', stderr);                                 \
            ++failures;                                          \
        }                                                        \
    } while (0)

/* The message gets a recipe whose command starts with `prefix`. */
static void expect(const char *message, ShellKind shell, RecipeOS os, const char *prefix) {
    Recipe r;
    int ok = recipe_find(message, shell, os, &r);
    CHECK(ok && !strncmp(r.command, prefix, strlen(prefix)), "'%s' -> %s (want %s)", message,
          ok ? r.command : "(model)", prefix);
}

static void expect_model(const char *message, ShellKind shell, RecipeOS os) {
    Recipe r;
    int ok = recipe_find(message, shell, os, &r);
    CHECK(!ok, "'%s' should go to the model, got %s", message, ok ? r.command : "");
}

/* The questions Gemma 3 1B answered with wrong commands (Get-PSDrive | Select-Object
 * Size, FreeSpace; Get-Disk without free space). */
static void test_disk(void) {
    const ShellKind ps = SHELL_POWERSHELL;
    const RecipeOS win = RECIPE_WINDOWS;
    Recipe r;
    expect("Qual comando do PowerShell mostra a capacidade do disco?", ps, win, "Get-PSDrive -PSProvider FileSystem");
    expect("Me dê um comando no PowerShell para verificar o disco, com capacidade total e espaço livre.", ps, win,
           "Get-PSDrive");
    expect("verifique o disco no powershell", ps, win, "Get-PSDrive");
    expect("Como vejo o tamanho e o espaço livre do disco C: no PowerShell?", ps, win, "Get-PSDrive");
    expect("Quanto espaço livre tem no meu computador?", ps, win, "Get-PSDrive");
    expect("QUANTO ESPAÇO TEM NO HD?", ps, win, "Get-PSDrive");
    expect("verifique o disco", SHELL_POSIX, RECIPE_LINUX, "df -h");
    expect("verifique o disco", SHELL_POSIX, RECIPE_MACOS, "df -h");
    expect("verifique o disco", SHELL_POSIX, RECIPE_WINDOWS, "df -h");
    expect("verifique o disco", ps, RECIPE_LINUX, "Get-PSDrive");
    expect_model("verifique o disco", SHELL_CMD, win); /* no checked cmd command: the model answers */
    /* Only properties that exist on PSDriveInfo, never the ones Gemma made up. */
    CHECK(recipe_find("verifique o disco", ps, win, &r) && strstr(r.command, "$_.Used") && strstr(r.command, "$_.Free") &&
              !strstr(r.command, "FreeSpace") && !strstr(r.command, "Size"),
          "disk command: %s", r.command);
    CHECK(!strcmp(r.topic, "o espaço em disco (total, usado e livre)"), "topic: %s", r.topic);
}

static void test_topics(void) {
    const ShellKind ps = SHELL_POWERSHELL;
    const RecipeOS win = RECIPE_WINDOWS;
    expect("quanta memória ram eu tenho?", ps, win, "Get-CimInstance Win32_OperatingSystem");
    expect("quanta memória ram eu tenho?", SHELL_POSIX, RECIPE_LINUX, "free -h");
    expect("quanta memória ram eu tenho?", SHELL_POSIX, RECIPE_MACOS, "top -l 1");
    expect_model("quanta memória ram eu tenho?", ps, RECIPE_LINUX); /* no CIM outside Windows */
    expect_model("quanta memória ram eu tenho?", SHELL_POSIX, RECIPE_WINDOWS); /* Git Bash has no free */
    expect("quais processos usam mais memória?", ps, win, "Get-Process | Sort-Object WorkingSet64");
    expect("quais processos usam mais memória?", SHELL_POSIX, RECIPE_LINUX, "ps aux --sort=-%mem");
    expect("liste os processos", ps, win, "Get-Process | Sort-Object CPU");
    expect("liste os processos", SHELL_CMD, win, "tasklist");
    expect("liste os processos", SHELL_POSIX, RECIPE_MACOS, "ps aux -r");
    expect("qual o meu ip?", ps, win, "Get-NetIPAddress");
    expect("qual o meu ip?", SHELL_CMD, win, "ipconfig");
    expect("qual o meu ip?", SHELL_POSIX, RECIPE_LINUX, "ip -brief address");
    expect("quais os maiores arquivos do disco?", ps, win, "Get-ChildItem -Recurse -File");
    expect("quais os maiores arquivos do disco?", SHELL_POSIX, RECIPE_LINUX, "find . -type f");
    expect("qual o tamanho desta pasta?", SHELL_POSIX, RECIPE_LINUX, "du -sh .");
    expect("qual o tamanho desta pasta?", ps, win, "Get-ChildItem -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object");
    expect("qual a versão do windows?", ps, win, "Get-CimInstance Win32_OperatingSystem | Select-Object Caption");
    expect("qual a versão do windows?", SHELL_CMD, win, "ver");
    expect("qual a versão do sistema?", SHELL_POSIX, RECIPE_MACOS, "sw_vers");
}

/* Concepts, advice and unrelated questions stay with the model. */
static void test_model(void) {
    const ShellKind ps = SHELL_POWERSHELL;
    const RecipeOS win = RECIPE_WINDOWS;
    expect_model("o que é um disco rígido?", ps, win);
    expect_model("qual a diferença entre memória RAM e disco?", ps, win);
    expect_model("explique como funciona a memória", ps, win);
    expect_model("qual SSD é melhor para comprar?", ps, win);
    expect_model("por que meu disco está cheio?", ps, win);
    expect_model("meu disco está fazendo barulho", ps, win); /* no request word */
    expect_model("qual a capital da França?", ps, win);
    expect_model("quantos dias faltam para o natal?", ps, win);
    expect_model("qual a versão do python?", ps, win);
    expect_model("como renomear um arquivo no PowerShell?", ps, win);
    expect_model("", ps, win);
}

int main(void) {
    test_disk();
    test_topics();
    test_model();
    if (failures) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    puts("recipes: ok");
    return 0;
}
