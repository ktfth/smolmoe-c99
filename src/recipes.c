#include "recipes.h"
#include "textfold.h"

#include <stddef.h>

/* Where a command runs: the shell and, for the commands that differ, the system.
 * PowerShell 7 on Linux or macOS has no CIM; Git Bash and MSYS2 are POSIX shells
 * on Windows without free or the Linux ps. */
enum { PS_WINDOWS, PS_OTHER, CMD, SH_LINUX, SH_MACOS, SH_WINDOWS, TARGETS };

typedef struct Entry {
    const char *topic;
    const char *const *need[2];      /* every non-NULL group must match one of its words */
    const char *command[TARGETS];    /* NULL: no checked command there, the model answers */
} Entry;

#define WORDS(...) ((const char *const[]){__VA_ARGS__, NULL})

/* PowerShell: explicit properties that exist on that object, sizes rounded in GB
 * or MB, few columns. The default tables drop columns that do not fit the
 * window, and Select-Object prints an empty column for a missing property. */
#define PS_DISK                                                                                                   \
    "Get-PSDrive -PSProvider FileSystem | Where-Object Used | Select-Object Name, Root, "                         \
    "@{n='Total(GB)';e={[math]::Round(($_.Used + $_.Free) / 1GB, 1)}}, "                                          \
    "@{n='Usado(GB)';e={[math]::Round($_.Used / 1GB, 1)}}, @{n='Livre(GB)';e={[math]::Round($_.Free / 1GB, 1)}}"
#define PS_MEMORY                                                                                                 \
    "Get-CimInstance Win32_OperatingSystem | Select-Object "                                                      \
    "@{n='Total(GB)';e={[math]::Round($_.TotalVisibleMemorySize / 1MB, 1)}}, "                                    \
    "@{n='Livre(GB)';e={[math]::Round($_.FreePhysicalMemory / 1MB, 1)}}"
#define PS_PROCESSES(key)                                                                                         \
    "Get-Process | Sort-Object " key " -Descending | Select-Object -First 10 Name, Id, "                         \
    "@{n='CPU(s)';e={[math]::Round($_.CPU, 1)}}, @{n='Mem(MB)';e={[math]::Round($_.WorkingSet64 / 1MB)}}"
#define PS_FOLDER                                                                                                 \
    "Get-ChildItem -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum | " \
    "Select-Object @{n='Arquivos';e={$_.Count}}, @{n='Total(MB)';e={[math]::Round($_.Sum / 1MB, 1)}}"
#define PS_LARGEST                                                                                                \
    "Get-ChildItem -Recurse -File -Force -ErrorAction SilentlyContinue | Sort-Object Length -Descending | "       \
    "Select-Object -First 10 @{n='MB';e={[math]::Round($_.Length / 1MB, 1)}}, FullName"
#define SH_LARGEST "find . -type f -exec du -k {} + 2>/dev/null | sort -rn | head -n 10"

static const Entry ENTRIES[] = {
    /* The specific topics first: "os maiores arquivos do disco" is about files. */
    {"os 10 maiores arquivos desta pasta e das subpastas (tamanho em KB no bash)",
     {WORDS("arquivo", "arquivos"), WORDS("maior", "maiores", "grande", "grandes", "pesado", "pesados")},
     {PS_LARGEST, PS_LARGEST, NULL, SH_LARGEST, SH_LARGEST, SH_LARGEST}},
    {"o tamanho desta pasta",
     {WORDS("pasta", "diretorio"), WORDS("tamanho", "ocupa", "espaco", "pesa", "grande")},
     {PS_FOLDER, PS_FOLDER, NULL, "du -sh .", "du -sh .", "du -sh ."}},
    {"os 10 processos que mais usam memória",
     {WORDS("processo", "processos", "programa", "programas", "aplicativo", "aplicativos"), WORDS("memoria", "ram")},
     {PS_PROCESSES("WorkingSet64"), PS_PROCESSES("WorkingSet64"), NULL, "ps aux --sort=-%mem | head -n 11",
      "ps aux -m | head -n 11", NULL}},
    {"os 10 processos que mais usam CPU",
     {WORDS("processo", "processos"), NULL},
     {PS_PROCESSES("CPU"), PS_PROCESSES("CPU"), "tasklist", "ps aux --sort=-%cpu | head -n 11", "ps aux -r | head -n 11",
      NULL}},
    {"a memória RAM total e livre",
     {WORDS("memoria", "ram"), NULL},
     {PS_MEMORY, NULL, NULL, "free -h", "top -l 1 -s 0 | grep PhysMem", NULL}},
    {"o espaço em disco (total, usado e livre)",
     {WORDS("disco", "discos", "hd", "ssd", "armazenamento", "particao", "particoes"), NULL},
     {PS_DISK, PS_DISK, NULL, "df -h", "df -h", "df -h"}},
    {"o espaço em disco (total, usado e livre)",
     {WORDS("espaco"), WORDS("livre", "disponivel", "sobrando", "ocupado", "usado")},
     {PS_DISK, PS_DISK, NULL, "df -h", "df -h", "df -h"}},
    {"os endereços IP desta máquina",
     {WORDS("ip", "ips"), NULL},
     {"Get-NetIPAddress -AddressFamily IPv4 | Select-Object InterfaceAlias, IPAddress, PrefixLength", NULL, "ipconfig",
      "ip -brief address", "ifconfig | grep \"inet \"", "ipconfig"}},
    {"a versão do sistema operacional",
     {WORDS("versao"), WORDS("windows", "sistema", "linux", "macos")},
     {"Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, OSArchitecture", "$PSVersionTable.OS",
      "ver", "cat /etc/os-release", "sw_vers", NULL}},
};

/* Asking for something to be shown or for a command. */
static const char *const REQUEST[] = {"comando", "comandos", "verificar", "verifique", "verifica", "checar", "cheque",
                                      "ver", "vejo", "veja", "mostrar", "mostre", "mostra", "listar", "liste", "lista",
                                      "exibir", "exiba", "consultar", "consulte", "saber", "descobrir", "descubro",
                                      "quanto", "quanta", "quantos", "quantas", "qual", "quais", "como", NULL};
/* Questions about concepts or advice, which the model answers. */
static const char *const CONCEPT[] = {"o que e", "o que sao", "diferenca", "explique", "explica", "explicar",
                                      "significa", "por que", "porque", "funciona", "serve", "comprar", "melhor",
                                      "recomenda", NULL};

static int has_any(const char *folded, const char *const *words) {
    for (; *words; ++words)
        if (text_find_word(folded, folded, *words)) return 1;
    return 0;
}

RecipeOS recipe_host_os(void) {
#if defined(_WIN32)
    return RECIPE_WINDOWS;
#elif defined(__APPLE__)
    return RECIPE_MACOS;
#else
    return RECIPE_LINUX;
#endif
}

static int target(ShellKind shell, RecipeOS os) {
    switch (shell) {
    case SHELL_POWERSHELL: return os == RECIPE_WINDOWS ? PS_WINDOWS : PS_OTHER;
    case SHELL_CMD: return CMD;
    default: return os == RECIPE_WINDOWS ? SH_WINDOWS : os == RECIPE_MACOS ? SH_MACOS : SH_LINUX;
    }
}

int recipe_find(const char *message, ShellKind shell, RecipeOS os, Recipe *out) {
    char folded[FOLD_MAX];
    size_t i;
    int t = target(shell, os);
    if (!text_fold(message, folded, sizeof(folded))) return 0; /* a long text is not a quick request */
    if (!has_any(folded, REQUEST) || has_any(folded, CONCEPT)) return 0;
    for (i = 0; i < sizeof(ENTRIES) / sizeof(ENTRIES[0]); ++i) {
        const Entry *e = &ENTRIES[i];
        if (!has_any(folded, e->need[0]) || (e->need[1] && !has_any(folded, e->need[1]))) continue;
        if (!e->command[t]) return 0; /* the topic matched: a broader entry would answer something else */
        out->topic = e->topic;
        out->command = e->command[t];
        return 1;
    }
    return 0;
}
