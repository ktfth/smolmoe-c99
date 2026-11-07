#include "args.h"

#include <cstring>

extern "C" {
#include "config.h"
#include "util.h"
}

namespace {

constexpr uint32_t kMaxThreads = 256;

}  // namespace

const char *gui_usage() {
    return "smolmoe-gui " SMOLMOE_VERSION " - chat local numa janela\n\n"
           "uso: smolmoe-gui [MODELO.smoe] [opções]\n"
           "  Sem modelo, abre o que vem dentro do executável (smolmoe-bundle) ou pede um arquivo.\n"
           "  O .smoe pode ser autocontido ou ter config.ini e tokenizer.qbp ao lado.\n\n"
           "  --context N   posições do KV cache (padrão 4096, limitado ao do modelo)\n"
           "  --threads N   threads de cálculo (padrão: os núcleos físicos)\n"
           "  --version, --help\n";
}

const char *gui_version() { return "smolmoe-gui " SMOLMOE_VERSION "\n"; }

GuiArgs parse_gui_args(int argc, char **argv) {
    GuiArgs args;
    for (int a = 1; a < argc; ++a) {
        const char *opt = argv[a];
        if (!std::strcmp(opt, "--help") || !std::strcmp(opt, "-h")) {
            args.help = true;
        } else if (!std::strcmp(opt, "--version")) {
            args.version = true;
        } else if (!std::strcmp(opt, "--context") || !std::strcmp(opt, "--threads")) {
            if (a + 1 >= argc) die("missing argument for %s", opt);
            const bool context = !std::strcmp(opt, "--context");
            const uint32_t value = parse_u32(argv[++a], context ? MAX_CONTEXT : kMaxThreads, opt + 2);
            if (!value) die("%s must be positive", opt);
            (context ? args.session.context : args.session.threads) = value;
        } else if (opt[0] == '-' && opt[1]) {
            die("unknown option: %s (see --help)", opt);
        } else if (!args.model.empty()) {
            die("only one model file: %s", opt);
        } else {
            args.model = opt;
        }
    }
    return args;
}
