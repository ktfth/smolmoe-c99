// The desktop window's command line: smolmoe-gui [MODEL.smoe] [--context N] [--threads N].
#ifndef SMOLMOE_GUI_ARGS_H
#define SMOLMOE_GUI_ARGS_H

#include <string>

#include "session.h"

struct GuiArgs {
    std::string model;  // empty: the model inside this executable, or the file picker
    SessionOptions session;
    bool help = false;
    bool version = false;
};

// argv in UTF-8 (platform_init on Windows). A bad value goes through die().
GuiArgs parse_gui_args(int argc, char **argv);
const char *gui_usage();
const char *gui_version();

#endif
