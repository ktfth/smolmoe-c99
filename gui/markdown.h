// Lightweight Markdown pretty-printer for Dear ImGui with selectable code blocks
// and one-click copy buttons.
#ifndef SMOLMOE_GUI_MARKDOWN_H
#define SMOLMOE_GUI_MARKDOWN_H

#include <string>
#include <string_view>

#include "imgui.h"

struct MarkdownStyle {
    ImFont *large = nullptr;
    ImFont *bold = nullptr;
    ImFont *mono = nullptr;
    ImVec4 text;
    ImVec4 dim;
    ImVec4 accent;
    ImVec4 code_bg;
    ImVec4 code_border;
    ImVec4 inline_code_bg;
    ImVec4 quote_bar;
    float scale = 1.0f;
};

struct CopyFeedback {
    std::string id;
    double until = 0.0;
};

void draw_markdown(std::string_view text, const MarkdownStyle &style, int entry_id, CopyFeedback &copied);

#endif
