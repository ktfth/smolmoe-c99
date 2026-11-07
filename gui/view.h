// The desktop window's screens, drawn with Dear ImGui every frame: picking a
// model, opening it, the conversation, and a fatal error. The platform layer
// (platform_win32.cpp, platform_glfw.cpp) owns the window and the renderer.
#ifndef SMOLMOE_GUI_VIEW_H
#define SMOLMOE_GUI_VIEW_H

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "imgui.h"
#include "session.h"

struct Platform {
    // A native "open file" dialog; empty when the platform has none (the
    // window then shows a path field). Returns nullopt when cancelled.
    std::function<std::optional<std::string>()> pick_file;
    std::function<void(const std::string &title)> set_title;
};

// Fonts and colors, rebuilt when the window's scale changes.
struct Look {
    ImFont *large = nullptr;  // titles; nullptr: the regular font
    float scale = 1.0f;       // sizes in pixels at 96 DPI are multiplied by this
    bool light = false;
};

// Theme and spacing for `scale`, light or dark.
void apply_style(const Look &look);

// SMOLMOE_GUI_THEME=light or dark (claro, escuro) wins over the system's choice.
bool theme_is_light(bool system_light);

// Loads the first TrueType font that exists (SMOLMOE_GUI_FONT first), at
// `pixel_scale` times the base size, with the Latin, punctuation and symbol
// ranges a Portuguese chat needs; Dear ImGui's own font when none does.
// Returns the large font (or nullptr). Call before the renderer builds the atlas.
ImFont *load_fonts(float pixel_scale, const std::vector<std::string> &candidates);

class ChatView {
public:
    ChatView(ChatSession &session, Platform platform);

    void draw(const Look &look);           // one frame, between NewFrame() and Render()
    void drop(const std::string &path);    // a file dropped on the window
    bool animating() const;                // loading or replying: keep redrawing
    bool close_requested() const { return close_requested_; }

private:
    void refresh();
    void draw_pick();
    void draw_opening();
    void draw_failed();
    void draw_chat();
    void draw_header();
    void draw_transcript(float height);
    void draw_entry(const Entry &entry, bool last);
    void draw_input();
    void draw_status();
    void send_input();
    void title(const char *text) const;

    ChatSession &session_;
    const Platform platform_;
    Look look_;
    SessionState state_;
    std::string input_;
    std::string path_input_;
    std::string shown_name_;
    bool content_changed_ = false;
    bool scroll_to_bottom_ = false;
    bool focus_input_ = true;
    bool close_requested_ = false;
};

#endif
