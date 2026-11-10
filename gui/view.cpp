#define IMGUI_DEFINE_MATH_OPERATORS
#include "view.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

#include "misc/cpp/imgui_stdlib.h"

extern "C" {
#include "platform.h"
}

namespace {

constexpr float kFontPixels = 17.0f;       // body text at 96 DPI
constexpr float kTitlePixels = 26.0f;
constexpr float kBubbleShare = 0.78f;      // a message bubble takes at most this much of the width
constexpr float kColumnPixels = 560.0f;    // width of the centered screens
constexpr int kInputLines = 3;
// Dear ImGui swaps Cmd and Ctrl on macOS, so the shortcuts are Cmd+N and Cmd+Enter there.
#if defined(__APPLE__)
#define SHORTCUT_KEY "Cmd"
#else
#define SHORTCUT_KEY "Ctrl"
#endif

// Latin-1, Latin Extended-A, general punctuation (quotes, dashes, ellipsis),
// currency, letterlike symbols, arrows, math operators, geometric shapes.
const ImWchar kGlyphRanges[] = {0x0020, 0x00FF, 0x0100, 0x017F, 0x2000, 0x206F, 0x20A0, 0x20CF, 0x2100, 0x214F,
                                0x2190, 0x21FF, 0x2200, 0x22FF, 0x25A0, 0x25FF, 0};

struct Palette {
    ImVec4 text, dim, error, accent, bubble, background;
};

Palette palette(bool light) {
    if (light)
        return {ImVec4(0.11f, 0.12f, 0.14f, 1), ImVec4(0.42f, 0.45f, 0.50f, 1), ImVec4(0.75f, 0.16f, 0.16f, 1),
                ImVec4(0.13f, 0.42f, 0.80f, 1), ImVec4(0.86f, 0.91f, 0.98f, 1), ImVec4(0.97f, 0.97f, 0.96f, 1)};
    return {ImVec4(0.90f, 0.91f, 0.93f, 1), ImVec4(0.56f, 0.59f, 0.64f, 1), ImVec4(0.96f, 0.45f, 0.42f, 1),
            ImVec4(0.42f, 0.66f, 1.00f, 1), ImVec4(0.17f, 0.24f, 0.36f, 1), ImVec4(0.09f, 0.10f, 0.11f, 1)};
}

bool file_exists(const std::string &path) {
    FILE *f = open_file(path.c_str(), "rb");
    if (!f) return false;
    std::fclose(f);
    return true;
}

// "12,3": decimals with a comma, as Brazilian Portuguese writes them.
std::string decimal(double value, int places) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", places, value);
    std::string s = buf;
    std::replace(s.begin(), s.end(), '.', ',');
    return s;
}

// Runtime messages start in lower case ("o arquivo não é..."); on screen they open a sentence.
std::string sentence(const std::string &s) {
    std::string out = s;
    if (!out.empty() && out[0] >= 'a' && out[0] <= 'z') out[0] = static_cast<char>(out[0] - 'a' + 'A');
    return out;
}

std::string trimmed(const std::string &s) {
    const char *space = " \t\r\n";
    const size_t first = s.find_first_not_of(space);
    if (first == std::string::npos) return {};
    return s.substr(first, s.find_last_not_of(space) - first + 1);
}

// Wrapped at the column's edge when one is set (begin_column), else at the window's.
void text(const std::string &s, const ImVec4 *color = nullptr) {
    if (color) ImGui::PushStyleColor(ImGuiCol_Text, *color);
    ImGui::TextWrapped("%s", s.c_str());  // "%s" is passed through, with no length limit
    if (color) ImGui::PopStyleColor();
}

void copy_menu(const std::string &s) {
    if (ImGui::BeginPopupContextItem("copiar")) {
        if (ImGui::MenuItem("Copiar")) ImGui::SetClipboardText(s.c_str());
        ImGui::EndPopup();
    }
}

// A centered column for the screens without a conversation.
struct Column {
    float width, indent;
};

Column begin_column(float scale, float top_share) {
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float width = std::min(kColumnPixels * scale, avail.x);
    const Column column{width, (avail.x - width) / 2};
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + avail.y * top_share);
    if (column.indent > 0) ImGui::Indent(column.indent);
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + width);
    return column;
}

void end_column(const Column &column) {
    ImGui::PopTextWrapPos();
    if (column.indent > 0) ImGui::Unindent(column.indent);
}

std::string dots() { return std::string(1 + static_cast<int>(ImGui::GetTime() * 2.5) % 3, '.'); }

}  // namespace

bool theme_is_light(bool system_light) {
    const char *theme = std::getenv("SMOLMOE_GUI_THEME");
    const std::string chosen = theme ? theme : "";
    if (chosen == "light" || chosen == "claro") return true;
    if (chosen == "dark" || chosen == "escuro") return false;
    return system_light;
}

void apply_style(const Look &look) {
    const Palette p = palette(look.light);
    ImGuiStyle style;
    if (look.light) ImGui::StyleColorsLight(&style);
    else ImGui::StyleColorsDark(&style);
    style.WindowPadding = ImVec2(18, 14);
    style.FramePadding = ImVec2(10, 7);
    style.ItemSpacing = ImVec2(10, 8);
    style.FrameRounding = 6;
    style.ChildRounding = 6;
    style.PopupRounding = 6;
    style.ScrollbarRounding = 6;
    style.GrabRounding = 6;
    style.WindowBorderSize = 0;
    style.ScrollbarSize = 12;
    style.Colors[ImGuiCol_Text] = p.text;
    style.Colors[ImGuiCol_TextDisabled] = p.dim;
    style.Colors[ImGuiCol_WindowBg] = p.background;
    style.Colors[ImGuiCol_ChildBg] = p.background;
    style.Colors[ImGuiCol_Button] = ImVec4(p.accent.x, p.accent.y, p.accent.z, 0.85f);
    style.Colors[ImGuiCol_ButtonHovered] = p.accent;
    style.Colors[ImGuiCol_ButtonActive] = ImVec4(p.accent.x * 0.85f, p.accent.y * 0.85f, p.accent.z * 0.85f, 1);
    style.ScaleAllSizes(look.scale);
    ImGui::GetStyle() = style;
}

ImFont *load_fonts(float pixel_scale, const std::vector<std::string> &candidates) {
    ImGuiIO &io = ImGui::GetIO();
    io.Fonts->Clear();
    std::vector<std::string> paths;
    if (const char *chosen = std::getenv("SMOLMOE_GUI_FONT")) paths.push_back(chosen);
    paths.insert(paths.end(), candidates.begin(), candidates.end());
    for (const std::string &path : paths) {
        // Dear ImGui asserts on a missing file, so look first.
        if (path.empty() || !file_exists(path)) continue;
        if (!io.Fonts->AddFontFromFileTTF(path.c_str(), kFontPixels * pixel_scale, nullptr, kGlyphRanges)) continue;
        return io.Fonts->AddFontFromFileTTF(path.c_str(), kTitlePixels * pixel_scale, nullptr, kGlyphRanges);
    }
    ImFontConfig config;
    config.SizePixels = std::round(13.0f * pixel_scale);
    io.Fonts->AddFontDefault(&config);
    return nullptr;
}

ChatView::ChatView(ChatSession &session, Platform platform) : session_(session), platform_(std::move(platform)) {}

bool ChatView::animating() const {
    return state_.phase == Phase::Opening || state_.phase == Phase::Busy;
}

void ChatView::drop(const std::string &path) {
    if (state_.phase == Phase::NoModel) session_.open(path);
}

void ChatView::refresh() {
    content_changed_ = false;
    std::optional<SessionState> next = session_.changed_since(state_.version);
    if (!next) return;
    const bool opened = state_.phase == Phase::Opening && next->phase == Phase::Ready;
    state_ = std::move(*next);
    content_changed_ = true;
    if (opened) focus_input_ = true;
    if (state_.model_name != shown_name_ && platform_.set_title) {
        shown_name_ = state_.model_name;
        platform_.set_title(shown_name_.empty() ? "smolmoe" : "smolmoe — " + shown_name_);
    }
}

void ChatView::title(const char *s) const {
    if (look_.large) ImGui::PushFont(look_.large);
    ImGui::TextUnformatted(s);
    if (look_.large) ImGui::PopFont();
}

void ChatView::draw(const Look &look) {
    look_ = look;
    refresh();
    const ImGuiViewport *viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus;
    ImGui::Begin("smolmoe", nullptr, flags);
    switch (state_.phase) {
    case Phase::NoModel: draw_pick(); break;
    case Phase::Opening: draw_opening(); break;
    case Phase::Failed: draw_failed(); break;
    case Phase::Ready:
    case Phase::Busy: draw_chat(); break;
    }
    ImGui::End();
}

void ChatView::draw_pick() {
    const Palette p = palette(look_.light);
    const Column column = begin_column(look_.scale, 0.18f);
    const float width = column.width;
    title("smolmoe");
    ImGui::Spacing();
    text("Converse com um modelo de linguagem no seu computador. Nada do que você escreve sai da sua máquina.");
    ImGui::Spacing();
    ImGui::Spacing();
    if (platform_.pick_file) {
        if (ImGui::Button("Abrir modelo (.smoe)…")) {
            if (std::optional<std::string> path = platform_.pick_file()) session_.open(*path);
        }
    } else {
        text("Caminho do modelo (.smoe):");
        ImGui::SetNextItemWidth(width - ImGui::CalcTextSize("Abrir").x - ImGui::GetStyle().FramePadding.x * 2 -
                                ImGui::GetStyle().ItemSpacing.x);
        const bool entered = ImGui::InputText("##caminho", &path_input_, ImGuiInputTextFlags_EnterReturnsTrue);
        ImGui::SameLine();
        if ((ImGui::Button("Abrir") || entered) && !trimmed(path_input_).empty()) session_.open(trimmed(path_input_));
    }
    ImGui::Spacing();
    text("Ou arraste um arquivo .smoe para esta janela.", &p.dim);
    text("Os modelos estão na página de releases do projeto, por exemplo qwen2.5-0.5b-instruct-q8.smoe (560 MB).",
         &p.dim);
    if (!state_.error.empty()) {
        ImGui::Spacing();
        text(sentence(state_.error), &p.error);
    }
    end_column(column);
}

void ChatView::draw_opening() {
    const Palette p = palette(look_.light);
    const Column column = begin_column(look_.scale, 0.30f);
    title(("Abrindo " + state_.model_name + dots()).c_str());
    ImGui::Spacing();
    text(state_.stage == DESKTOP_CHECKING ? "Conferindo os pesos, uma vez a cada abertura."
                                          : "Preparando o modelo e as instruções da conversa.");
    const double elapsed = now_seconds() - state_.phase_started;
    text(decimal(elapsed, 0) + " s", &p.dim);
    end_column(column);
}

void ChatView::draw_failed() {
    const Palette p = palette(look_.light);
    const Column column = begin_column(look_.scale, 0.25f);
    title("Não deu para continuar");
    ImGui::Spacing();
    text(sentence(state_.error), &p.error);
    ImGui::Spacing();
    text("Feche a janela e abra o programa de novo. Se o erro voltar, o arquivo do modelo pode estar corrompido: "
         "baixe-o outra vez.");
    ImGui::Spacing();
    if (ImGui::Button("Fechar")) close_requested_ = true;
    end_column(column);
}

void ChatView::draw_chat() {
    const ImGuiStyle &style = ImGui::GetStyle();
    // Over the message field, which would otherwise take them: Esc stops a
    // reply and never undoes a draft; Ctrl+N starts a new conversation.
    const ImGuiInputFlags over_field = ImGuiInputFlags_RouteGlobal | ImGuiInputFlags_RouteOverActive;
    if (ImGui::Shortcut(ImGuiKey_Escape, over_field) && state_.phase == Phase::Busy) session_.stop();
    if (ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_N, over_field) && state_.phase == Phase::Ready) session_.reset();
    draw_header();
    const float input_height = ImGui::GetTextLineHeight() * kInputLines + style.FramePadding.y * 2;
    const float footer = input_height + ImGui::GetTextLineHeightWithSpacing() + style.ItemSpacing.y * 3;
    draw_transcript(std::max(ImGui::GetContentRegionAvail().y - footer, ImGui::GetTextLineHeight()));
    draw_input();
    draw_status();
}

void ChatView::draw_header() {
    const Palette p = palette(look_.light);
    const DesktopInfo &info = state_.info;
    const char *reset_label = "Nova conversa";
    const float reset_width = ImGui::CalcTextSize(reset_label).x + ImGui::GetStyle().FramePadding.x * 2;
    const float right = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x;
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(state_.model_name.c_str());
    ImGui::SameLine();
    char details[160];
    std::snprintf(details, sizeof(details), "%s · %u threads · contexto %u%s", info.backend ? info.backend : "",
                  info.threads, info.context, info.tools ? " · ferramentas de data e contas" : "");
    ImGui::TextColored(p.dim, "%s", details);
    ImGui::SameLine(std::max(right - reset_width, ImGui::GetCursorPosX()));
    ImGui::BeginDisabled(state_.phase != Phase::Ready || state_.entries.empty());
    if (ImGui::Button(reset_label)) session_.reset();
    ImGui::SetItemTooltip(SHORTCUT_KEY "+N");
    ImGui::EndDisabled();
    ImGui::Separator();
}

void ChatView::draw_transcript(float height) {
    const Palette p = palette(look_.light);
    ImGui::BeginChild("##conversa", ImVec2(0, height));
    const bool at_bottom = ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 4.0f;
    if (state_.entries.empty()) {
        ImGui::Dummy(ImVec2(0, ImGui::GetTextLineHeight()));
        text("Escreva uma mensagem abaixo. O modelo responde em português e conta com ferramentas para datas e contas: "
             "experimente \"quanto é 15% de 230?\" ou \"quantos dias faltam para o Natal?\".",
             &p.dim);
    }
    for (size_t i = 0; i < state_.entries.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        draw_entry(state_.entries[i], i + 1 == state_.entries.size());
        ImGui::PopID();
    }
    const bool waiting = state_.phase == Phase::Busy &&
                         (state_.entries.empty() || state_.entries.back().role != Role::Assistant);
    if (waiting) text(dots(), &p.dim);
    if (scroll_to_bottom_ || (content_changed_ && at_bottom)) ImGui::SetScrollHereY(1.0f);
    scroll_to_bottom_ = false;
    ImGui::EndChild();
}

void ChatView::draw_entry(const Entry &entry, bool last) {
    const Palette p = palette(look_.light);
    const float gap = 10.0f * look_.scale;
    if (entry.role == Role::Assistant && entry.text.empty()) return;
    if (entry.role == Role::Notice) {
        text(entry.text, &p.dim);
        copy_menu(entry.text);
    } else if (entry.role == Role::Assistant) {
        const size_t shown = last && state_.phase == Phase::Busy ? complete_utf8(entry.text) : entry.text.size();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextUnformatted(entry.text.data(), entry.text.data() + shown);
        ImGui::PopTextWrapPos();
        copy_menu(entry.text);
    } else {
        // The user's message: a bubble on the right.
        const ImVec2 pad = ImVec2(12, 8) * look_.scale;
        const float avail = ImGui::GetContentRegionAvail().x;
        const float wrap = std::max(avail * kBubbleShare - pad.x * 2, 40.0f * look_.scale);
        const char *begin = entry.text.data(), *end = begin + entry.text.size();
        const ImVec2 size = ImGui::CalcTextSize(begin, end, false, wrap);
        const ImVec2 box = size + pad * 2;
        const ImVec2 start = ImGui::GetCursorScreenPos() + ImVec2(std::max(avail - box.x, 0.0f), 0);
        ImGui::GetWindowDrawList()->AddRectFilled(start, start + box, ImGui::GetColorU32(p.bubble), 10.0f * look_.scale);
        ImGui::SetCursorScreenPos(start + pad);
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + wrap);
        ImGui::TextUnformatted(begin, end);
        ImGui::PopTextWrapPos();
        ImGui::SetCursorScreenPos(start);
        ImGui::Dummy(box);
        copy_menu(entry.text);
    }
    ImGui::Dummy(ImVec2(0, gap));
}

void ChatView::send_input() {
    const std::string message = trimmed(input_);
    if (message.empty() || !session_.send(message)) return;
    input_.clear();
    scroll_to_bottom_ = true;
}

void ChatView::draw_input() {
    const Palette p = palette(look_.light);
    const ImGuiStyle &style = ImGui::GetStyle();
    const bool busy = state_.phase == Phase::Busy;
    const float button_width = ImGui::CalcTextSize("Enviar").x + style.FramePadding.x * 4;
    const float height = ImGui::GetTextLineHeight() * kInputLines + style.FramePadding.y * 2;
    const ImVec2 size(ImGui::GetContentRegionAvail().x - button_width - style.ItemSpacing.x, height);
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    if (focus_input_) {
        ImGui::SetKeyboardFocusHere();
        focus_input_ = false;
    }
    // Enter sends, Ctrl+Enter breaks the line.
    const ImGuiInputTextFlags flags = ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CtrlEnterForNewLine;
    const bool entered = ImGui::InputTextMultiline("##mensagem", &input_, size, flags);
    if (input_.empty() && !ImGui::IsItemActive()) {
        const ImVec4 clip(origin.x, origin.y, origin.x + size.x - style.FramePadding.x, origin.y + size.y);
        ImGui::GetWindowDrawList()->AddText(ImGui::GetFont(), ImGui::GetFontSize(), origin + style.FramePadding,
                                            ImGui::GetColorU32(p.dim),
                                            "Escreva sua mensagem (Enter envia, " SHORTCUT_KEY "+Enter quebra a linha)", nullptr,
                                            0.0f, &clip);
    }
    if (entered) {
        if (!busy) send_input();
        focus_input_ = true;  // Enter leaves the field
    }
    ImGui::SameLine();
    if (busy) {
        if (ImGui::Button("Parar", ImVec2(button_width, height))) session_.stop();
    } else {
        ImGui::BeginDisabled(trimmed(input_).empty());
        if (ImGui::Button("Enviar", ImVec2(button_width, height))) {
            send_input();
            focus_input_ = true;
        }
        ImGui::EndDisabled();
    }
}

void ChatView::draw_status() {
    const Palette p = palette(look_.light);
    std::string status;
    if (state_.phase == Phase::Busy) {
        status = "Respondendo" + dots() + " (Esc ou Parar interrompe)";
    } else if (!state_.error.empty()) {
        status = sentence(state_.error);
    } else if (state_.last_stopped) {
        status = "Resposta interrompida.";
    } else if (state_.last_rate > 0) {
        status = decimal(state_.last_rate, 1) + " tokens por segundo na última resposta.";
    } else {
        status = "Pronto.";
    }
    ImGui::TextColored(!state_.error.empty() && state_.phase != Phase::Busy ? p.error : p.dim, "%s", status.c_str());
}
