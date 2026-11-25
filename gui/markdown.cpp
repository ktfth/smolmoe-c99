#define IMGUI_DEFINE_MATH_OPERATORS
#include "markdown.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

struct Span {
    enum Kind { Plain, Bold, Italic, Code, Link } kind = Plain;
    std::string text;
};

std::string trim_sv(std::string_view sv) {
    size_t a = 0, b = sv.size();
    while (a < b && (sv[a] == ' ' || sv[a] == '\t' || sv[a] == '\r')) ++a;
    while (b > a && (sv[b - 1] == ' ' || sv[b - 1] == '\t' || sv[b - 1] == '\r')) --b;
    return std::string(sv.substr(a, b - a));
}

std::vector<Span> parse_inline(std::string_view s) {
    std::vector<Span> out;
    std::string buf;
    auto flush = [&]() {
        if (!buf.empty()) {
            out.push_back({Span::Plain, std::move(buf)});
            buf.clear();
        }
    };
    size_t i = 0;
    while (i < s.size()) {
        if (s[i] == '`') {
            const size_t end = s.find('`', i + 1);
            if (end != std::string_view::npos && end > i + 1) {
                flush();
                out.push_back({Span::Code, std::string(s.substr(i + 1, end - i - 1))});
                i = end + 1;
                continue;
            }
        } else if (i + 1 < s.size() && ((s[i] == '*' && s[i + 1] == '*') || (s[i] == '_' && s[i + 1] == '_'))) {
            const char delim[3] = {s[i], s[i + 1], 0};
            const size_t end = s.find(delim, i + 2);
            if (end != std::string_view::npos && end > i + 2) {
                flush();
                out.push_back({Span::Bold, std::string(s.substr(i + 2, end - i - 2))});
                i = end + 2;
                continue;
            }
        } else if (s[i] == '*' && i + 1 < s.size() && s[i + 1] != ' ' && s[i + 1] != '*') {
            const size_t end = s.find('*', i + 1);
            if (end != std::string_view::npos && end > i + 1 && s[end - 1] != ' ') {
                flush();
                out.push_back({Span::Italic, std::string(s.substr(i + 1, end - i - 1))});
                i = end + 1;
                continue;
            }
        } else if (s[i] == '[') {
            const size_t close_b = s.find(']', i + 1);
            if (close_b != std::string_view::npos && close_b + 1 < s.size() && s[close_b + 1] == '(') {
                const size_t close_p = s.find(')', close_b + 2);
                if (close_p != std::string_view::npos && close_b > i + 1) {
                    flush();
                    out.push_back({Span::Link, std::string(s.substr(i + 1, close_b - i - 1))});
                    i = close_p + 1;
                    continue;
                }
            }
        }
        buf.push_back(s[i]);
        ++i;
    }
    flush();
    return out;
}

// Draws rich inline spans with word wrapping inside the current content region.
void draw_inline_wrapped(std::string_view line, const MarkdownStyle &st, const ImVec4 *override_color = nullptr,
                         ImFont *override_font = nullptr) {
    const std::vector<Span> spans = parse_inline(line);
    if (spans.empty()) {
        ImGui::NewLine();
        return;
    }
    const float start_x = ImGui::GetCursorPosX();
    const float max_x = start_x + std::max(ImGui::GetContentRegionAvail().x, 40.0f * st.scale);
    float cur_x = start_x;
    bool first_on_line = true;

    for (const Span &sp : spans) {
        ImFont *font = override_font;
        ImVec4 col = override_color ? *override_color : st.text;
        bool pill = false;
        if (sp.kind == Span::Bold) {
            if (!font && st.bold) font = st.bold;
        } else if (sp.kind == Span::Code) {
            if (st.mono) font = st.mono;
            col = st.accent;
            pill = true;
        } else if (sp.kind == Span::Italic) {
            col = override_color ? *override_color : ImVec4(st.text.x, st.text.y, st.text.z, 0.85f);
        } else if (sp.kind == Span::Link) {
            col = st.accent;
        }

        if (font) ImGui::PushFont(font);
        if (pill) {
            const ImVec2 pad(4.0f * st.scale, 1.0f * st.scale);
            const ImVec2 sz = ImGui::CalcTextSize(sp.text.c_str(), sp.text.c_str() + sp.text.size());
            const float box_w = sz.x + pad.x * 2.0f;
            if (!first_on_line && cur_x + box_w > max_x) {
                cur_x = start_x;
                first_on_line = true;
            }
            if (!first_on_line) ImGui::SameLine(0.0f, 0.0f);
            else ImGui::SetCursorPosX(start_x);
            const ImVec2 p0 = ImGui::GetCursorScreenPos();
            ImGui::GetWindowDrawList()->AddRectFilled(
                p0 - ImVec2(0.0f, pad.y), p0 + ImVec2(box_w, sz.y + pad.y),
                ImGui::GetColorU32(st.inline_code_bg), 4.0f * st.scale);
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + pad.x);
            ImGui::PushStyleColor(ImGuiCol_Text, col);
            ImGui::TextUnformatted(sp.text.c_str(), sp.text.c_str() + sp.text.size());
            ImGui::PopStyleColor();
            ImGui::SameLine(0.0f, 0.0f);
            ImGui::Dummy(ImVec2(pad.x, 0.0f));
            cur_x += box_w;
            first_on_line = false;
            if (font) ImGui::PopFont();
            continue;
        }

        // Split into words (keeping trailing spaces attached to words) so lines wrap naturally.
        size_t pos = 0;
        while (pos < sp.text.size()) {
            size_t next = pos;
            while (next < sp.text.size() && sp.text[next] != ' ' && sp.text[next] != '\t') ++next;
            while (next < sp.text.size() && (sp.text[next] == ' ' || sp.text[next] == '\t')) ++next;
            const char *w0 = sp.text.data() + pos;
            const char *w1 = sp.text.data() + next;
            const ImVec2 sz = ImGui::CalcTextSize(w0, w1);
            if (!first_on_line && cur_x + sz.x > max_x) {
                cur_x = start_x;
                first_on_line = true;
                while (w0 < w1 && (*w0 == ' ' || *w0 == '\t')) ++w0;
                if (w0 == w1) {
                    pos = next;
                    continue;
                }
            }
            if (!first_on_line) ImGui::SameLine(0.0f, 0.0f);
            else ImGui::SetCursorPosX(start_x);
            ImGui::PushStyleColor(ImGuiCol_Text, col);
            ImGui::TextUnformatted(w0, w1);
            ImGui::PopStyleColor();
            cur_x += ImGui::CalcTextSize(w0, w1).x;
            first_on_line = false;
            pos = next;
        }
        if (font) ImGui::PopFont();
    }
}

bool is_hr_line(std::string_view s) {
    size_t count = 0;
    char ch = 0;
    for (char c : s) {
        if (c == ' ' || c == '\t' || c == '\r') continue;
        if (c != '-' && c != '*' && c != '_') return false;
        if (!ch) ch = c;
        else if (c != ch) return false;
        ++count;
    }
    return count >= 3;
}

bool is_table_sep(std::string_view s) {
    bool has_dash = false, has_pipe = false;
    for (char c : s) {
        if (c == '|') has_pipe = true;
        else if (c == '-') has_dash = true;
        else if (c != ':' && c != ' ' && c != '\t' && c != '\r') return false;
    }
    return has_dash && has_pipe;
}

std::vector<std::string> split_table_row(std::string_view s) {
    std::string t = trim_sv(s);
    if (!t.empty() && t.front() == '|') t.erase(t.begin());
    if (!t.empty() && t.back() == '|') t.pop_back();
    std::vector<std::string> cells;
    size_t start = 0;
    for (;;) {
        const size_t p = t.find('|', start);
        if (p == std::string::npos) {
            cells.push_back(trim_sv(std::string_view(t).substr(start)));
            break;
        }
        cells.push_back(trim_sv(std::string_view(t).substr(start, p - start)));
        start = p + 1;
    }
    return cells;
}

std::vector<std::string_view> split_lines(std::string_view text) {
    std::vector<std::string_view> lines;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t nl = text.find('\n', start);
        if (nl == std::string_view::npos) {
            lines.push_back(text.substr(start));
            break;
        }
        std::string_view ln = text.substr(start, nl - start);
        if (!ln.empty() && ln.back() == '\r') ln.remove_suffix(1);
        lines.push_back(ln);
        start = nl + 1;
    }
    return lines;
}

bool ordered_list_prefix(std::string_view s, size_t &prefix_end, std::string &marker) {
    size_t i = 0;
    while (i < s.size() && s[i] == ' ') ++i;
    const size_t d0 = i;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
    if (i == d0 || i - d0 > 4 || i + 1 >= s.size() || (s[i] != '.' && s[i] != ')') || s[i + 1] != ' ')
        return false;
    marker = std::string(s.substr(d0, i - d0 + 1));
    prefix_end = i + 2;
    return true;
}

void draw_code_block(const std::string &lang, const std::string &code, const MarkdownStyle &st, int entry_id,
                     int block_idx, CopyFeedback &copied) {
    const float avail = std::max(ImGui::GetContentRegionAvail().x, 120.0f * st.scale);
    const std::string copy_key = "code_" + std::to_string(entry_id) + "_" + std::to_string(block_idx);
    const bool is_copied = copied.id == copy_key && ImGui::GetTime() < copied.until;

    ImGui::PushStyleColor(ImGuiCol_ChildBg, st.code_bg);
    ImGui::PushStyleColor(ImGuiCol_Border, st.code_border);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f * st.scale);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f * st.scale, 8.0f * st.scale));

    const int line_count = std::max(1, static_cast<int>(std::count(code.begin(), code.end(), '\n')) + 1);
    const float line_h = st.mono ? st.mono->FontSize : ImGui::GetTextLineHeight();
    const float header_h = ImGui::GetFrameHeight() + 4.0f * st.scale;
    const float body_h = std::min(static_cast<float>(line_count) * (line_h + 2.0f * st.scale) + 12.0f * st.scale,
                                  360.0f * st.scale);
    const float total_h = header_h + body_h + 18.0f * st.scale;

    char cid[48];
    std::snprintf(cid, sizeof(cid), "##code_%d_%d", entry_id, block_idx);
    if (ImGui::BeginChild(cid, ImVec2(avail, total_h), ImGuiChildFlags_Borders, ImGuiWindowFlags_NoScrollbar)) {
        const char *label = lang.empty() ? "código" : lang.c_str();
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(st.dim, "%s", label);
        const char *btn_label = is_copied ? "Copiado!" : "Copiar código";
        const float btn_w = ImGui::CalcTextSize(btn_label).x + ImGui::GetStyle().FramePadding.x * 2.0f;
        ImGui::SameLine(std::max(ImGui::GetCursorPosX(), ImGui::GetContentRegionAvail().x + ImGui::GetCursorPosX() - btn_w));
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, st.inline_code_bg);
        ImGui::PushStyleColor(ImGuiCol_Text, is_copied ? st.accent : st.dim);
        if (ImGui::SmallButton(btn_label)) {
            ImGui::SetClipboardText(code.c_str());
            copied.id = copy_key;
            copied.until = ImGui::GetTime() + 1.5;
        }
        ImGui::PopStyleColor(3);
        ImGui::Separator();

        if (st.mono) ImGui::PushFont(st.mono);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2.0f * st.scale, 4.0f * st.scale));
        // Read-only InputTextMultiline lets the user click, drag-select any text, and press Ctrl+C.
        char iid[48];
        std::snprintf(iid, sizeof(iid), "##code_in_%d_%d", entry_id, block_idx);
        std::string mutable_copy = code;
        ImGui::InputTextMultiline(iid, mutable_copy.data(), mutable_copy.size() + 1,
                                  ImVec2(-FLT_MIN, body_h), ImGuiInputTextFlags_ReadOnly);
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();
        if (st.mono) ImGui::PopFont();
    }
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(2);
    ImGui::Spacing();
}

}  // namespace

void draw_markdown(std::string_view text, const MarkdownStyle &st, int entry_id, CopyFeedback &copied) {
    const std::vector<std::string_view> lines = split_lines(text);
    int block_idx = 0;
    size_t i = 0;
    while (i < lines.size()) {
        const std::string_view ln = lines[i];
        size_t indent = 0;
        while (indent < ln.size() && indent < 3 && ln[indent] == ' ') ++indent;
        const std::string_view rest = ln.substr(indent);

        // Blank line
        if (trim_sv(ln).empty()) {
            ImGui::Dummy(ImVec2(0.0f, 4.0f * st.scale));
            ++i;
            continue;
        }

        // Fenced code block: ```[lang]
        if (rest.size() >= 3 && rest.substr(0, 3) == "```") {
            const std::string lang = trim_sv(rest.substr(3));
            std::string code;
            ++i;
            while (i < lines.size()) {
                size_t ci = 0;
                while (ci < lines[i].size() && ci < 3 && lines[i][ci] == ' ') ++ci;
                if (lines[i].size() - ci >= 3 && lines[i].substr(ci, 3) == "```") {
                    ++i;
                    break;
                }
                if (!code.empty()) code.push_back('\n');
                code.append(lines[i].data(), lines[i].size());
                ++i;
            }
            draw_code_block(lang, code, st, entry_id, block_idx++, copied);
            continue;
        }

        // Heading: # ... ####
        if (!rest.empty() && rest[0] == '#') {
            size_t level = 0;
            while (level < rest.size() && level < 6 && rest[level] == '#') ++level;
            if (level < rest.size() && rest[level] == ' ') {
                const std::string_view heading = rest.substr(level + 1);
                ImGui::Spacing();
                ImFont *hfont = (level == 1 && st.large) ? st.large : st.bold;
                draw_inline_wrapped(heading, st, &st.accent, hfont);
                if (level <= 2) ImGui::Separator();
                ImGui::Spacing();
                ++i;
                continue;
            }
        }

        // Horizontal rule
        if (is_hr_line(rest)) {
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ++i;
            continue;
        }

        // Table: header line + separator line
        if (rest.find('|') != std::string_view::npos && i + 1 < lines.size() && is_table_sep(lines[i + 1])) {
            const std::vector<std::string> headers = split_table_row(ln);
            const int cols = static_cast<int>(headers.size());
            if (cols > 0 && cols <= 16) {
                std::vector<std::vector<std::string>> rows;
                i += 2;
                while (i < lines.size() && lines[i].find('|') != std::string_view::npos && !trim_sv(lines[i]).empty()) {
                    rows.push_back(split_table_row(lines[i]));
                    ++i;
                }
                char tid[48];
                std::snprintf(tid, sizeof(tid), "##tbl_%d_%d", entry_id, block_idx++);
                const ImGuiTableFlags tflags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                               ImGuiTableFlags_SizingStretchSame;
                if (ImGui::BeginTable(tid, cols, tflags)) {
                    for (int c = 0; c < cols; ++c) ImGui::TableSetupColumn(headers[static_cast<size_t>(c)].c_str());
                    ImGui::TableHeadersRow();
                    for (const auto &row : rows) {
                        ImGui::TableNextRow();
                        for (int c = 0; c < cols; ++c) {
                            ImGui::TableSetColumnIndex(c);
                            if (static_cast<size_t>(c) < row.size()) draw_inline_wrapped(row[static_cast<size_t>(c)], st);
                        }
                    }
                    ImGui::EndTable();
                }
                ImGui::Spacing();
                continue;
            }
        }

        // Blockquote
        if (!rest.empty() && rest[0] == '>') {
            std::string_view q = rest.substr(1);
            if (!q.empty() && q[0] == ' ') q.remove_prefix(1);
            const ImVec2 p0 = ImGui::GetCursorScreenPos();
            ImGui::Indent(14.0f * st.scale);
            draw_inline_wrapped(q, st, &st.dim);
            const float y1 = ImGui::GetCursorScreenPos().y;
            ImGui::Unindent(14.0f * st.scale);
            ImGui::GetWindowDrawList()->AddRectFilled(
                p0 + ImVec2(2.0f * st.scale, 1.0f * st.scale),
                ImVec2(p0.x + 5.0f * st.scale, std::max(y1 - 2.0f * st.scale, p0.y + ImGui::GetTextLineHeight())),
                ImGui::GetColorU32(st.quote_bar), 2.0f * st.scale);
            ++i;
            continue;
        }

        // Bullet list
        {
            size_t sp = 0;
            while (sp < ln.size() && ln[sp] == ' ') ++sp;
            if (sp + 1 < ln.size() && (ln[sp] == '-' || ln[sp] == '*') && ln[sp + 1] == ' ') {
                const float extra = (10.0f + static_cast<float>(sp) * 6.0f) * st.scale;
                ImGui::Indent(extra);
                ImGui::TextColored(st.accent, "•");
                ImGui::SameLine(0.0f, 6.0f * st.scale);
                draw_inline_wrapped(ln.substr(sp + 2), st);
                ImGui::Unindent(extra);
                ++i;
                continue;
            }
        }

        // Ordered list
        {
            size_t pref_end = 0;
            std::string marker;
            if (ordered_list_prefix(ln, pref_end, marker)) {
                ImGui::Indent(10.0f * st.scale);
                ImGui::TextColored(st.accent, "%s", marker.c_str());
                ImGui::SameLine(0.0f, 6.0f * st.scale);
                draw_inline_wrapped(ln.substr(pref_end), st);
                ImGui::Unindent(10.0f * st.scale);
                ++i;
                continue;
            }
        }

        // Regular paragraph line
        draw_inline_wrapped(ln, st);
        ++i;
    }
}
