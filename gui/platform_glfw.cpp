// The desktop window on Linux and macOS: GLFW + OpenGL 3, files dropped on
// the window, the monitor's content scale. There is no native file dialog in
// GLFW, so the first screen takes a path.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define GL_SILENCE_DEPRECATION  // macOS: OpenGL 3.2 still works, with warnings
#include <GLFW/glfw3.h>      // also the system OpenGL header, for glClear and glViewport

#include "args.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "session.h"
#include "view.h"

extern "C" {
#include "platform.h"
}

namespace {

constexpr double kIdleWaitSeconds = 0.5;  // the text cursor blinks even when nothing happens
constexpr int kFramesAfterEvent = 3;      // Dear ImGui settles hover and focus over a few frames
constexpr int kWindowWidth = 960, kWindowHeight = 720;

ChatView *g_view = nullptr;
float g_pending_scale = 0;  // set when the window moves to a monitor with another scale

std::vector<std::string> font_candidates() {
#if defined(__APPLE__)
    return {"/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Verdana.ttf"};
#else
    return {"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/noto/NotoSans-Regular.ttf", "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"};
#endif
}

// macOS lays the window out in points and renders retina pixels: fonts are
// rasterized at the pixel scale and drawn back at 1/scale. Elsewhere the
// whole interface grows with the scale.
Look make_look(float content_scale, bool renderer_ready) {
    Look look;
    look.light = theme_is_light(false);
#if defined(__APPLE__)
    look.scale = 1.0f;
    look.large = load_fonts(content_scale, font_candidates());
    ImGui::GetIO().FontGlobalScale = 1.0f / content_scale;
#else
    look.scale = content_scale;
    look.large = load_fonts(content_scale, font_candidates());
#endif
    apply_style(look);
    if (renderer_ready) ImGui_ImplOpenGL3_DestroyFontsTexture();
    return look;
}

void on_glfw_error(int code, const char *description) { std::fprintf(stderr, "smolmoe-gui: GLFW %d: %s\n", code, description); }

}  // namespace

int main(int argc, char **argv) {
    platform_init(&argc, &argv);
    ChatSession::install_die_handler(nullptr);  // die() already printed the message
    const GuiArgs args = parse_gui_args(argc, argv);
    if (args.help || args.version) {
        std::fputs(args.help ? gui_usage() : gui_version(), stdout);
        return 0;
    }

    glfwSetErrorCallback(on_glfw_error);
    if (!glfwInit()) {
        std::fputs("smolmoe-gui: não consegui abrir uma janela (sem tela gráfica?)\n", stderr);
        return 1;
    }
#if defined(__APPLE__)
    const char *glsl = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#else
    const char *glsl = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
#endif
    GLFWwindow *window = glfwCreateWindow(kWindowWidth, kWindowHeight, "smolmoe", nullptr, nullptr);
    if (!window) {
        std::fputs("smolmoe-gui: não consegui criar a janela com OpenGL 3\n", stderr);
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetDropCallback(window, [](GLFWwindow *, int count, const char **paths) {
        if (count > 0 && g_view) g_view->drop(paths[0]);
    });
    glfwSetWindowContentScaleCallback(window, [](GLFWwindow *, float x, float) { g_pending_scale = x; });

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.IniFilename = nullptr;  // nothing written next to the program
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    float scale_x = 1, scale_y = 1;
    glfwGetWindowContentScale(window, &scale_x, &scale_y);
    Look look = make_look(scale_x > 0 ? scale_x : 1.0f, false);
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl);

    ChatSession session(args.session, [] { glfwPostEmptyEvent(); });
    Platform platform;
    platform.set_title = [window](const std::string &title) { glfwSetWindowTitle(window, title.c_str()); };
    ChatView view(session, platform);
    g_view = &view;
    if (!args.model.empty()) session.open(args.model);
    else if (desktop_has_bundle()) session.open("");

    int frames = kFramesAfterEvent;
    while (!glfwWindowShouldClose(window) && !view.close_requested()) {
        if (view.animating() || frames > 0) {
            glfwPollEvents();
        } else {
            glfwWaitEventsTimeout(kIdleWaitSeconds);
            frames = kFramesAfterEvent;
        }
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED)) {
            glfwWaitEventsTimeout(kIdleWaitSeconds);
            continue;
        }
        if (g_pending_scale > 0) {
            look = make_look(g_pending_scale, true);
            g_pending_scale = 0;
        }
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        view.draw(look);
        ImGui::Render();
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        const ImVec4 bg = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];
        glViewport(0, 0, width, height);
        glClearColor(bg.x, bg.y, bg.z, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
        frames = std::max(frames - 1, 0);
    }

    g_view = nullptr;
    const bool clean = session.shutdown();
    // A worker still loading, or parked by an error, may still call the window
    // (its wake-up): end the process before tearing anything down.
    if (!clean) std::_Exit(EXIT_SUCCESS);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
