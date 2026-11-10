// The desktop window on Windows: Win32 + Direct3D 11, the system's "open
// file" dialog, files dropped on the window, the light or dark theme the user
// chose for apps, and per-monitor DPI.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef UNICODE
#define UNICODE  // IDC_ARROW and friends as wide strings, like every call here
#endif
#include <windows.h>
#include <commdlg.h>
#include <d3d11.h>
#include <dwmapi.h>
#include <shellapi.h>
#include <io.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include "args.h"
#include "imgui.h"
#include "imgui_impl_dx11.h"
#include "imgui_impl_win32.h"
#include "session.h"
#include "view.h"

extern "C" {
#include "platform.h"
}

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam);

namespace {

constexpr UINT kWakeMessage = WM_APP + 1;
constexpr DWORD kIdleWaitMs = 500;   // the text cursor blinks even when nothing happens
constexpr int kFramesAfterEvent = 3; // Dear ImGui settles hover and focus over a few frames
constexpr DWORD kOccludedWaitMs = 100;
constexpr int kWindowWidth = 960, kWindowHeight = 720;
constexpr DWORD kDarkTitleBar = 20;  // DWMWA_USE_IMMERSIVE_DARK_MODE

ID3D11Device *g_device = nullptr;
ID3D11DeviceContext *g_context = nullptr;
IDXGISwapChain *g_swap_chain = nullptr;
ID3D11RenderTargetView *g_target = nullptr;
UINT g_resize_width = 0, g_resize_height = 0;
float g_pending_scale = 0;  // set by WM_DPICHANGED
ChatView *g_view = nullptr;
// SMOLMOE_GUI_SCREENSHOT=file.bmp: F12 saves the next frame there, read back
// from Direct3D, so it works even while the window is covered or the screen
// locked (pictures for the docs, checks by hand).
const char *g_screenshot_path = nullptr;
int g_capture_frames = 0;  // frames to draw before saving: layout settles over a few

std::string narrow(const wchar_t *w) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1) return {};
    std::string s(static_cast<size_t>(n - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, &s[0], n, nullptr, nullptr);
    return s;
}

std::wstring widen(const std::string &s) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if (n <= 1) return {};
    std::wstring w(static_cast<size_t>(n - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    return w;
}

// die() on the window's thread: it already printed the message, which is
// enough when stderr goes somewhere (a terminal redirect, a test); started
// from Explorer there is none, so a message box says it.
void show_error(const char *message) {
    if (_get_osfhandle(_fileno(stderr)) >= 0) return;  // where die() wrote it
    MessageBoxW(nullptr, widen(std::string("smolmoe: ") + message).c_str(), L"smolmoe", MB_ICONERROR | MB_OK);
}

// --help and --version from a terminal: a GUI program has no console of its own.
void print(const char *text) {
    if (_get_osfhandle(_fileno(stdout)) < 0 && AttachConsole(ATTACH_PARENT_PROCESS)) std::freopen("CONOUT$", "w", stdout);
    std::fputs(text, stdout);
    std::fflush(stdout);
}

bool light_theme() {
    DWORD value = 0, size = sizeof(value);
    const LSTATUS status =
        RegGetValueW(HKEY_CURRENT_USER, L"Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize",
                     L"AppsUseLightTheme", RRF_RT_REG_DWORD, nullptr, &value, &size);
    return status == ERROR_SUCCESS && value == 1;
}

std::vector<std::string> font_candidates() {
    wchar_t dir[MAX_PATH];
    const UINT n = GetWindowsDirectoryW(dir, MAX_PATH);
    if (!n || n >= MAX_PATH) return {};
    const std::string fonts = narrow(dir) + "\\Fonts\\";
    return {fonts + "segoeui.ttf", fonts + "arial.ttf", fonts + "tahoma.ttf"};
}

std::optional<std::string> pick_file(HWND owner) {
    std::vector<wchar_t> path(32768, L'\0');
    OPENFILENAMEW dialog{};
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = owner;
    dialog.lpstrFilter = L"Modelos do smolmoe (*.smoe)\0*.smoe\0Todos os arquivos\0*.*\0";
    dialog.lpstrFile = path.data();
    dialog.nMaxFile = static_cast<DWORD>(path.size());
    dialog.lpstrTitle = L"Abrir modelo";
    dialog.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR | OFN_EXPLORER;
    if (!GetOpenFileNameW(&dialog)) return std::nullopt;
    return narrow(path.data());
}

// The back buffer (RGBA) as a top-down 32-bit BMP.
void save_screenshot(const char *path) {
    ID3D11Texture2D *back = nullptr, *staging = nullptr;
    if (FAILED(g_swap_chain->GetBuffer(0, IID_PPV_ARGS(&back))) || !back) return;
    D3D11_TEXTURE2D_DESC desc;
    back->GetDesc(&desc);
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = 0;
    D3D11_MAPPED_SUBRESOURCE mapped;
    if (SUCCEEDED(g_device->CreateTexture2D(&desc, nullptr, &staging))) {
        g_context->CopyResource(staging, back);
        if (SUCCEEDED(g_context->Map(staging, 0, D3D11_MAP_READ, 0, &mapped))) {
            const DWORD row = desc.Width * 4, pixels = row * desc.Height;
            BITMAPFILEHEADER file{};
            BITMAPINFOHEADER info{};
            file.bfType = 0x4D42;  // "BM"
            file.bfOffBits = sizeof(file) + sizeof(info);
            file.bfSize = file.bfOffBits + pixels;
            info.biSize = sizeof(info);
            info.biWidth = static_cast<LONG>(desc.Width);
            info.biHeight = -static_cast<LONG>(desc.Height);
            info.biPlanes = 1;
            info.biBitCount = 32;
            info.biCompression = BI_RGB;
            if (FILE *f = open_file(path, "wb")) {
                std::fwrite(&file, sizeof(file), 1, f);
                std::fwrite(&info, sizeof(info), 1, f);
                std::vector<unsigned char> line(row);
                for (UINT y = 0; y < desc.Height; ++y) {
                    const unsigned char *src = static_cast<const unsigned char *>(mapped.pData) + y * mapped.RowPitch;
                    for (UINT x = 0; x < row; x += 4) {
                        line[x] = src[x + 2];
                        line[x + 1] = src[x + 1];
                        line[x + 2] = src[x];
                        line[x + 3] = 255;
                    }
                    std::fwrite(line.data(), 1, row, f);
                }
                std::fclose(f);
            }
            g_context->Unmap(staging, 0);
        }
        staging->Release();
    }
    back->Release();
}

void create_target() {
    ID3D11Texture2D *back = nullptr;
    if (FAILED(g_swap_chain->GetBuffer(0, IID_PPV_ARGS(&back))) || !back) return;
    g_device->CreateRenderTargetView(back, nullptr, &g_target);
    back->Release();
}

void release_target() {
    if (g_target) g_target->Release();
    g_target = nullptr;
}

void release_device() {
    release_target();
    if (g_swap_chain) g_swap_chain->Release();
    if (g_context) g_context->Release();
    if (g_device) g_device->Release();
    g_swap_chain = nullptr;
    g_context = nullptr;
    g_device = nullptr;
}

bool create_device(HWND hwnd) {
    DXGI_SWAP_CHAIN_DESC desc{};
    desc.BufferCount = 2;
    desc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.BufferDesc.RefreshRate.Numerator = 60;
    desc.BufferDesc.RefreshRate.Denominator = 1;
    desc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.OutputWindow = hwnd;
    desc.SampleDesc.Count = 1;
    desc.Windowed = TRUE;
    desc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0};
    D3D_FEATURE_LEVEL level;
    HRESULT hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, levels, 2,
                                               D3D11_SDK_VERSION, &desc, &g_swap_chain, &g_device, &level, &g_context);
    if (hr == DXGI_ERROR_UNSUPPORTED)  // no GPU driver (a VM, a remote desktop): software rendering
        hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, levels, 2, D3D11_SDK_VERSION,
                                           &desc, &g_swap_chain, &g_device, &level, &g_context);
    if (FAILED(hr)) return false;
    create_target();
    return true;
}

LRESULT WINAPI window_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wparam, lparam)) return 1;
    switch (msg) {
    case WM_SIZE:
        if (wparam != SIZE_MINIMIZED) {
            g_resize_width = LOWORD(lparam);
            g_resize_height = HIWORD(lparam);
        }
        return 0;
    case WM_DPICHANGED: {
        const RECT *suggested = reinterpret_cast<const RECT *>(lparam);
        g_pending_scale = HIWORD(wparam) / 96.0f;
        SetWindowPos(hwnd, nullptr, suggested->left, suggested->top, suggested->right - suggested->left,
                     suggested->bottom - suggested->top, SWP_NOZORDER | SWP_NOACTIVATE);
        return 0;
    }
    case WM_DROPFILES: {
        HDROP drop = reinterpret_cast<HDROP>(wparam);
        const UINT n = DragQueryFileW(drop, 0, nullptr, 0);
        std::wstring path(n + 1, L'\0');
        if (n && DragQueryFileW(drop, 0, &path[0], n + 1) && g_view) g_view->drop(narrow(path.c_str()));
        DragFinish(drop);
        return 0;
    }
    case WM_KEYDOWN:
        if (wparam == VK_F12 && g_screenshot_path) g_capture_frames = kFramesAfterEvent;
        break;
    case WM_SYSCOMMAND:
        if ((wparam & 0xfff0) == SC_KEYMENU) return 0;  // Alt does not open a window menu
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    default:
        break;
    }
    return DefWindowProcW(hwnd, msg, wparam, lparam);
}

// Fonts and style for a DPI scale; the renderer rebuilds its font texture.
Look make_look(float scale, bool light, bool renderer_ready) {
    Look look;
    look.scale = scale;
    look.light = light;
    look.large = load_fonts(scale, font_candidates());
    apply_style(look);
    if (renderer_ready) ImGui_ImplDX11_InvalidateDeviceObjects();
    return look;
}

}  // namespace

int main(int argc, char **argv) {
    platform_init(&argc, &argv);
    ChatSession::install_die_handler(show_error);
    g_screenshot_path = std::getenv("SMOLMOE_GUI_SCREENSHOT");
    const GuiArgs args = parse_gui_args(argc, argv);
    if (args.help || args.version) {
        print(args.help ? gui_usage() : gui_version());
        return 0;
    }

    ImGui_ImplWin32_EnableDpiAwareness();
    const float scale = ImGui_ImplWin32_GetDpiScaleForMonitor(MonitorFromPoint(POINT{0, 0}, MONITOR_DEFAULTTOPRIMARY));
    WNDCLASSEXW window_class{};
    window_class.cbSize = sizeof(window_class);
    window_class.style = CS_CLASSDC;
    window_class.lpfnWndProc = window_proc;
    window_class.hInstance = GetModuleHandleW(nullptr);
    window_class.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    window_class.hIcon = LoadIconW(nullptr, IDI_APPLICATION);
    window_class.lpszClassName = L"smolmoe";
    RegisterClassExW(&window_class);
    HWND hwnd = CreateWindowW(window_class.lpszClassName, L"smolmoe", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                              static_cast<int>(kWindowWidth * scale), static_cast<int>(kWindowHeight * scale), nullptr,
                              nullptr, window_class.hInstance, nullptr);
    if (!hwnd || !create_device(hwnd)) {
        show_error("não consegui criar a janela com Direct3D 11");
        release_device();
        return 1;
    }
    const bool light = theme_is_light(light_theme());
    const BOOL dark = light ? FALSE : TRUE;
    DwmSetWindowAttribute(hwnd, kDarkTitleBar, &dark, sizeof(dark));
    DragAcceptFiles(hwnd, TRUE);
    ShowWindow(hwnd, SW_SHOWDEFAULT);
    UpdateWindow(hwnd);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.IniFilename = nullptr;  // nothing written next to the program
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    Look look = make_look(ImGui_ImplWin32_GetDpiScaleForHwnd(hwnd), light, false);
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_device, g_context);

    ChatSession session(args.session, [hwnd] { PostMessageW(hwnd, kWakeMessage, 0, 0); });
    Platform platform;
    platform.pick_file = [hwnd] { return pick_file(hwnd); };
    platform.set_title = [hwnd](const std::string &title) { SetWindowTextW(hwnd, widen(title).c_str()); };
    ChatView view(session, platform);
    g_view = &view;
    if (!args.model.empty()) session.open(args.model);
    else if (desktop_has_bundle()) session.open("");

    bool done = false, occluded = false;
    int frames = kFramesAfterEvent;
    while (!done) {
        if (!view.animating() && frames <= 0) MsgWaitForMultipleObjects(0, nullptr, FALSE, kIdleWaitMs, QS_ALLINPUT);
        MSG msg;
        bool events = false;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
            done |= msg.message == WM_QUIT;
            events = true;
        }
        if (done) break;
        frames = events ? kFramesAfterEvent : std::max(frames, 1);
        if (occluded && !g_capture_frames && g_swap_chain->Present(0, DXGI_PRESENT_TEST) == DXGI_STATUS_OCCLUDED) {
            MsgWaitForMultipleObjects(0, nullptr, FALSE, kOccludedWaitMs, QS_ALLINPUT);
            continue;
        }
        occluded = false;
        if (g_resize_width && g_resize_height) {
            release_target();
            g_swap_chain->ResizeBuffers(0, g_resize_width, g_resize_height, DXGI_FORMAT_UNKNOWN, 0);
            g_resize_width = g_resize_height = 0;
            create_target();
        }
        if (!g_target) {  // a lost device: nothing to draw into until it comes back
            create_target();
            if (!g_target) {
                MsgWaitForMultipleObjects(0, nullptr, FALSE, kOccludedWaitMs, QS_ALLINPUT);
                continue;
            }
        }
        if (g_pending_scale > 0) {
            look = make_look(g_pending_scale, light, true);
            g_pending_scale = 0;
        }

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();
        view.draw(look);
        ImGui::Render();
        const ImVec4 bg = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];
        const float clear[4] = {bg.x, bg.y, bg.z, 1.0f};
        g_context->OMSetRenderTargets(1, &g_target, nullptr);
        g_context->ClearRenderTargetView(g_target, clear);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        if (g_capture_frames && --g_capture_frames == 0) save_screenshot(g_screenshot_path);
        occluded = g_swap_chain->Present(1, 0) == DXGI_STATUS_OCCLUDED;
        --frames;
        if (view.close_requested()) DestroyWindow(hwnd);
    }

    g_view = nullptr;
    session.shutdown();
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    release_device();
    DestroyWindow(hwnd);
    UnregisterClassW(window_class.lpszClassName, window_class.hInstance);
    return 0;
}
