// The window on Android: a NativeActivity (android/, docs/ANDROID.md) with
// EGL + OpenGL ES 3. The model is an uncompressed asset of the APK, read in
// place through its file descriptor and offset (desktop.h). Text is typed in
// the system's own text box (MainActivity.java), which keeps accents,
// autocorrect and paste; Dear ImGui only draws.
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <android/asset_manager.h>
#include <android/log.h>
#include <android_native_app_glue.h>
#include <jni.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "imgui.h"
#include "imgui_impl_android.h"
#include "imgui_impl_opengl3.h"
#include "session.h"
#include "view.h"

extern "C" {
#include "platform.h"
}

namespace {

constexpr const char *kTag = "smolmoe";
constexpr const char *kModelAsset = "model.smoe";
constexpr const char *kNameAsset = "model-name.txt";
constexpr int kIdleWaitMs = 500;      // the text cursor blinks even when nothing happens
constexpr int kFramesAfterEvent = 3;  // Dear ImGui settles hover and focus over a few frames
constexpr float kDensityBase = 160.0f;  // Android's 1x density (dpi)

// The system text box's answers, written on Java's UI thread.
std::mutex g_inbox_mutex;
std::vector<std::pair<std::string, bool>> g_inbox;
android_app *g_app = nullptr;
// The system bars' heights from MainActivity.onInsets, in pixels.
std::atomic<int> g_inset_top{0}, g_inset_bottom{0};

struct Display {
    EGLDisplay display = EGL_NO_DISPLAY;
    EGLConfig config = nullptr;
    EGLContext context = EGL_NO_CONTEXT;
    EGLSurface surface = EGL_NO_SURFACE;
};

struct Window {
    Display egl;
    ChatView *view = nullptr;
    JNIEnv *env = nullptr;
    bool imgui_ready = false;
    bool rebuild_look = true;  // density or night mode changed
    Look look;
    int frames = kFramesAfterEvent;
};

// ---- strings across JNI: real UTF-16, since NewStringUTF wants modified UTF-8 ----

std::u16string utf16(const std::string &s) {
    std::u16string out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        const size_t n = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        if (i + n > s.size()) break;
        uint32_t cp = n == 1 ? c : n == 2 ? c & 0x1Fu : n == 3 ? c & 0x0Fu : c & 0x07u;
        for (size_t k = 1; k < n; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3Fu);
        i += n;
        if (cp >= 0x10000) {
            cp -= 0x10000;
            out.push_back(static_cast<char16_t>(0xD800 + (cp >> 10)));
            out.push_back(static_cast<char16_t>(0xDC00 + (cp & 0x3FF)));
        } else {
            out.push_back(static_cast<char16_t>(cp));
        }
    }
    return out;
}

std::string utf8(const jchar *s, jsize n) {
    std::string out;
    for (jsize i = 0; i < n; ++i) {
        uint32_t cp = s[i];
        if (cp >= 0xD800 && cp < 0xDC00 && i + 1 < n && s[i + 1] >= 0xDC00 && s[i + 1] < 0xE000)
            cp = 0x10000 + ((cp - 0xD800) << 10) + (s[++i] - 0xDC00);
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
    return out;
}

// Calls MainActivity.<method>(String) on the activity.
void call_activity(JNIEnv *env, const char *method, const std::string &text) {
    if (!env || !g_app) return;
    jobject activity = g_app->activity->clazz;
    jclass cls = env->GetObjectClass(activity);
    jmethodID id = env->GetMethodID(cls, method, "(Ljava/lang/String;)V");
    if (id) {
        const std::u16string wide = utf16(text);
        jstring arg = env->NewString(reinterpret_cast<const jchar *>(wide.data()), static_cast<jsize>(wide.size()));
        env->CallVoidMethod(activity, id, arg);
        env->DeleteLocalRef(arg);
    }
    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }
    env->DeleteLocalRef(cls);
}

// ---- the APK's assets ----------------------------------------------------------

std::string read_asset_text(AAssetManager *assets, const char *name) {
    AAsset *asset = AAssetManager_open(assets, name, AASSET_MODE_BUFFER);
    if (!asset) return {};
    const void *data = AAsset_getBuffer(asset);
    std::string text = data ? std::string(static_cast<const char *>(data), static_cast<size_t>(AAsset_getLength(asset))) : "";
    AAsset_close(asset);
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) text.pop_back();
    return text;
}

// The model asset as a file the runtime can map: /proc/self/fd/N, with the
// asset's offset and length inside the APK. The descriptor stays open for
// the life of the process. False when the asset is missing or compressed.
bool model_asset(AAssetManager *assets, std::string *path, uint64_t *offset, uint64_t *length) {
    AAsset *asset = AAssetManager_open(assets, kModelAsset, AASSET_MODE_RANDOM);
    if (!asset) return false;
    off64_t start = 0, size = 0;
    const int fd = AAsset_openFileDescriptor64(asset, &start, &size);
    AAsset_close(asset);
    if (fd < 0) return false;
    *path = "/proc/self/fd/" + std::to_string(fd);
    *offset = static_cast<uint64_t>(start);
    *length = static_cast<uint64_t>(size);
    return true;
}

// ---- the processor ---------------------------------------------------------------

// Phones mix fast and slow cores (4 Cortex-A78 and 4 A55 on a Dimensity
// 7300): the work is split evenly, so the slow ones would hold every step
// back. Counts the cores that reach the highest maximum frequency; 0 when
// cpufreq cannot be read (the runtime then picks).
uint32_t fast_cores() {
    const uint32_t n = cpu_logical_count();
    std::vector<long> freq;
    for (uint32_t i = 0; i < n; ++i) {
        char path[96];
        std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u/cpufreq/cpuinfo_max_freq", i);
        long khz = 0;
        if (FILE *f = std::fopen(path, "r")) {
            if (std::fscanf(f, "%ld", &khz) != 1) khz = 0;
            std::fclose(f);
        }
        freq.push_back(khz);
    }
    const long top = freq.empty() ? 0 : *std::max_element(freq.begin(), freq.end());
    if (top <= 0) return 0;
    return static_cast<uint32_t>(std::count(freq.begin(), freq.end(), top));
}

// ---- EGL ---------------------------------------------------------------------------

bool egl_open(Display *d, ANativeWindow *native) {
    if (d->display == EGL_NO_DISPLAY) {
        d->display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (d->display == EGL_NO_DISPLAY || !eglInitialize(d->display, nullptr, nullptr)) return false;
        const EGLint attributes[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
                                     EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_NONE};
        EGLint count = 0;
        if (!eglChooseConfig(d->display, attributes, &d->config, 1, &count) || count < 1) return false;
        const EGLint context_attributes[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
        d->context = eglCreateContext(d->display, d->config, EGL_NO_CONTEXT, context_attributes);
        if (d->context == EGL_NO_CONTEXT) return false;
    }
    d->surface = eglCreateWindowSurface(d->display, d->config, native, nullptr);
    if (d->surface == EGL_NO_SURFACE) return false;
    return eglMakeCurrent(d->display, d->surface, d->surface, d->context) == EGL_TRUE;
}

// The window went away (the app left the screen); the context and the model stay.
void egl_drop_surface(Display *d) {
    if (d->surface == EGL_NO_SURFACE) return;
    eglMakeCurrent(d->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroySurface(d->display, d->surface);
    d->surface = EGL_NO_SURFACE;
}

void egl_close(Display *d) {
    egl_drop_surface(d);
    if (d->context != EGL_NO_CONTEXT) eglDestroyContext(d->display, d->context);
    if (d->display != EGL_NO_DISPLAY) eglTerminate(d->display);
    *d = Display{};
}

// ---- looks ---------------------------------------------------------------------------

Look make_look(android_app *app, bool renderer_ready) {
    const int32_t density = AConfiguration_getDensity(app->config);
    const bool night = AConfiguration_getUiModeNight(app->config) == ACONFIGURATION_UI_MODE_NIGHT_YES;
    Look look;
    look.light = theme_is_light(!night);
    look.scale = density > 0 ? static_cast<float>(density) / kDensityBase : 2.0f;
    look.large = load_fonts(look.scale, {"/system/fonts/RobotoStatic-Regular.ttf", "/system/fonts/Roboto-Regular.ttf",
                                         "/system/fonts/NotoSans-Regular.ttf", "/system/fonts/DroidSans.ttf"});
    apply_style(look);
    if (renderer_ready) ImGui_ImplOpenGL3_DestroyFontsTexture();
    return look;
}

void set_clipboard(ImGuiContext *, const char *text) {
    JNIEnv *env = nullptr;
    if (g_app && g_app->activity->vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) == JNI_OK)
        call_activity(env, "copyText", text ? text : "");
}

// ---- the activity's events ---------------------------------------------------------

void on_command(android_app *app, int32_t command) {
    Window *w = static_cast<Window *>(app->userData);
    switch (command) {
    case APP_CMD_INIT_WINDOW:
        if (!app->window) break;
        if (!egl_open(&w->egl, app->window)) {
            __android_log_print(ANDROID_LOG_ERROR, kTag, "EGL: não consegui criar a superfície (0x%x)", eglGetError());
            break;
        }
        ImGui_ImplAndroid_Init(app->window);
        if (!w->imgui_ready) {
            ImGui_ImplOpenGL3_Init("#version 300 es");
            w->imgui_ready = true;
        }
        w->frames = kFramesAfterEvent;
        break;
    case APP_CMD_TERM_WINDOW:
        if (w->imgui_ready) ImGui_ImplAndroid_Shutdown();
        egl_drop_surface(&w->egl);
        break;
    case APP_CMD_CONTENT_RECT_CHANGED:  // read in draw_frame
        w->frames = kFramesAfterEvent;
        break;
    case APP_CMD_CONFIG_CHANGED:
        w->rebuild_look = true;
        w->frames = kFramesAfterEvent;
        break;
    default:
        break;
    }
}

int32_t on_input(android_app *app, AInputEvent *event) {
    Window *w = static_cast<Window *>(app->userData);
    w->frames = kFramesAfterEvent;
    return ImGui_ImplAndroid_HandleInputEvent(event);
}

void draw_frame(Window *w, android_app *app) {
    if (w->rebuild_look) {
        w->look = make_look(app, true);
        w->rebuild_look = false;
    }
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplAndroid_NewFrame();
    ImGui::NewFrame();
    // NativeActivity draws behind the status and navigation bars. contentRect
    // should be the part they leave free, but some phones report the whole
    // screen there, so the window insets from Java win when they are larger.
    const ARect &free = app->contentRect;
    const float height = ImGui::GetIO().DisplaySize.y;
    float top = 0.0f, bottom = 0.0f;
    if (free.bottom > free.top && free.bottom <= height) {
        top = static_cast<float>(free.top);
        bottom = height - static_cast<float>(free.bottom);
    }
    w->look.inset_top = std::max(top, static_cast<float>(g_inset_top.load()));
    w->look.inset_bottom = std::max(bottom, static_cast<float>(g_inset_bottom.load()));
    w->view->draw(w->look);
    ImGui::Render();
    const ImGuiIO &io = ImGui::GetIO();
    const ImVec4 bg = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];
    glViewport(0, 0, static_cast<int>(io.DisplaySize.x), static_cast<int>(io.DisplaySize.y));
    glClearColor(bg.x, bg.y, bg.z, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    eglSwapBuffers(w->egl.display, w->egl.surface);
}

void on_window_die(const char *message) { __android_log_print(ANDROID_LOG_FATAL, kTag, "%s", message ? message : ""); }

}  // namespace

// MainActivity.onText: the system text box closed; send is true for "Enviar".
extern "C" JNIEXPORT void JNICALL Java_io_smolmoe_app_MainActivity_onText(JNIEnv *env, jclass, jstring text,
                                                                         jboolean send) {
    const jsize n = env->GetStringLength(text);
    const jchar *chars = env->GetStringChars(text, nullptr);
    std::string message = chars ? utf8(chars, n) : std::string();
    if (chars) env->ReleaseStringChars(text, chars);
    {
        std::lock_guard<std::mutex> lock(g_inbox_mutex);
        g_inbox.emplace_back(std::move(message), send == JNI_TRUE);
    }
    if (g_app) ALooper_wake(g_app->looper);
}

// MainActivity.onInsets: the status and navigation bars changed.
extern "C" JNIEXPORT void JNICALL Java_io_smolmoe_app_MainActivity_onInsets(JNIEnv *, jclass, jint top, jint bottom) {
    g_inset_top.store(std::max(0, static_cast<int>(top)));
    g_inset_bottom.store(std::max(0, static_cast<int>(bottom)));
    if (g_app) ALooper_wake(g_app->looper);
}

void android_main(android_app *app) {
    g_app = app;
    Window w;
    app->userData = &w;
    app->onAppCmd = on_command;
    app->onInputEvent = on_input;
    app->activity->vm->AttachCurrentThread(&w.env, nullptr);
    ChatSession::install_die_handler(on_window_die);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.IniFilename = nullptr;  // nothing written on the phone
    ImGui::GetPlatformIO().Platform_SetClipboardTextFn = set_clipboard;

    SessionOptions options;
    options.threads = fast_cores();
    options.name = read_asset_text(app->activity->assetManager, kNameAsset);
    ChatSession session(options, [] {
        if (g_app) ALooper_wake(g_app->looper);
    });
    Platform platform;
    platform.touch = true;
    platform.edit_text = [&w](const std::string &draft) { call_activity(w.env, "editText", draft); };
    ChatView view(session, platform);
    w.view = &view;

    std::string path;
    uint64_t offset = 0, length = 0;
    if (model_asset(app->activity->assetManager, &path, &offset, &length)) {
        session.open(path, offset, length);
    } else {
        __android_log_print(ANDROID_LOG_ERROR, kTag, "o APK não traz %s sem compressão", kModelAsset);
        session.open("modelo-ausente-no-apk.smoe");  // refused, with the reason on screen
    }

    bool finishing = false;
    while (!app->destroyRequested) {
        const bool drawing = w.egl.surface != EGL_NO_SURFACE;
        const int timeout = drawing && (view.animating() || w.frames > 0) ? 0 : kIdleWaitMs;
        android_poll_source *source = nullptr;
        int events = 0;
        const int result = ALooper_pollOnce(timeout, nullptr, &events, reinterpret_cast<void **>(&source));
        if (result == ALOOPER_POLL_ERROR) break;
        if (source) source->process(app, source);
        if (result == ALOOPER_POLL_TIMEOUT || result == ALOOPER_POLL_WAKE) w.frames = std::max(w.frames, 1);
        {
            std::vector<std::pair<std::string, bool>> inbox;
            {
                std::lock_guard<std::mutex> lock(g_inbox_mutex);
                inbox.swap(g_inbox);
            }
            for (const auto &[text, send] : inbox) view.edited(text, send);
            if (!inbox.empty()) w.frames = kFramesAfterEvent;
        }
        if (view.close_requested() && !finishing) {
            ANativeActivity_finish(app->activity);  // destroyRequested follows
            finishing = true;
        }
        if (w.egl.surface == EGL_NO_SURFACE || !w.imgui_ready) continue;
        draw_frame(&w, app);
        w.frames = std::max(w.frames - 1, 0);
    }

    w.view = nullptr;
    session.shutdown();
    if (w.imgui_ready) {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplAndroid_Shutdown();
    }
    ImGui::DestroyContext();
    egl_close(&w.egl);
    app->activity->vm->DetachCurrentThread();
    g_app = nullptr;
}
