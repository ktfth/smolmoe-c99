#include "session.h"

#include <chrono>
#include <cstdlib>
#include <utility>

extern "C" {
#include "platform.h"
#include "util.h"
}

namespace {

constexpr auto kShutdownWait = std::chrono::seconds(3);
constexpr size_t kNoticeBytes = 1100;  // engine notices are at most 1024 bytes

// die() reports to the one session; a die() on the window's own thread
// (before the window exists, or while reading the command line) is shown by
// the platform layer instead.
ChatSession *g_session = nullptr;
std::thread::id g_window_thread;
void (*g_window_die)(const char *message) = nullptr;

// "qwen2.5-0.5b-instruct-q8" for .../qwen2.5-0.5b-instruct-q8.smoe; the folder's
// name for models/<name>/model.smoe, whose file name says nothing.
std::string display_name(const std::string &path) {
    const char *self = path.empty() ? self_executable_path() : nullptr;
    const std::string full = path.empty() ? (self ? self : "smolmoe") : path;
    const size_t slash = full.find_last_of("/\\");
    const std::string file = slash == std::string::npos ? full : full.substr(slash + 1);
    const size_t dot = file.rfind('.');
    const std::string stem = dot == std::string::npos || dot == 0 ? file : file.substr(0, dot);
    if (slash == std::string::npos || (stem != "model" && stem.rfind("model-", 0) != 0)) return stem;
    const std::string folder = full.substr(0, slash);
    const size_t parent = folder.find_last_of("/\\");
    const std::string name = parent == std::string::npos ? folder : folder.substr(parent + 1);
    return name.empty() ? stem : name + stem.substr(5);
}

}  // namespace

size_t complete_utf8(const std::string &s) {
    const size_t n = s.size();
    size_t back = 0;
    while (back < 4 && back < n && (static_cast<unsigned char>(s[n - 1 - back]) & 0xC0) == 0x80) ++back;
    if (back == n) return n;
    const unsigned char lead = static_cast<unsigned char>(s[n - 1 - back]);
    const size_t need = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
    return back + 1 < need ? n - 1 - back : n;
}

ChatSession::ChatSession(SessionOptions options, std::function<void()> wake)
    : options_(options), wake_(std::move(wake)) {
    g_session = this;
}

ChatSession::~ChatSession() {
    if (worker_.joinable() && !shutdown()) std::_Exit(EXIT_SUCCESS);  // the worker uses this object
    if (g_session == this) g_session = nullptr;
}

void ChatSession::install_die_handler(void (*on_window_thread)(const char *message)) {
    g_window_thread = std::this_thread::get_id();
    g_window_die = on_window_thread;
    die_set_handler(&ChatSession::on_die);
}

void ChatSession::on_die(const char *message) {
    ChatSession *session = g_session;
    if (!session || std::this_thread::get_id() == g_window_thread) {
        if (g_window_die) g_window_die(message);
        return;  // die() exits
    }
    {
        std::lock_guard<std::mutex> lock(session->mutex_);
        session->state_.phase = Phase::Failed;
        session->state_.error = message;
        session->changed();
    }
    session->worker_done_.notify_all();
    for (;;) std::this_thread::sleep_for(std::chrono::hours(1));  // die() never returns
}

void ChatSession::changed() {
    ++state_.version;
    if (wake_) wake_();
}

void ChatSession::open(const std::string &path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_.phase != Phase::NoModel || worker_running_) return;
    if (worker_.joinable()) worker_.join();  // the thread of a refused file, already finished
    state_.phase = Phase::Opening;
    state_.stage = DESKTOP_CHECKING;
    state_.error.clear();
    state_.model_name = display_name(path);
    state_.phase_started = now_seconds();
    worker_running_ = true;
    changed();
    worker_ = std::thread(&ChatSession::run, this, path);
}

bool ChatSession::send(const std::string &text) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_.phase != Phase::Ready || text.empty()) return false;
    state_.entries.push_back({Role::User, text});
    state_.phase = Phase::Busy;
    state_.phase_started = now_seconds();
    state_.last_stopped = false;
    stop_ = false;
    requests_.push_back({Request::Kind::Send, text});
    changed();
    requests_ready_.notify_one();
    return true;
}

void ChatSession::stop() { stop_ = true; }

void ChatSession::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_.phase != Phase::Ready) return;
    state_.entries.clear();
    state_.last_rate = 0;
    state_.last_stopped = false;
    requests_.push_back({Request::Kind::Reset, {}});
    changed();
    requests_ready_.notify_one();
}

std::optional<SessionState> ChatSession::changed_since(uint64_t seen) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_.version == seen) return std::nullopt;
    return state_;
}

bool ChatSession::shutdown() {
    std::unique_lock<std::mutex> lock(mutex_);
    quit_ = true;
    stop_ = true;
    requests_ready_.notify_all();
    worker_done_.wait_for(lock, kShutdownWait, [this] { return !worker_running_ || state_.phase == Phase::Failed; });
    const bool finished = !worker_running_ && state_.phase != Phase::Failed;
    lock.unlock();
    if (!finished) return false;
    if (worker_.joinable()) worker_.join();
    return true;
}

void ChatSession::run(std::string path) {
    DesktopOptions o{};
    char err[DESKTOP_ERROR_BYTES];
    o.path = path.empty() ? nullptr : path.c_str();
    o.context = options_.context;
    o.threads = options_.threads;
    o.sink = EngineSink{this, on_text, on_reply_end, on_notice, on_stop};
    o.stage = on_stage;
    Desktop *desktop = desktop_open(&o, err);
    if (!desktop) {
        // A refused file: back to the first screen, ready for another one at once.
        std::lock_guard<std::mutex> lock(mutex_);
        state_.phase = Phase::NoModel;
        state_.error = err;
        state_.model_name.clear();
        state_.phase_started = now_seconds();
        worker_running_ = false;
        changed();
        worker_done_.notify_all();
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        state_.phase = Phase::Ready;
        desktop_info(desktop, &state_.info);
        state_.phase_started = now_seconds();
        changed();
    }
    serve(desktop);
    desktop_close(desktop);
    std::lock_guard<std::mutex> lock(mutex_);
    worker_running_ = false;
    worker_done_.notify_all();
}

void ChatSession::serve(Desktop *desktop) {
    for (;;) {
        Request request;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            requests_ready_.wait(lock, [this] { return quit_ || !requests_.empty(); });
            if (quit_) return;
            request = std::move(requests_.front());
            requests_.pop_front();
        }
        DesktopInfo before{}, after{};
        desktop_info(desktop, &before);
        if (request.kind == Request::Kind::Send) desktop_send(desktop, request.text.data(), request.text.size());
        else desktop_reset(desktop);
        desktop_info(desktop, &after);
        std::lock_guard<std::mutex> lock(mutex_);
        state_.info = after;
        if (request.kind == Request::Kind::Send) {
            const double seconds = after.decode_seconds - before.decode_seconds;
            const uint32_t tokens = after.generated - before.generated;
            state_.last_rate = seconds > 0 ? tokens / seconds : 0;
            state_.last_stopped = after.stopped != 0;
            state_.phase = Phase::Ready;
            state_.phase_started = now_seconds();
        }
        changed();
    }
}

void ChatSession::on_text(void *ctx, const unsigned char *bytes, uint32_t n) {
    auto *session = static_cast<ChatSession *>(ctx);
    if (!n) return;
    std::lock_guard<std::mutex> lock(session->mutex_);
    auto &entries = session->state_.entries;
    if (entries.empty() || entries.back().role != Role::Assistant) entries.push_back({Role::Assistant, {}});
    entries.back().text.append(reinterpret_cast<const char *>(bytes), n);
    session->changed();
}

void ChatSession::on_reply_end(void *ctx) {
    auto *session = static_cast<ChatSession *>(ctx);
    std::lock_guard<std::mutex> lock(session->mutex_);
    auto &entries = session->state_.entries;
    if (!entries.empty() && entries.back().role == Role::Assistant) {
        std::string &text = entries.back().text;
        text.resize(complete_utf8(text));  // a reply stopped inside a character
        while (!text.empty() && (text.back() == '\n' || text.back() == ' ')) text.pop_back();
    }
    session->changed();
}

void ChatSession::on_notice(void *ctx, const char *line) {
    auto *session = static_cast<ChatSession *>(ctx);
    char text[kNoticeBytes];
    desktop_notice_text(line, text, sizeof(text));
    std::lock_guard<std::mutex> lock(session->mutex_);
    session->state_.entries.push_back({Role::Notice, text});
    session->changed();
}

int ChatSession::on_stop(void *ctx) { return static_cast<ChatSession *>(ctx)->stop_.load() ? 1 : 0; }

void ChatSession::on_stage(void *ctx, DesktopStage stage) {
    auto *session = static_cast<ChatSession *>(ctx);
    std::lock_guard<std::mutex> lock(session->mutex_);
    session->state_.stage = stage;
    session->changed();
}
