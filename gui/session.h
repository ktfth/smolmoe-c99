// One chat in the desktop window: a worker thread owns the model
// (src/desktop.h) and answers the requests the window queues; the window reads
// the transcript and the state under a lock. No Dear ImGui here.
#ifndef SMOLMOE_GUI_SESSION_H
#define SMOLMOE_GUI_SESSION_H

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "desktop.h"

enum class Phase {
    NoModel,  // waiting for a file; `error` may say why the last one was refused
    Opening,  // the worker is loading the model
    Ready,
    Busy,     // answering a message
    Failed    // a fatal error on the worker; the thread has finished and can be joined
};

enum class Role { User, Assistant, Notice };

struct Entry {
    Role role;
    std::string text;
};

// What the window draws; copied out whenever `version` changes.
struct SessionState {
    Phase phase = Phase::NoModel;
    DesktopStage stage = DESKTOP_CHECKING;
    std::vector<Entry> entries;
    std::string error;
    std::string model_name;
    DesktopInfo info{};
    double phase_started = 0;  // now_seconds() when the phase began
    double last_rate = 0;      // tokens per second of the last reply
    bool last_stopped = false;
    uint64_t version = 0;
};

// The length of s up to its last complete UTF-8 character: a reply arrives in
// pieces, and a stopped one can end inside a character.
size_t complete_utf8(const std::string &s);

struct SessionOptions {
    uint32_t context = 0;  // 0: desktop.h's default
    uint32_t threads = 0;  // 0: the physical cores
};

class ChatSession {
public:
    // wake: called from the worker after every change, so a window that sleeps
    // between events can redraw (PostMessage, glfwPostEmptyEvent).
    ChatSession(SessionOptions options, std::function<void()> wake);
    ~ChatSession();
    ChatSession(const ChatSession &) = delete;
    ChatSession &operator=(const ChatSession &) = delete;

    // Starts loading a model (path empty: the one inside this executable).
    // Ignored unless the phase is NoModel.
    void open(const std::string &path);
    // Queues a message; false when no model is ready for it.
    bool send(const std::string &text);
    // Ends the reply being written at the next token.
    void stop();
    // Starts a new conversation; ignored while a reply is being written.
    void reset();

    // The state, if it changed since `seen` (a version number).
    std::optional<SessionState> changed_since(uint64_t seen) const;

    // Stops the worker and closes the model. Waits until the thread has finished.
    bool shutdown();

    // Installs the die() handler for a failure on the window's own thread
    // (bad arguments, before a model is open). Call once, from that thread.
    static void install_die_handler(void (*on_window_thread)(const char *message));

private:
    struct Request {
        enum class Kind { Send, Reset } kind;
        std::string text;
    };

    void run(std::string path);
    void serve(Desktop *desktop);
    void changed();  // with the lock held

    static void on_text(void *ctx, const unsigned char *bytes, uint32_t n);
    static void on_reply_end(void *ctx);
    static void on_notice(void *ctx, const char *line);
    static int on_stop(void *ctx);
    static void on_stage(void *ctx, DesktopStage stage);
    static void on_die(const char *message);

    const SessionOptions options_;
    const std::function<void()> wake_;
    mutable std::mutex mutex_;
    std::condition_variable requests_ready_;
    std::deque<Request> requests_;
    SessionState state_;
    std::atomic<bool> stop_{false};
    bool quit_ = false;
    bool worker_running_ = false;
    std::thread worker_;
};

#endif
