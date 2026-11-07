/* Function calling for the chat, in the template Qwen2.5 was trained on: the
 * tools the runtime offers, the system-prompt section that declares them, and
 * the execution of a call the model wrote between <tool_call> and </tool_call>.
 *
 * The model only picks a tool and its arguments; dates and day counts are
 * computed here, exactly. Model output is untrusted: every call is parsed
 * strictly and any problem becomes an {"erro": ...} result, never a crash. */
#ifndef SMOLMOE_TOOLS_H
#define SMOLMOE_TOOLS_H

#include <stddef.h>

#define TOOL_NAME_MAX 32
#define TOOL_RESULT_MAX 384

/* Local wall-clock time the tools answer from. */
typedef struct ToolClock {
    int year, month, day, hour, minute;
} ToolClock;

/* Parses "YYYY-MM-DDTHH:MM" (a fixed clock for reproducible runs); 1 if valid. */
int tool_clock_parse(const char *s, ToolClock *out);
ToolClock tool_clock_now(void);

/* Writes `system` followed by the template's "# Tools" section declaring the
 * built-in tools. Returns the length, or 0 when it does not fit in cap. */
size_t tools_system_prompt(const char *system, char *out, size_t cap);

/* Whether to offer the tools for this user message: it names a date or talks
 * about time. Elsewhere small models call tools they do not need. */
int tools_relevant(const char *user_text);

/* The one tool a message clearly needs, or NULL: "calcular" for a calculation
 * that names no date or time. Used when the model's own call was discarded
 * (it picked a tool that does not fit): the runtime then opens that call. */
const char *tools_forced_choice(const char *user_text);

/* The call text the model would write after <tool_call> up to the first
 * argument's value, for a forced choice. Returns its length, or 0 when the
 * tool has no such prefix or it does not fit in cap. */
size_t tools_call_prefix(const char *tool, char *out, size_t cap);

/* Whether a call fits the user message it answers: data_atual always does;
 * dias_ate only when the message names a date or holiday. A call that does not
 * (or cannot be read) is discarded and the model answers without tools. */
int tool_call_grounded(const char *call, size_t n, const char *user_text);

/* Executes one call, given the JSON text between the markers and the user
 * message it answers: a date the user wrote wins over the model's argument,
 * and a past year the model invents is dropped. Always fills `name` ("?" when
 * unreadable) and `result` with a JSON object; returns 1 when the tool ran, 0
 * when `result` holds an error for the model to relay. */
int tool_run(const char *call, size_t n, const char *user_text, const ToolClock *now, char name[TOOL_NAME_MAX],
             char result[TOOL_RESULT_MAX]);

/* The string value of `key` in a tool result (a flat JSON object, as tool_run
 * writes), into out; 0 when it is absent, not a string, too long, or the
 * result is malformed, and then out is "" (when cap > 0). */
int tool_result_string(const char *result, const char *key, char *out, size_t cap);

#endif
