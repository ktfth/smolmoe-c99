/* Ready-made shell commands for common questions ("quanto espaço livre tem no
 * disco?"). A small model writes plausible commands with properties that do
 * not exist (Get-PSDrive | Select-Object Size, FreeSpace prints empty columns
 * and no error), so the chat answers these questions itself, with a command
 * checked for that shell and system, and the model never writes it. Every
 * recipe only reads: nothing is deleted, moved or sent over the network.
 * Portable: matching and the table only; chat.c prints and runs. */
#ifndef SMOLMOE_RECIPES_H
#define SMOLMOE_RECIPES_H

#include "shell.h"

typedef enum { RECIPE_WINDOWS, RECIPE_LINUX, RECIPE_MACOS } RecipeOS;

typedef struct Recipe {
    const char *topic;   /* what the command shows, for the reply: "o espaço em disco" */
    const char *command; /* the exact line for that shell */
} Recipe;

/* The system this binary was built for (WebAssembly and other POSIX count as Linux). */
RecipeOS recipe_host_os(void);

/* The recipe for a chat message, when it asks for something a recipe covers
 * (a topic plus a request word such as "comando", "verifique", "quanto") and
 * is not a question about concepts ("o que é", "diferença", "explique"), and
 * the table has a command for that shell on that system. Returns 0 otherwise:
 * the message then goes to the model as usual. */
int recipe_find(const char *message, ShellKind shell, RecipeOS os, Recipe *out);

#endif
