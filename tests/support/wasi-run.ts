/* Runs a WASI module like a native program:
 *   bun tests/support/wasi-run.ts build/wasi/smolmoe.wasm --model C:/x/model.smoe ...
 * Each argument that is an absolute host path (of a file, or of a file to be
 * created in an existing directory) becomes a path under a preopened guest
 * directory, so the module sees only the directories its arguments name.
 * Stdin is read whole before the module starts; SM_* environment variables
 * pass through. */
import { existsSync } from "node:fs";
import { basename, dirname, isAbsolute } from "node:path";
import { Wasi } from "./wasi";

export function guestArgs(args: readonly string[]): { args: string[]; preopens: Record<string, string> } {
  const preopens: Record<string, string> = {};
  const byHost = new Map<string, string>();
  const mapped = args.map((arg) => {
    if (!isAbsolute(arg)) return arg;
    const dir = existsSync(arg) && !existsSync(`${arg}/.`) ? dirname(arg) : existsSync(dirname(arg)) ? dirname(arg) : undefined;
    if (!dir) return arg;
    let guest = byHost.get(dir);
    if (!guest) {
      guest = `/host${byHost.size}`;
      byHost.set(dir, guest);
      preopens[guest] = dir;
    }
    return `${guest}/${basename(arg)}`;
  });
  return { args: mapped, preopens };
}

if (import.meta.main) {
  const [file, ...rest] = process.argv.slice(2);
  if (!file) {
    console.error("usage: bun tests/support/wasi-run.ts MODULE.wasm [args...]");
    process.exit(2);
  }
  const module = new WebAssembly.Module(await Bun.file(file).arrayBuffer());
  const { args, preopens } = guestArgs(rest);
  const stdin = new Uint8Array(await Bun.stdin.arrayBuffer());
  const env: Record<string, string> = {};
  for (const [k, v] of Object.entries(process.env)) if (k.startsWith("SM_") && v !== undefined) env[k] = v;
  process.exitCode = new Wasi({ args: ["smolmoe", ...args], preopens, env, stdin }).run(module);
}
