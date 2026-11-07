/* Runs a WASI module like a native program:
 *   bun tests/support/wasi-run.ts build/wasi/smolmoe.wasm --model C:/x/model.smoe ...
 * Each argument that is an absolute host path (of a file, a directory, or a
 * file to be created in an existing directory) becomes a path under a
 * preopened guest directory, so the module sees only the directories its
 * arguments name.
 * Stdin is read whole before the module starts; SM_* environment variables
 * pass through. */
import { existsSync, statSync } from "node:fs";
import { basename, dirname, isAbsolute, parse } from "node:path";
import { Wasi } from "./wasi";

/* The host directory to preopen for an argument and the name inside it, or
 * undefined when the argument is not a path: an existing directory is opened
 * itself; a file (or a file to create) through its directory, never a
 * filesystem root, so "/sair" stays text. */
function hostPath(arg: string): { dir: string; name: string } | undefined {
  if (!isAbsolute(arg)) return undefined;
  if (existsSync(arg) && statSync(arg).isDirectory()) return { dir: arg, name: "" };
  const dir = dirname(arg);
  if (dir === parse(arg).root || !existsSync(dir)) return undefined;
  return { dir, name: basename(arg) };
}

export function guestArgs(args: readonly string[]): { args: string[]; preopens: Record<string, string> } {
  const preopens: Record<string, string> = {};
  const byHost = new Map<string, string>();
  const mapped = args.map((arg) => {
    const host = hostPath(arg);
    if (!host) return arg;
    let guest = byHost.get(host.dir);
    if (!guest) {
      guest = `/host${byHost.size}`;
      byHost.set(host.dir, guest);
      preopens[guest] = host.dir;
    }
    return host.name ? `${guest}/${host.name}` : guest;
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
  /* Worker instances for a threaded module (smolmoe-mt.wasm); its --threads picks how many it uses. */
  const threads = Number(process.env.SMOLMOE_WASM_THREADS ?? 4);
  process.exit(await new Wasi({ args: ["smolmoe", ...args], preopens, env, stdin }).run(module, threads));
}
