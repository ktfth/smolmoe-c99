/* Serves build/web locally with the headers the page needs (the same as
 * web/vercel.json): COOP/COEP for shared memory and threads.
 *   bun scripts/serve-web.ts [PORT] [--port 8080] [--dir build/web] */
import { spawnSync } from "node:child_process";
import { existsSync, statSync } from "node:fs";
import { extname, isAbsolute, join, relative, resolve, sep } from "node:path";

const args = process.argv.slice(2);
const option = (name: string, shortName: string, fallback: string, allowPositionalPort = false): string => {
  const at = args.findIndex((a) => a === name || a === shortName);
  if (at >= 0 && args[at + 1]) return args[at + 1]!;
  if (allowPositionalPort) {
    const positional = args.find((a, i) => /^\d+$/.test(a) && (i === 0 || !args[i - 1]!.startsWith("-")));
    if (positional) return positional;
  }
  return fallback;
};
const root = resolve(option("--dir", "-d", "build/web"));
const port = Number(option("--port", "-p", process.env.PORT ?? "8080", true));
const TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm",
  ".smoe": "application/octet-stream",
};
const HEADERS = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
  "X-Content-Type-Options": "nosniff",
};

if (!existsSync(join(root, "index.html"))) {
  console.error(`${root} has no index.html: run make web first`);
  process.exit(1);
}

/* Bundle the local Qwen model into build/web/ when present so 1-click loading works on localhost. */
const localBundled = join(root, "qwen2.5-0.5b-instruct-q8.smoe");
const qwenDir = resolve("models/qwen2.5-0.5b-instruct");
const bundleBin = resolve(process.platform === "win32" ? "build/smolmoe-bundle.exe" : "build/smolmoe-bundle");
if (!existsSync(localBundled) && existsSync(join(qwenDir, "model.smoe")) && existsSync(bundleBin)) {
  spawnSync(bundleBin, [
    "--model", join(qwenDir, "model.smoe"),
    "--config", join(qwenDir, "config.ini"),
    "--tokenizer", join(qwenDir, "tokenizer.qbp"),
    "--out", localBundled,
  ], { stdio: "inherit" });
}

Bun.serve({
  port,
  hostname: "localhost", /* a development server: not for the rest of the network */
  fetch(request) {
    let path: string;
    try {
      path = decodeURIComponent(new URL(request.url).pathname);
    } catch {
      return new Response("bad request", { status: 400, headers: HEADERS });
    }
    if (path === "/models.json") {
      const list = [
        {
          name: "Qwen 2.5 0.5B Instruct (Q8)",
          file: "qwen2.5-0.5b-instruct-q8.smoe",
          bytes: existsSync(localBundled) ? statSync(localBundled).size : 560052800,
          sha256: "local",
          url: existsSync(localBundled) ? "./qwen2.5-0.5b-instruct-q8.smoe" : "",
        },
        {
          name: "Qwen 2.5 0.5B Instruct (Q5)",
          file: "qwen2.5-0.5b-instruct-q5.smoe",
          bytes: 343932928,
          sha256: "",
          url: existsSync(join(root, "qwen2.5-0.5b-instruct-q5.smoe")) ? "./qwen2.5-0.5b-instruct-q5.smoe" : "",
        },
        {
          name: "Gemma 3 1B Instruct (Q8)",
          file: "gemma-3-1b-it-q8.smoe",
          bytes: 1073741824,
          sha256: "",
          url: existsSync(join(root, "gemma-3-1b-it-q8.smoe")) ? "./gemma-3-1b-it-q8.smoe" : "",
        },
        {
          name: "Gemma 3 270M Instruct (Q8)",
          file: "gemma-3-270m-it-q8.smoe",
          bytes: 293601280,
          sha256: "",
          url: existsSync(join(root, "gemma-3-270m-it-q8.smoe")) ? "./gemma-3-270m-it-q8.smoe" : "",
        },
        {
          name: "Phi-4-mini Instruct 3.8B (Q8)",
          file: "phi-4-mini-instruct-q8.smoe",
          bytes: 4080218931,
          sha256: "",
          url: existsSync(join(root, "phi-4-mini-instruct-q8.smoe")) ? "./phi-4-mini-instruct-q8.smoe" : "",
        },
      ];
      return new Response(JSON.stringify(list), { headers: { ...HEADERS, "Content-Type": TYPES[".json"]! } });
    }
    const file = resolve(join(root, path.endsWith("/") ? `${path}index.html` : path));
    const inside = relative(root, file);
    if (inside === ".." || inside.startsWith(`..${sep}`) || isAbsolute(inside) || !existsSync(file) || !statSync(file).isFile())
      return new Response("not found", { status: 404, headers: HEADERS });
    return new Response(Bun.file(file), {
      headers: { ...HEADERS, "Content-Type": TYPES[extname(file)] ?? "application/octet-stream" },
    });
  },
});
console.log(`http://localhost:${port}/ (${root})`);
