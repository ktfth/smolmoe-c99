/* Serves build/web locally with the headers the page needs (the same as
 * web/vercel.json): COOP/COEP for shared memory and threads.
 *   bun scripts/serve-web.ts [--port 8080] [--dir build/web] */
import { existsSync, statSync } from "node:fs";
import { extname, isAbsolute, join, relative, resolve, sep } from "node:path";

const args = process.argv.slice(2);
const option = (name: string, fallback: string): string => {
  const at = args.indexOf(name);
  return at >= 0 && args[at + 1] ? args[at + 1]! : fallback;
};
const root = resolve(option("--dir", "build/web"));
const port = Number(option("--port", "8080"));
const TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm",
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
