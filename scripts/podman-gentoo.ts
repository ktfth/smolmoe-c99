#!/usr/bin/env bun
// Isolated builds of smolmoe parts inside Gentoo containers via Podman.
// Includes the live working tree (including uncommitted local changes) by
// mounting the repo read-only and copying sources into an ephemeral /work
// inside each throwaway container.

import { copyFileSync, mkdirSync } from "node:fs";
import { resolve, join } from "node:path";

const REPO = resolve(import.meta.dir, "..");
const IMAGE = process.env.SMOLMOE_GENTOO_IMAGE ?? "localhost/smolmoe-gentoo-mingw:latest";
const WASM_IMAGE = process.env.SMOLMOE_GENTOO_WASM_IMAGE ?? "localhost/smolmoe-gentoo-wasm:latest";
const OUT_ROOT = resolve(REPO, process.env.SMOLMOE_GENTOO_OUT ?? "build/gentoo");

interface PartSpec {
  name: string;
  target: "windows-x64" | "linux-x64" | "wasm32";
  image?: string;
  description: string;
  cmd: string;
  postHost?: (outDir: string) => boolean;
}

const WIN_ENV =
  "OS=Windows_NT CC=x86_64-w64-mingw32-gcc CXX=x86_64-w64-mingw32-g++";

const COPY_SRC = [
  "set -eu",
  "mkdir -p /work",
  "tar -C /src --exclude='./build' --exclude='./dist' --exclude='./models' --exclude='./downloads' --exclude='./.git' --exclude='./.claude' --exclude='./node_modules' --exclude='./NUL' -cf - . | tar -C /work -xf -",
  "cd /work",
  "JOBS=$(nproc)",
].join(" && ");

const PARTS: PartSpec[] = [
  {
    name: "win-cli",
    target: "windows-x64",
    description: "Windows x64 CLI + tools (smolmoe.exe, smolmoe-quantize.exe, smolmoe-bundle.exe, static)",
    cmd: [
      COPY_SRC,
      `make -j"$JOBS" ${WIN_ENV} BUILD=/tmp/b OPT="-O3" LDFLAGS="-s -static" all`,
      "cp -f /tmp/b/smolmoe.exe /tmp/b/smolmoe-quantize.exe /tmp/b/smolmoe-bundle.exe /out/",
      "echo '--- DLL imports (smolmoe.exe) ---'",
      "x86_64-w64-mingw32-objdump -p /out/smolmoe.exe | grep 'DLL Name:' || true",
      "sha256sum /out/*.exe",
    ].join(" && "),
  },
  {
    name: "win-scalar",
    target: "windows-x64",
    description: "Windows x64 scalar-only runtime (smolmoe-scalar.exe, -DSM_NO_SIMD, static)",
    cmd: [
      COPY_SRC,
      `make -j"$JOBS" ${WIN_ENV} BUILD=/tmp/b OPT="-O3" LDFLAGS="-s -static" scalar`,
      "cp -f /tmp/b/smolmoe-scalar.exe /out/",
      "sha256sum /out/*.exe",
    ].join(" && "),
  },
  {
    name: "win-gui",
    target: "windows-x64",
    description: "Windows x64 Desktop GUI (smolmoe-gui.exe, Dear ImGui + Win32 + D3D11, static)",
    cmd: [
      COPY_SRC,
      `make -j"$JOBS" ${WIN_ENV} BUILD=/tmp/b OPT="-O3" LDFLAGS="-s" gui`,
      "cp -f /tmp/b/smolmoe-gui.exe /out/",
      "echo '--- DLL imports (smolmoe-gui.exe) ---'",
      "x86_64-w64-mingw32-objdump -p /out/smolmoe-gui.exe | grep 'DLL Name:' || true",
      "sha256sum /out/*.exe",
    ].join(" && "),
  },
  {
    name: "win-tests",
    target: "windows-x64",
    description: "Windows x64 C unit tests & test helpers (static .exe binaries)",
    cmd: [
      COPY_SRC,
      `make -j"$JOBS" ${WIN_ENV} BUILD=/tmp/b OPT="-O3" LDFLAGS="-static" ` +
        "/tmp/b/test-kernels.exe /tmp/b/test-tools.exe /tmp/b/test-tokenizer.exe " +
        "/tmp/b/test-calc.exe /tmp/b/test-shell.exe /tmp/b/test-recipes.exe " +
        "/tmp/b/test-cmdcheck.exe /tmp/b/test-book.exe /tmp/b/test-desktop.exe " +
        "/tmp/b/test-errors.exe /tmp/b/test-papel.exe /tmp/b/test-regex.exe /tmp/b/console-input.exe",
      "cp -f /tmp/b/test-*.exe /tmp/b/console-input.exe /out/",
      "sha256sum /out/*.exe",
    ].join(" && "),
  },
  {
    name: "win-inventario",
    target: "windows-x64",
    description: "Windows x64 defensive inventory (smolmoe-inventario.exe, test-inventory.exe, static)",
    cmd: [
      COPY_SRC,
      `make -j"$JOBS" ${WIN_ENV} BUILD=/tmp/b OPT="-O3" LDFLAGS="-s -static" /tmp/b/smolmoe-inventario.exe /tmp/b/test-inventory.exe`,
      "cp -f /tmp/b/smolmoe-inventario.exe /tmp/b/test-inventory.exe /out/",
      "sha256sum /out/*.exe",
    ].join(" && "),
  },
  {
    name: "linux-cli",
    target: "linux-x64",
    description: "Linux x64 native Gentoo GCC build + C unit tests",
    cmd: [
      COPY_SRC,
      `make -j"$JOBS" BUILD=/tmp/b all scalar inventario test-unit`,
      "cp -f /tmp/b/smolmoe /tmp/b/smolmoe-scalar /tmp/b/smolmoe-quantize /tmp/b/smolmoe-bundle /tmp/b/smolmoe-inventario /out/",
      "sha256sum /out/smolmoe*",
    ].join(" && "),
  },
  {
    name: "wasm",
    target: "wasm32",
    image: WASM_IMAGE,
    description: "WebAssembly (WASI + browser reactor SIMD128/threads via Gentoo zig-bin)",
    cmd: [
      COPY_SRC,
      "ZIG_LOCAL_CACHE_DIR=/tmp/zig-local ZIG_GLOBAL_CACHE_DIR=/tmp/zig-global make -j\"$JOBS\" WASI_MAX_MEMORY=2147418112 wasi build/web/smolmoe.wasm",
      "mkdir -p /out/wasi /out/web",
      "cp -f build/wasi/smolmoe.wasm build/wasi/smolmoe-mt.wasm build/wasi/test-kernels.wasm /out/wasi/",
      "cp -f build/web/smolmoe.wasm /out/web/",
      "sha256sum /out/wasi/*.wasm /out/web/*.wasm",
    ].join(" && "),
    postHost: (outDir) => {
      const webDir = join(REPO, "build", "web");
      const wasiDir = join(REPO, "build", "wasi");
      mkdirSync(webDir, { recursive: true });
      mkdirSync(wasiDir, { recursive: true });
      copyFileSync(join(outDir, "web", "smolmoe.wasm"), join(webDir, "smolmoe.wasm"));
      for (const f of ["smolmoe.wasm", "smolmoe-mt.wasm", "test-kernels.wasm"]) {
        copyFileSync(join(outDir, "wasi", f), join(wasiDir, f));
      }
      const bundle = run(
        ["bun", "build", "web/app.ts", "web/engine-worker.ts", "web/pool-worker.ts", "--outdir", webDir, "--target", "browser", "--minify"],
        "bundle web/*.ts -> build/web"
      );
      if (!bundle.ok) return false;
      for (const f of ["index.html", "vercel.json", "models.json"]) {
        copyFileSync(join(REPO, "web", f), join(webDir, f));
      }
      return true;
    },
  },
];

function run(args: string[], label: string): { ok: boolean; durationMs: number } {
  const t0 = performance.now();
  const proc = Bun.spawnSync(args, {
    cwd: REPO,
    stdout: "inherit",
    stderr: "inherit",
  });
  const durationMs = Math.round(performance.now() - t0);
  if (proc.exitCode !== 0) {
    console.error(`[FAIL] ${label} (exit ${proc.exitCode}, ${(durationMs / 1000).toFixed(1)}s)`);
    return { ok: false, durationMs };
  }
  console.log(`[ OK ] ${label} (${(durationMs / 1000).toFixed(1)}s)`);
  return { ok: true, durationMs };
}

function buildImage(target?: string): boolean {
  let ok = true;
  if (!target || target === "mingw" || target === "toolchain" || target === "all") {
    ok =
      run(["podman", "build", "--target", "toolchain", "-t", IMAGE, "container/gentoo"], `podman build ${IMAGE}`).ok &&
      ok;
  }
  if (!target || target === "wasm" || target === "all") {
    ok =
      run(["podman", "build", "--target", "wasm", "-t", WASM_IMAGE, "container/gentoo"], `podman build ${WASM_IMAGE}`).ok &&
      ok;
  }
  return ok;
}

function imageExists(image: string): boolean {
  const proc = Bun.spawnSync(["podman", "image", "exists", image], { cwd: REPO });
  return proc.exitCode === 0;
}

function buildPart(part: PartSpec): { name: string; ok: boolean; durationMs: number } {
  const img = part.image ?? IMAGE;
  if (!imageExists(img)) {
    const target = img === WASM_IMAGE ? "wasm" : "toolchain";
    console.log(`\nImagem ${img} ainda não existe localmente; construindo estágio '${target}'...`);
    if (!buildImage(target)) return { name: part.name, ok: false, durationMs: 0 };
  }
  const outDir = join(OUT_ROOT, part.name);
  mkdirSync(outDir, { recursive: true });
  console.log(`\n=== [${part.name}] ${part.description} ===`);
  const res = run(
    [
      "podman",
      "run",
      "--rm",
      "--name",
      `smolmoe-gentoo-${part.name}-${process.pid}`,
      "-v",
      `${REPO}:/src:ro`,
      "-v",
      `${outDir}:/out`,
      img,
      "/bin/sh",
      "-c",
      part.cmd,
    ],
    part.name
  );
  if (!res.ok) return { name: part.name, ok: false, durationMs: res.durationMs };
  if (part.postHost && !part.postHost(outDir)) {
    return { name: part.name, ok: false, durationMs: res.durationMs };
  }
  return { name: part.name, ok: true, durationMs: res.durationMs };
}

const args = process.argv.slice(2);
const cmd = args[0] ?? "help";

if (cmd === "image") {
  process.exit(buildImage(args[1]) ? 0 : 1);
} else if (cmd === "list") {
  for (const p of PARTS) {
    console.log(`${p.name.padEnd(16)} [${p.target}] ${p.description}`);
  }
  process.exit(0);
} else if (cmd === "build") {
  const requested = args.slice(1);
  const selected =
    requested.length === 0
      ? PARTS.filter((p) => p.target === "windows-x64")
      : requested.includes("all")
        ? PARTS
        : PARTS.filter((p) => requested.includes(p.name));

  if (selected.length === 0) {
    console.error("Nenhuma parte correspondente. Partes disponíveis:", PARTS.map((p) => p.name).join(", "));
    process.exit(2);
  }

  const results = selected.map(buildPart);
  console.log("\n=== Resumo dos builds isolados no Gentoo (Podman) ===");
  for (const r of results) {
    const status = r.ok ? "PASS" : "FAIL";
    console.log(`  ${status}  ${r.name.padEnd(16)} ${(r.durationMs / 1000).toFixed(1)}s  -> ${join(OUT_ROOT, r.name)}`);
  }
  process.exit(results.every((r) => r.ok) ? 0 : 1);
} else {
  console.log("Uso: bun scripts/podman-gentoo.ts <image|list|build [partes...|all]>");
  console.log("Partes:", PARTS.map((p) => p.name).join(", "));
}
