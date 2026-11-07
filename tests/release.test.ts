/** The pure parts of scripts/release.ts: split parts, checksums, release notes. */
import { afterAll, describe, expect, test } from "bun:test";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import {
  absoluteLinks, changelogSection, exeSuffix, GUI_PLATFORMS, partNames, sha256File, splitFile, sumsText,
} from "../scripts/release";
import { tempDir } from "./support/run";

const work = tempDir("release");
afterAll(() => work.dispose());

describe("release files", () => {
  test("parts are numbered so that cat name.0* joins them in order", () => {
    expect(partNames("phi.smoe", 3)).toEqual(["phi.smoe.001", "phi.smoe.002", "phi.smoe.003"]);
  });

  test("a split joins back to the same bytes and reports the whole file's hash", () => {
    const path = join(work.path, "model.smoe");
    const bytes = Buffer.from(Array.from({ length: 10_001 }, (_, i) => (i * 31) & 255));
    writeFileSync(path, bytes);
    const out = join(work.path, "parts");
    mkdirSync(out);
    const { parts, sha256 } = splitFile(path, out, 4096);
    expect(parts.map((p) => p.slice(out.length + 1))).toEqual(["model.smoe.001", "model.smoe.002", "model.smoe.003"]);
    expect(Buffer.concat(parts.map((p) => readFileSync(p)))).toEqual(bytes);
    expect(sha256).toBe(sha256File(path));
    expect(readFileSync(parts[2]!).length).toBe(10_001 - 2 * 4096);
  });

  test("sums use the sha256sum format, sorted by name", () => {
    const text = sumsText(new Map([["b.exe", "22"], ["a.smoe", "11"]]));
    expect(text).toBe("11  a.smoe\n22  b.exe\n");
  });

  test("notes are the CHANGELOG section of the version, with or without the v", () => {
    const changelog = "# Changelog\n\n## Não lançado\n- x\n\n## 0.3.0 — 2026-11-07\n\n### Adicionado\n- !comando\n\n" +
      "## 0.2.0 — 2026-11-06\n- old\n";
    expect(changelogSection(changelog, "v0.3.0")).toBe("### Adicionado\n- !comando\n");
    expect(changelogSection(changelog, "v0.3.0-rc1")).toBe("### Adicionado\n- !comando\n");
    expect(changelogSection(changelog, "0.2.0")).toBe("- old\n");
    expect(() => changelogSection(changelog, "9.9.9")).toThrow("no \"## 9.9.9\" section");
  });

  test("relative links point at the tagged files on the release page", () => {
    const text = "Ver [INSTALL](docs/INSTALL.md), [site](https://x.org) e [acima](#uso).";
    expect(absoluteLinks(text, "dono/repo", "v0.3.0")).toBe(
      "Ver [INSTALL](https://github.com/dono/repo/blob/v0.3.0/docs/INSTALL.md), [site](https://x.org) e [acima](#uso).");
  });

  test("only Windows executables end in .exe", () => {
    expect(exeSuffix("windows-x64")).toBe(".exe");
    expect(exeSuffix("linux-arm64")).toBe("");
    expect(exeSuffix("macos")).toBe("");
  });

  test("the Windows release must carry the desktop window; the others build it from source", () => {
    expect([...GUI_PLATFORMS]).toEqual(["windows-x64"]);
  });
});
