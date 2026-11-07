/**
 * The desktop window (make gui): its model layer with a real model
 * (tests/test_desktop.c on src/desktop.c: stopping a reply, rolling back a
 * stopped tool call, the files it refuses) and the window program's command
 * line. Drawing and input were checked by hand (docs/DESKTOP.md).
 */
import { afterAll, describe, expect, test } from "bun:test";
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";

import { bundledModel, DESKTOP_TEST_BINARY, GUI_BINARY, ROOT, run, tempDir } from "./support/run";

const SLOW = Number(process.env.SMOLMOE_SLOW_MS ?? 180_000);
const qwen = bundledModel("qwen2.5-0.5b-instruct", "tokenizer.qbp");
const stories = join(ROOT, "models", "tinystories-15m");
const haveStories = ["model.smoe", "config.ini", "tokenizer.spb"].every((f) => existsSync(join(stories, f)));
const guiBuilt = existsSync(GUI_BINARY);
const work = tempDir("desktop");
afterAll(() => work.dispose());

test("the window program is present where CI expects it", () => {
  if (process.env.SM_EXPECT_GUI === "1") expect(guiBuilt).toBe(true);
});

test.skipIf(!qwen || !haveStories)("model layer: stopping replies and refusing files, with Qwen", () => {
  /* A bare .smoe without config.ini beside it, and TinyStories (no chat template) with its files. */
  mkdirSync(join(work.path, "bare"));
  mkdirSync(join(work.path, "stories"));
  copyFileSync(join(stories, "model.smoe"), join(work.path, "bare", "model.smoe"));
  for (const f of ["model.smoe", "config.ini", "tokenizer.spb"]) copyFileSync(join(stories, f), join(work.path, "stories", f));
  const result = run([qwen!.model, work.path], DESKTOP_TEST_BINARY);
  expect(result.stderr).toBe("");
  expect(result.stdout.toString()).toContain("desktop tests passed (with a model)");
  expect(result.code).toBe(0);
}, SLOW);

describe.skipIf(!guiBuilt)("window program", () => {
  test("--version and --help print without opening a window", () => {
    const version = run(["--version"], GUI_BINARY);
    expect(version.code).toBe(0);
    expect(version.stdout.toString()).toMatch(/^smolmoe-gui \d+\.\d+\.\d+\n$/);
    const help = run(["--help"], GUI_BINARY).stdout.toString();
    for (const option of ["--context", "--threads", "MODELO.smoe"]) expect(help).toContain(option);
  });

  test("a bad command line fails on stderr before any window", () => {
    for (const [args, message] of [
      [["--threads", "0"], "--threads must be positive"],
      [["--context"], "missing argument for --context"],
      [["--nada"], "unknown option: --nada"],
      [["a.smoe", "b.smoe"], "only one model file"],
    ] as const) {
      const result = run(args, GUI_BINARY);
      expect(result.code).toBe(1);
      expect(result.stderr).toContain(message);
    }
  });
});
