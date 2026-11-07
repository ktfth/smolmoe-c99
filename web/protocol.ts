/* Messages between the page (web/app.ts) and its engine worker. */
import type { EngineStats } from "./wasm-host";

export type ToWorker =
  | { type: "open"; url?: string; file?: File; context: number; threads: number }
  | { type: "send"; text: string }
  | { type: "reset" };

export type LoadPhase = "download" | "cache" | "file" | "prepare";

export type FromWorker =
  | { type: "progress"; phase: LoadPhase; loaded: number; total: number }
  | { type: "ready"; backend: string; threads: number; context: number; stats: EngineStats }
  | { type: "text"; chunk: string }
  | { type: "notice"; line: string }
  | { type: "done"; ok: boolean; stats: EngineStats }
  | { type: "error"; message: string; fatal: boolean };

/* A model the page offers (web/models.json). An empty url: only "open a file". */
export interface ModelEntry {
  readonly name: string;
  readonly file: string;
  readonly bytes: number;
  readonly sha256: string;
  readonly url: string;
}
