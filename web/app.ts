/* The page: model choice and loading, then the chat. The engine runs in a
 * worker (web/engine-worker.ts); this file only touches the DOM. Model text
 * is always inserted as text, never as HTML. */
import { readableNotice } from "./format";
import type { FromWorker, LoadPhase, ModelEntry, ToWorker } from "./protocol";
import type { EngineStats } from "./wasm-host";

const DEFAULT_CONTEXT = 2048;
const MB = 1_000_000; /* as the release page and the docs count */

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const number = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 1 });

const ui = {
  setup: $("setup"),
  modelName: $("model-name"),
  modelSelect: $<HTMLSelectElement>("model-select"),
  modelHelp: $("model-help"),
  download: $<HTMLButtonElement>("download"),
  file: $<HTMLInputElement>("file"),
  progress: $<HTMLProgressElement>("progress"),
  status: $("status"),
  chat: $("chat"),
  log: $("log"),
  form: $<HTMLFormElement>("form"),
  input: $<HTMLTextAreaElement>("input"),
  send: $<HTMLButtonElement>("send"),
  stats: $("stats"),
  reset: $<HTMLButtonElement>("reset"),
};

const params = new URLSearchParams(location.search);
const worker = new Worker(new URL("./engine-worker.js", import.meta.url), { type: "module" });
let reply: HTMLElement | undefined;
let busy = false;
let backend = "";

function post(message: ToWorker): void {
  worker.postMessage(message);
}

/* Threads: half the logical cores (the other half are usually hyperthreads
 * or efficiency cores), between 1 and 8; ?threads=N overrides. */
function threads(): number {
  const asked = Number(params.get("threads"));
  if (asked >= 1) return Math.min(Math.floor(asked), 16);
  return Math.max(1, Math.min(8, Math.floor((navigator.hardwareConcurrency || 4) / 2)));
}

function setStatus(text: string, error = false): void {
  ui.status.textContent = text;
  ui.status.classList.toggle("error", error);
}

/* The page needs shared memory (cross-origin isolation) and SIMD128. */
function unsupported(): string | undefined {
  if (!self.crossOriginIsolated) return "Esta página precisa de isolamento de origem (cabeçalhos COOP e COEP) para usar threads.";
  /* A module whose one function returns a v128.const. */
  const simd = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 22, 1, 20, 0, 253, 12,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11]);
  if (!WebAssembly.validate(simd)) return "Este navegador não tem WebAssembly SIMD; use uma versão recente do Chrome, Edge, Firefox ou Safari.";
  return undefined;
}

const PHASES: Record<LoadPhase, string> = {
  download: "Baixando",
  cache: "Lendo do cache do navegador",
  file: "Lendo o arquivo",
  prepare: "Preparando o modelo (o prompt do sistema é processado uma vez)",
};

function progress(phase: LoadPhase, loaded: number, total: number): void {
  ui.progress.hidden = false;
  if (phase === "prepare") {
    ui.progress.removeAttribute("value");
    setStatus(`${PHASES.prepare}…`);
    return;
  }
  ui.progress.value = total ? loaded / total : 0;
  setStatus(`${PHASES[phase]}: ${number.format(loaded / MB)} de ${number.format(total / MB)} MB`);
}

function rate(tokens: number, seconds: number): string {
  return seconds > 0 ? `${number.format(tokens / seconds)} tokens/s` : "";
}

function showStats(s: EngineStats): void {
  const parts = [backend, `${s.threads} thread${s.threads === 1 ? "" : "s"}`, `contexto ${s.context}`];
  const speed = rate(s.generated, s.decodeSeconds);
  if (speed) parts.push(`escrita ${speed}`);
  ui.stats.textContent = parts.join(" · ");
}

function addMessage(kind: "user" | "bot" | "notice", text: string): HTMLElement {
  const div = document.createElement("div");
  div.className = `msg ${kind}`;
  div.textContent = text;
  ui.log.append(div);
  ui.log.scrollTop = ui.log.scrollHeight;
  return div;
}

function setBusy(on: boolean): void {
  busy = on;
  ui.send.disabled = on;
  ui.reset.disabled = on;
}

function start(message: ToWorker): void {
  ui.download.disabled = true;
  ui.file.disabled = true;
  setStatus("Iniciando…");
  post(message);
}

worker.onmessage = (event: MessageEvent<FromWorker>) => {
  const m = event.data;
  switch (m.type) {
    case "progress":
      progress(m.phase, m.loaded, m.total);
      break;
    case "ready":
      backend = m.backend;
      ui.setup.hidden = true;
      ui.chat.hidden = false;
      showStats(m.stats);
      addMessage("notice", `Modelo pronto em ${number.format(m.stats.prefillSeconds)} s. Ferramentas: data_atual, dias_ate, calcular, converter, contar.`);
      ui.input.focus();
      break;
    case "text":
      reply ??= addMessage("bot", "");
      reply.textContent = (reply.textContent ?? "") + m.chunk;
      ui.log.scrollTop = ui.log.scrollHeight;
      break;
    case "notice":
      reply = undefined; /* the answer after a tool call is a new bubble */
      addMessage("notice", readableNotice(m.line));
      break;
    case "done":
      if (reply) reply.textContent = (reply.textContent ?? "").trimEnd();
      reply = undefined;
      setBusy(false);
      showStats(m.stats);
      ui.input.focus();
      break;
    case "error":
      setBusy(false);
      if (!ui.chat.hidden) addMessage("notice", `Erro: ${m.message}. Recarregue a página.`);
      else setStatus(`Erro: ${m.message}. Recarregue a página para tentar de novo.`, true);
      break;
  }
};

async function init(): Promise<void> {
  const problem = unsupported();
  let models: ModelEntry[] = [];
  try {
    models = (await (await fetch("./models.json")).json()) as ModelEntry[];
  } catch {
    models = [];
  }
  let selected: ModelEntry | undefined = models[0];
  const renderModel = (model: ModelEntry | undefined) => {
    selected = model;
    if (!model) return;
    ui.modelName.textContent = model.name;
    ui.modelHelp.textContent = model.url
      ? `${number.format(model.bytes / MB)} MB, carregados localmente ou guardados no cache do navegador. Ou abra outro arquivo .smoe.`
      : `Abra o arquivo ${model.file} (${number.format(model.bytes / MB)} MB) convertido ou baixado do Release.`;
    ui.download.hidden = !model.url;
  };
  if (models.length > 1) {
    ui.modelSelect.hidden = false;
    models.forEach((m, idx) => {
      const opt = document.createElement("option");
      opt.value = String(idx);
      opt.textContent = m.url ? `${m.name} (disponível)` : m.name;
      ui.modelSelect.append(opt);
    });
    ui.modelSelect.onchange = () => renderModel(models[Number(ui.modelSelect.value)]);
  }
  renderModel(selected);
  if (problem) {
    setStatus(problem, true);
    ui.download.disabled = true;
    ui.file.disabled = true;
    return;
  }
  ui.download.onclick = () => start({ type: "open", model: selected, context: DEFAULT_CONTEXT, threads: threads() });
  ui.file.onchange = () => {
    const file = ui.file.files?.[0];
    if (file) start({ type: "open", file, context: DEFAULT_CONTEXT, threads: threads() });
  };
}

ui.form.onsubmit = (event) => {
  event.preventDefault();
  const text = ui.input.value.trim();
  if (!text || busy) return;
  ui.input.value = "";
  addMessage("user", text);
  setBusy(true);
  post({ type: "send", text });
};

ui.input.onkeydown = (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    ui.form.requestSubmit();
  }
};

ui.reset.onclick = () => {
  ui.log.replaceChildren();
  post({ type: "reset" });
  addMessage("notice", "Conversa reiniciada.");
};

void init();
