/**
 * The command book's search: a Portuguese question finds the book's tasks
 * whose phrases share its words, weighted by how rare each word is in the book.
 * Kept to what C does cheaply (fold, split, a crude stem, a table of weights),
 * so the runtime can do the same if the measurement says it helps.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

export type Shell = "powershell" | "posix";

export interface Tarefa {
  readonly id: string;
  readonly frases: readonly string[];
  readonly powershell: readonly string[];
  readonly posix: readonly string[];
}

export interface Achado {
  readonly tarefa: Tarefa;
  readonly pontos: number;
}

export const LIVRO = join(import.meta.dir, "livro.json");
export const MIN_PONTOS = 2.5; /* below this a match is one common word: no book in the prompt */
export const SEGUNDA = 0.75; /* a second task joins when it scores this close to the first */
export const MAX_TAREFAS = 2;

export const STOP = new Set(("a o as os um uma uns umas de do da dos das no na nos nas em num numa para pra por pelo pela " +
  "com sem se e ou que qual quais quem como onde quando quanto eu me meu minha meus minhas seu sua isso esse essa " +
  "este esta aqui ali ja nao sim mais muito ver vejo veja posso pode consigo faco fazer quero preciso comando " +
  "tem ter ta estar estou ser sao vai vou foi saber sei sobre todo toda tudo " +
  "comandos powershell bash shell terminal linux windows ubuntu debian maquina computador pc").split(" "));

export function fold(s: string): string {
  return s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
}

/* Plural, verb ending and final vowel off, then six letters: "reiniciar",
 * "reinicie" and "reinício" meet at "reinic"; "serviço" and "servidor" do not. */
export function stem(w: string): string {
  if (w.length > 3 && w.endsWith("s")) w = w.slice(0, -1);
  if (w.length > 5 && /(ando|endo|indo)$/.test(w)) w = w.slice(0, -4);
  else if (w.length > 4 && /(ar|er|ir|am|em)$/.test(w)) w = w.slice(0, -2);
  if (w.length > 3 && /[aeo]$/.test(w)) w = w.slice(0, -1);
  return w.slice(0, 6);
}

export function palavras(s: string): string[] {
  return fold(s).split(/[^a-z0-9]+/).filter((w) => w.length >= 2 && !STOP.has(w)).map(stem);
}

export interface Livro {
  readonly tarefas: readonly Tarefa[];
  readonly termos: ReadonlyMap<string, ReadonlySet<string>>; /* task id -> its stems */
  readonly frases: ReadonlyMap<string, readonly (readonly string[])[]>; /* task id -> each phrase's stems */
  readonly peso: ReadonlyMap<string, number>; /* stem -> idf */
}

export function carregar(file = LIVRO): Livro {
  const tarefas = (JSON.parse(readFileSync(file, "utf8")) as { tarefas: Tarefa[] }).tarefas;
  const frases = new Map(tarefas.map((t) => [t.id, t.frases.map((f) => [...new Set(palavras(f))])]));
  const termos = new Map(tarefas.map((t) => [t.id, new Set(frases.get(t.id)!.flat())]));
  const df = new Map<string, number>();
  for (const set of termos.values()) for (const w of set) df.set(w, (df.get(w) ?? 0) + 1);
  const peso = new Map([...df].map(([w, n]) => [w, Math.log(1 + tarefas.length / n)]));
  return { tarefas, termos, frases, peso };
}

export const BONUS_FRASE = 2; /* points for a phrase the question covers whole: breaks ties between tasks sharing words */

export function buscar(livro: Livro, pergunta: string, shell: Shell): Achado[] {
  const q = new Set(palavras(pergunta));
  const achados = livro.tarefas.filter((t) => t[shell].length > 0).map((tarefa) => {
    const set = livro.termos.get(tarefa.id)!;
    let pontos = 0, casadas = 0;
    for (const w of q) if (set.has(w)) pontos += livro.peso.get(w)!, ++casadas;
    let cobertura = 0;
    for (const f of livro.frases.get(tarefa.id)!) {
      const total = f.reduce((t, w) => t + livro.peso.get(w)!, 0);
      const coberto = f.reduce((t, w) => t + (q.has(w) ? livro.peso.get(w)! : 0), 0);
      if (total > 0) cobertura = Math.max(cobertura, coberto / total);
    }
    /* One shared word is a match only when it is a whole phrase ("docker", "ping"):
     * "sobre" in a poem request is not "quanto espaço sobra". */
    const vale = casadas >= 2 || cobertura === 1;
    return { tarefa, pontos: vale ? pontos + BONUS_FRASE * cobertura : 0 };
  }).filter((a) => a.pontos >= MIN_PONTOS).sort((x, y) => y.pontos - x.pontos);
  if (!achados.length) return [];
  const top = achados[0]!.pontos;
  return achados.slice(0, MAX_TAREFAS).filter((a, i) => i === 0 || a.pontos >= SEGUNDA * top);
}

/* The prompt prefix: the tasks' example commands in a fenced block. */
export function prefixo(achados: readonly Achado[], shell: Shell): string {
  if (!achados.length) return "";
  const lines = achados.flatMap((a) => a.tarefa[shell]);
  const tag = shell === "powershell" ? "powershell" : "bash";
  return `Comandos reais que costumam resolver isso (${shell === "powershell" ? "PowerShell" : "bash"}):\n` +
    `\`\`\`${tag}\n${lines.join("\n")}\n\`\`\`\n\n`;
}
