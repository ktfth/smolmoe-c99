/* Text shown on the page, kept apart from the DOM code so the tests can import it. */

/* "[ferramenta] name: call -> {json}" reads as "name: summary" when the
 * result has one; any other notice is shown as it came. */
export function readableNotice(line: string): string {
  const tool = /^\[ferramenta\] ([a-z_]+): .* -> (\{.*\})$/.exec(line);
  if (!tool) return line;
  try {
    const result = JSON.parse(tool[2]!) as { resumo?: unknown; erro?: unknown };
    const text = typeof result.resumo === "string" ? result.resumo : typeof result.erro === "string" ? `erro: ${result.erro}` : "";
    return text ? `ferramenta ${tool[1]}: ${text}` : line;
  } catch {
    return line;
  }
}
