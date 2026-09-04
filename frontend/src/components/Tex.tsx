import katex from "katex";
import "katex/dist/katex.min.css";
import { useMemo } from "react";

interface TexProps {
  children: string;
  display?: boolean;
}

/**
 * Renders a LaTeX snippet with KaTeX.
 *
 * Named `Tex` rather than `Math` deliberately: a component called `Math` would
 * shadow the global `Math` object in any module that imports it.
 */
export function Tex({ children, display = false }: TexProps) {
  const html = useMemo(
    () =>
      katex.renderToString(children, {
        displayMode: display,
        throwOnError: false,
      }),
    [children, display],
  );
  return (
    <span
      className={display ? "math-display" : "math-inline"}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
