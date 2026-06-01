import { useMemo } from "react";
import { highlightColor } from "../theme";

// Renders the submission text with watermelon highlights over flagged character
// ranges (highlights: [{start, end, score}], already merged by the backend).
export default function HighlightedText({ text = "", highlights = [] }) {
  const segments = useMemo(() => {
    if (!text) return [];
    const hs = [...highlights].sort((a, b) => a.start - b.start);
    const out = [];
    let cursor = 0;
    for (const h of hs) {
      const start = Math.max(0, Math.min(text.length, h.start));
      const end = Math.max(start, Math.min(text.length, h.end));
      if (start > cursor) out.push({ t: text.slice(cursor, start) });
      out.push({ t: text.slice(start, end), score: h.score });
      cursor = end;
    }
    if (cursor < text.length) out.push({ t: text.slice(cursor) });
    return out;
  }, [text, highlights]);

  return (
    <div className="max-h-[30rem] overflow-auto whitespace-pre-wrap rounded-xl bg-lemon-50 p-4 text-sm leading-7">
      {segments.map((s, i) =>
        s.score != null ? (
          <mark key={i} className="flag" style={{ background: highlightColor(s.score) }}
            title={`similarity ${(s.score * 100).toFixed(0)}%`}>
            {s.t}
          </mark>
        ) : (
          <span key={i}>{s.t}</span>
        )
      )}
    </div>
  );
}
