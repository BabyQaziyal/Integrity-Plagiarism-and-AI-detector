import { api } from "../api/client";
import { COLORS, riskColor, verdictTone } from "../theme";
import { Bar, Badge, Card, Gauge, ScoreCard, Empty } from "./ui";
import HighlightedText from "./HighlightedText";
import PdfViewer from "./PdfViewer";

const pct = (v) => (typeof v === "number" ? `${Math.round(v * 100)}%` : "—");

export default function ResultView({ submission }) {
  if (!submission?.analysis) return <Empty>No analysis available for this submission.</Empty>;
  const a = submission.analysis;
  const r = a.result || {};
  const plag = r.plagiarism || {};
  const ai = r.ai_content || {};
  const sty = r.stylometry || {};
  const hist = sty.history || {};
  const fp = sty.fingerprint || {};
  const comp = ai.components || {};
  const tone = verdictTone(a.verdict);
  const isPdf = (submission.filename || "").toLowerCase().endsWith(".pdf");
  const errors = r.meta?.errors || {};

  const signals = [
    {
      k: "DistilBERT classifier",
      v: comp.classifier,
      d: ai.classifier?.is_trained ? "fine-tuned model" : "not trained — ignored",
      explain: "Supervised AI-vs-human classifier — the primary signal.",
    },
    {
      k: "GPT-2 perplexity",
      v: comp.perplexity,
      d: ai.perplexity?.perplexity != null ? `perplexity ${ai.perplexity.perplexity}` : "n/a",
      explain: "AI text is more predictable, so it scores lower perplexity.",
    },
    {
      k: "Burstiness",
      v: comp.burstiness,
      d: ai.burstiness?.cv != null ? `variation ${ai.burstiness.cv}` : "n/a",
      explain: "Humans vary sentence length more than models do.",
    },
  ];

  return (
    <div className="space-y-5">
      {/* header summary */}
      <Card>
        <div className="flex flex-wrap items-center justify-between gap-5">
          <div className="flex items-center gap-5">
            <Gauge score={a.integrity_score} />
            <div>
              <div className="text-xl font-bold">{submission.title}</div>
              <div className="text-sm text-muted">
                {submission.student?.name || "Unknown student"}
                {submission.course ? ` · ${submission.course}` : ""} · {submission.word_count} words
              </div>
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <Badge bg={tone.bg} fg={tone.fg} dot>
                  {a.verdict}
                </Badge>
                <Badge bg="#fff1f3" fg={COLORS.watermelon}>
                  Plagiarism {Math.round(a.plagiarism_percent)}%
                </Badge>
                <Badge bg="#fff1f3" fg={COLORS.watermelon}>
                  AI {Math.round(a.ai_percent)}%
                </Badge>
              </div>
            </div>
          </div>
          <a
            className="btn-primary"
            href={api.reportUrl(submission.id)}
            target="_blank"
            rel="noreferrer"
          >
            ⬇ Download PDF report
          </a>
        </div>
      </Card>

      {/* score cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <ScoreCard
          label="Integrity score"
          value={Math.round(a.integrity_score)}
          color={riskColor(a.integrity_score)}
          hint="100 = clean · lower = riskier"
          big
        />
        <ScoreCard
          label="Plagiarism"
          value={Math.round(a.plagiarism_percent)}
          suffix="%"
          color={COLORS.watermelon}
          hint={`${plag.flagged_count || 0}/${plag.chunk_count || 0} chunks flagged`}
        />
        <ScoreCard
          label="AI-generated"
          value={Math.round(a.ai_percent)}
          suffix="%"
          color={COLORS.watermelon}
          hint="blended detector confidence"
        />
      </div>

      {Object.keys(errors).length > 0 && (
        <div className="rounded-xl border border-lemon-300/60 bg-lemon-100 px-4 py-2.5 text-xs text-ink/70">
          <span className="font-semibold">Notes:</span>{" "}
          {Object.entries(errors)
            .map(([k, v]) => `${k}: ${v}`)
            .join(" · ")}
        </div>
      )}

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        {/* AI signals */}
        <Card title="AI-content signals" subtitle="How the AI score was reached">
          {signals.map((row) => (
            <div key={row.k} className="mb-4 last:mb-0">
              <div className="mb-1 flex items-center justify-between text-sm">
                <span className="font-medium">{row.k}</span>
                <span className="tabular-nums text-muted">
                  {pct(row.v)} <span className="text-xs">· {row.d}</span>
                </span>
              </div>
              <Bar value={(row.v || 0) * 100} color={COLORS.watermelon} />
              <div className="mt-1 text-[11px] text-muted/80">{row.explain}</div>
            </div>
          ))}
          <div className="mt-3 border-t border-black/5 pt-3 text-xs text-muted">
            Combined AI score blends the available signals (weights:{" "}
            {Object.entries(ai.used_weights || {})
              .map(([k, w]) => `${k} ${w}`)
              .join(", ") || "—"}
            ).
          </div>
        </Card>

        {/* Matched sources */}
        <Card title="Matched sources" subtitle="Closest reference passages">
          {plag.matched_sources?.length ? (
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs uppercase text-muted">
                  <th className="pb-2">Source</th>
                  <th className="pb-2 text-right">Similarity</th>
                  <th className="pb-2 text-right">Hits</th>
                </tr>
              </thead>
              <tbody>
                {plag.matched_sources.slice(0, 8).map((s, i) => (
                  <tr key={i} className="border-t border-black/5">
                    <td className="max-w-0 truncate py-1.5 pr-2" title={s.title || s.source_id}>
                      {s.title || s.source_id}
                    </td>
                    <td
                      className="py-1.5 text-right font-semibold tabular-nums"
                      style={{ color: COLORS.watermelon }}
                    >
                      {s.max_score?.toFixed(2)}
                    </td>
                    <td className="py-1.5 text-right tabular-nums text-muted">{s.hits}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <Empty icon="🔍">No significant source matches.</Empty>
          )}
        </Card>
      </div>

      {/* highlighted text */}
      <Card
        title="Submission text"
        subtitle={`${plag.flagged_count || 0} of ${plag.chunk_count || 0} chunks flagged`}
        action={
          <span className="hidden items-center gap-2 text-xs text-muted sm:flex">
            <span className="inline-block h-3 w-6 rounded" style={{ background: "rgba(240,72,95,0.45)" }} />
            similar to a source
          </span>
        }
      >
        <HighlightedText text={submission.text || ""} highlights={plag.highlights || []} />
      </Card>

      {/* PDF preview */}
      {isPdf && (
        <Card title="Original document">
          <PdfViewer url={api.fileUrl(submission.id)} />
        </Card>
      )}

      {/* stylometry */}
      <Card title="Writing fingerprint & history" subtitle="Style profile vs the student's past work">
        <div className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm sm:grid-cols-3">
          <Stat label="Avg sentence length" value={fp.avg_sentence_len} />
          <Stat label="Vocabulary richness (TTR)" value={fp.type_token_ratio} />
          <Stat label="Avg word length" value={fp.avg_word_len} />
          <Stat label="Readability (Flesch)" value={fp.flesch_reading_ease} />
          <Stat label="Grade level" value={fp.flesch_kincaid_grade} />
          <Stat label="Commas / 100 words" value={fp.punct_comma} />
        </div>
        <div className="mt-4 border-t border-black/5 pt-3 text-sm">
          {hist.consistency != null ? (
            <div>
              <span className="font-semibold">Style consistency vs history: </span>
              <span className="font-bold" style={{ color: riskColor(hist.consistency) }}>
                {Math.round(hist.consistency)}/100
              </span>
              <span className="text-muted">
                {" "}
                · {hist.n_history} prior submission(s) · {hist.flags?.length || 0} feature(s) deviated &gt;2σ
              </span>
            </div>
          ) : (
            <div className="text-muted">{hist.note || "No writing history yet."}</div>
          )}
        </div>
      </Card>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div>
      <div className="text-xs text-muted">{label}</div>
      <div className="font-semibold tabular-nums">{value ?? "—"}</div>
    </div>
  );
}
