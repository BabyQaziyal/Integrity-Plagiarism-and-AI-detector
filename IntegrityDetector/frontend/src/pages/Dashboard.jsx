import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import { riskColor, verdictTone } from "../theme";
import SubmissionForm from "../components/SubmissionForm";
import { Badge, Card, Empty, ErrorState, Skeleton } from "../components/ui";

const FEATURES = [
  ["Plagiarism", "TF-IDF + cosine over sliding chunks vs a web-like corpus"],
  ["AI content", "DistilBERT + GPT-2 perplexity + burstiness"],
  ["Stylometry", "Per-student writing fingerprint & drift detection"],
  ["Report", "One-click exportable PDF integrity report"],
];

function RecentSubmissions({ reloadKey }) {
  const [subs, setSubs] = useState(null);
  const [error, setError] = useState(null);

  const load = () => {
    setError(null);
    setSubs(null);
    api
      .listSubmissions()
      .then((rows) => setSubs(rows.slice(0, 8)))
      .catch((e) => setError(e));
  };
  useEffect(load, [reloadKey]);

  return (
    <Card title="Recent submissions" subtitle="Latest analyses across all students">
      {error ? (
        <ErrorState
          title="Couldn’t load submissions"
          message={error.offline ? "The backend isn’t running yet." : error.message}
          onRetry={load}
        />
      ) : !subs ? (
        <div className="space-y-2">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-14 w-full" />
          ))}
        </div>
      ) : subs.length === 0 ? (
        <Empty icon="📝">No submissions yet — analyze one above to get started.</Empty>
      ) : (
        <ul className="grid gap-2 sm:grid-cols-2">
          {subs.map((sub) => {
            const sum = sub.summary || {};
            const tone = verdictTone(sum.verdict);
            return (
              <li key={sub.id}>
                <Link
                  to={`/submissions/${sub.id}`}
                  className="flex items-center justify-between gap-3 rounded-xl border border-black/5 bg-white px-4 py-3 transition hover:border-watermelon/40 hover:shadow-card"
                >
                  <div className="min-w-0">
                    <div className="truncate font-medium">{sub.title}</div>
                    <div className="text-xs text-muted">
                      {new Date(sub.created_at).toLocaleDateString()} · {sub.word_count} words
                    </div>
                  </div>
                  <div className="flex flex-none items-center gap-2.5">
                    <span
                      className="text-lg font-bold tabular-nums"
                      style={{ color: riskColor(sum.integrity_score) }}
                      title="Integrity score"
                    >
                      {sum.integrity_score != null ? Math.round(sum.integrity_score) : "—"}
                    </span>
                    {sum.verdict && (
                      <Badge bg={tone.bg} fg={tone.fg}>
                        {sum.verdict}
                      </Badge>
                    )}
                  </div>
                </Link>
              </li>
            );
          })}
        </ul>
      )}
    </Card>
  );
}

export default function Dashboard() {
  const navigate = useNavigate();
  const [reloadKey, setReloadKey] = useState(0);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
        <div className="lg:col-span-3">
          <Card title="New submission">
            <SubmissionForm
              onAnalyzed={(res) => {
                setReloadKey((k) => k + 1);
                navigate(`/submissions/${res.id}`);
              }}
            />
          </Card>
        </div>

        <div className="lg:col-span-2">
          <div className="card overflow-hidden">
            <div className="relative bg-watermelon px-6 py-7 text-white">
              <div className="absolute -right-6 -top-10 h-28 w-28 rounded-full bg-white/10" />
              <div className="absolute -bottom-12 -left-4 h-24 w-24 rounded-full bg-white/10" />
              <h2 className="relative text-2xl font-extrabold leading-tight">
                Is this the student’s work — or the internet’s?
              </h2>
              <p className="relative mt-2 text-sm text-white/90">
                Get a plagiarism %, an AI-generated %, an overall integrity score, highlighted
                suspicious text, matched sources, and a writing fingerprint compared against the
                student’s history.
              </p>
            </div>
            <ul className="space-y-3 p-6 text-sm">
              {FEATURES.map(([t, d]) => (
                <li key={t} className="flex gap-3">
                  <span className="mt-0.5 h-2.5 w-2.5 flex-none rounded-full bg-watermelon" />
                  <span>
                    <span className="font-semibold">{t}.</span>{" "}
                    <span className="text-muted">{d}</span>
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      <RecentSubmissions reloadKey={reloadKey} />
    </div>
  );
}
