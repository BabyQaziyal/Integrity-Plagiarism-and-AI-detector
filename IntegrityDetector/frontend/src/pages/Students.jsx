import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { riskColor, verdictTone } from "../theme";
import { Badge, Card, Empty, ErrorState, Skeleton, StatPill } from "../components/ui";

export default function Students() {
  const [students, setStudents] = useState(null);
  const [error, setError] = useState(null);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(null);
  const [detail, setDetail] = useState(null);

  const load = () => {
    setError(null);
    setStudents(null);
    api.listStudents().then(setStudents).catch(setError);
  };
  useEffect(load, []);

  useEffect(() => {
    if (active == null) return;
    setDetail(null);
    api.getStudent(active).then(setDetail).catch(() => setDetail({ submissions: [] }));
  }, [active]);

  const filtered = useMemo(
    () => (students || []).filter((s) => s.name.toLowerCase().includes(query.toLowerCase())),
    [students, query]
  );

  const stats = useMemo(() => {
    const subs = detail?.submissions || [];
    const scored = subs.filter((s) => s.summary?.integrity_score != null);
    const avg = scored.length
      ? Math.round(scored.reduce((a, s) => a + s.summary.integrity_score, 0) / scored.length)
      : null;
    return { count: subs.length, avg };
  }, [detail]);

  if (error)
    return (
      <ErrorState
        title="Couldn’t load students"
        message={error.offline ? "The backend isn’t running yet." : error.message}
        onRetry={load}
      />
    );

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
      <Card title="Students" className="lg:col-span-1">
        <input
          className="input mb-3"
          placeholder="Search students…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        {!students ? (
          <div className="space-y-2">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-9 w-full" />
            ))}
          </div>
        ) : filtered.length === 0 ? (
          <Empty icon="🧑‍🎓">
            {students.length === 0 ? "No students yet. Add one from Analyze." : "No matches."}
          </Empty>
        ) : (
          <ul className="space-y-1">
            {filtered.map((s) => (
              <li key={s.id}>
                <button
                  onClick={() => setActive(s.id)}
                  className={`flex w-full items-center justify-between rounded-xl px-3 py-2 text-left text-sm transition ${
                    active === s.id ? "bg-watermelon-50 ring-1 ring-watermelon/20" : "hover:bg-lemon-50"
                  }`}
                >
                  <span className="font-medium">{s.name}</span>
                  <span className="text-xs text-muted">{s.submission_count} subs</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </Card>

      <div className="lg:col-span-2">
        <Card title={detail ? detail.name : "Submission history"}>
          {active == null ? (
            <Empty icon="📚">Select a student to see their submission history.</Empty>
          ) : !detail ? (
            <div className="space-y-2">
              {Array.from({ length: 3 }).map((_, i) => (
                <Skeleton key={i} className="h-16 w-full" />
              ))}
            </div>
          ) : (
            <>
              <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-3">
                <StatPill label="Submissions" value={stats.count} />
                <StatPill
                  label="Avg integrity"
                  value={stats.avg ?? "—"}
                  color={stats.avg != null ? riskColor(stats.avg) : undefined}
                />
                {detail.email && <StatPill label="Email" value={detail.email} />}
              </div>

              {detail.submissions?.length ? (
                <ul className="space-y-2">
                  {detail.submissions.map((sub) => {
                    const sum = sub.summary || {};
                    const tone = verdictTone(sum.verdict);
                    return (
                      <li key={sub.id}>
                        <Link
                          to={`/submissions/${sub.id}`}
                          className="flex items-center justify-between gap-3 rounded-xl border border-black/5 px-4 py-3 transition hover:border-watermelon/40 hover:shadow-card"
                        >
                          <div className="min-w-0">
                            <div className="truncate font-medium">{sub.title}</div>
                            <div className="text-xs text-muted">
                              {new Date(sub.created_at).toLocaleString()} · {sub.word_count} words
                            </div>
                          </div>
                          <div className="flex flex-none items-center gap-3 text-sm">
                            <span
                              title="integrity"
                              className="text-lg font-bold tabular-nums"
                              style={{ color: riskColor(sum.integrity_score) }}
                            >
                              {sum.integrity_score != null ? Math.round(sum.integrity_score) : "—"}
                            </span>
                            <span className="hidden text-muted sm:inline">
                              AI {Math.round(sum.ai_percent || 0)}%
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
              ) : (
                <Empty icon="📝">No submissions for this student yet.</Empty>
              )}
            </>
          )}
        </Card>
      </div>
    </div>
  );
}
