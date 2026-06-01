import { riskColor } from "../theme";

export function Card({ title, children, className = "", action, subtitle }) {
  return (
    <section className={`card p-5 ${className}`}>
      {(title || action) && (
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            {title && (
              <h3 className="text-sm font-bold uppercase tracking-wide text-muted">{title}</h3>
            )}
            {subtitle && <p className="mt-0.5 text-xs text-muted/80">{subtitle}</p>}
          </div>
          {action}
        </div>
      )}
      {children}
    </section>
  );
}

export function Bar({ value = 0, color = "#f0485f", track = "rgba(0,0,0,0.06)", height = 8 }) {
  const pct = Math.max(0, Math.min(100, value));
  return (
    <div className="w-full overflow-hidden rounded-full" style={{ background: track, height }}>
      <div
        className="h-full rounded-full transition-all duration-700 ease-out"
        style={{ width: `${pct}%`, background: color }}
      />
    </div>
  );
}

export function ScoreCard({ label, value, suffix = "", color, big = false, hint }) {
  return (
    <div className="card flex flex-col items-center justify-center p-5 text-center transition hover:shadow-soft">
      <div
        className={`font-extrabold leading-none ${big ? "text-5xl" : "text-4xl"}`}
        style={{ color: color || "#2b2b2b" }}
      >
        {value}
        <span className="text-xl">{suffix}</span>
      </div>
      <div className="mt-2 text-xs font-semibold uppercase tracking-wide text-muted">{label}</div>
      {hint && <div className="mt-1 text-[11px] text-muted/80">{hint}</div>}
    </div>
  );
}

export function Gauge({ score, size = 148, label = "Integrity" }) {
  const color = riskColor(score);
  const stroke = 13;
  const r = (size - stroke) / 2 - 2;
  const c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(100, score ?? 0));
  return (
    <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(0,0,0,0.06)" strokeWidth={stroke} />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c - (pct / 100) * c}
          style={{ transition: "stroke-dashoffset .8s ease-out" }}
        />
      </svg>
      <div className="absolute text-center">
        <div className="text-3xl font-extrabold tabular-nums" style={{ color }}>
          {Math.round(pct)}
        </div>
        <div className="text-[10px] font-semibold uppercase tracking-wide text-muted">{label}</div>
      </div>
    </div>
  );
}

export function Badge({ children, bg = "#eee", fg = "#2b2b2b", dot = false }) {
  return (
    <span className="chip" style={{ background: bg, color: fg }}>
      {dot && (
        <span className="mr-1.5 inline-block h-1.5 w-1.5 rounded-full" style={{ background: fg }} />
      )}
      {children}
    </span>
  );
}

export function Spinner({ label = "Working…" }) {
  return (
    <div className="flex items-center gap-3 text-sm text-muted">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-watermelon border-t-transparent" />
      {label}
    </div>
  );
}

export function Empty({ children, icon = "✦" }) {
  return (
    <div className="rounded-xl border border-dashed border-black/10 p-8 text-center">
      <div className="mx-auto mb-2 grid h-10 w-10 place-items-center rounded-full bg-lemon-100 text-lg text-watermelon">
        {icon}
      </div>
      <div className="text-sm text-muted">{children}</div>
    </div>
  );
}

export function ErrorState({ title = "Something went wrong", message, onRetry }) {
  return (
    <div className="rounded-xl border border-watermelon/20 bg-watermelon-50 p-6 text-center">
      <div className="mx-auto mb-2 grid h-10 w-10 place-items-center rounded-full bg-watermelon/15 text-lg">
        ⚠️
      </div>
      <div className="font-semibold text-watermelon-700">{title}</div>
      {message && <div className="mt-1 text-sm text-ink/70">{message}</div>}
      {onRetry && (
        <button onClick={onRetry} className="btn-ghost mt-4">
          Try again
        </button>
      )}
    </div>
  );
}

export function Skeleton({ className = "" }) {
  return <div className={`skeleton rounded-lg ${className}`} />;
}

export function StatPill({ label, value, color }) {
  return (
    <div className="rounded-xl border border-black/5 bg-white/60 px-3 py-2">
      <div className="text-[11px] uppercase tracking-wide text-muted">{label}</div>
      <div className="text-lg font-bold tabular-nums" style={{ color }}>
        {value}
      </div>
    </div>
  );
}
