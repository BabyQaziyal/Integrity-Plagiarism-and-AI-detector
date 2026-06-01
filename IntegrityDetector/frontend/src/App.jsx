import { NavLink, Route, Routes } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Students from "./pages/Students";
import SubmissionResult from "./pages/SubmissionResult";
import ApiStatusBanner from "./components/ApiStatusBanner";
import { useHealth } from "./hooks/useHealth";

function Brand() {
  return (
    <div className="flex items-center gap-2.5">
      <span className="grid h-9 w-9 place-items-center rounded-xl bg-watermelon text-lg font-black text-white shadow-soft">
        ◑
      </span>
      <div className="leading-tight">
        <div className="text-sm font-extrabold tracking-tight">Integrity Detector</div>
        <div className="text-[11px] text-muted">Plagiarism &amp; AI-content analysis</div>
      </div>
    </div>
  );
}

function HealthPill({ status, info }) {
  const map = {
    checking: { dot: "bg-lemon-300", text: "Checking…", ring: "border-black/10 text-muted" },
    online: {
      dot: "bg-good",
      text: info?.cuda ? "GPU ready" : "Online (CPU)",
      ring: "border-good/30 text-good",
    },
    offline: { dot: "bg-watermelon", text: "API offline", ring: "border-watermelon/30 text-watermelon" },
  };
  const s = map[status] || map.checking;
  return (
    <span
      className={`hidden items-center gap-2 rounded-full border bg-white/70 px-3 py-1.5 text-xs font-semibold sm:inline-flex ${s.ring}`}
      title={status === "online" ? info?.device : "Backend health"}
    >
      <span className={`h-2 w-2 rounded-full ${s.dot} ${status !== "offline" ? "animate-pulse" : ""}`} />
      {s.text}
    </span>
  );
}

const navClass = ({ isActive }) =>
  `rounded-full px-4 py-2 text-sm font-semibold transition ${
    isActive ? "bg-watermelon text-white shadow-soft" : "text-ink hover:bg-lemon-100"
  }`;

export default function App() {
  const { status, info, refresh } = useHealth();

  return (
    <div className="flex min-h-full flex-col">
      <header className="sticky top-0 z-20 border-b border-black/5 bg-lemon-50/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
          <Brand />
          <div className="flex items-center gap-2">
            <nav className="flex items-center gap-1.5">
              <NavLink to="/" end className={navClass}>
                Analyze
              </NavLink>
              <NavLink to="/students" className={navClass}>
                Students
              </NavLink>
            </nav>
            <HealthPill status={status} info={info} />
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6">
        <ApiStatusBanner status={status} onRetry={refresh} />
        <Routes>
          <Route path="/" element={<Dashboard health={status} />} />
          <Route path="/students" element={<Students />} />
          <Route path="/submissions/:id" element={<SubmissionResult />} />
        </Routes>
      </main>

      <footer className="mx-auto w-full max-w-6xl px-4 py-6 text-center text-xs text-muted">
        Decision-support evidence — scores are probabilistic; corroborate before any action.
      </footer>
    </div>
  );
}
