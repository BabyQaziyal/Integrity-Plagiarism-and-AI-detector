// A clear, actionable banner when the backend is unreachable — replaces the
// cryptic "ECONNREFUSED / Failed to fetch" errors users were seeing.
export default function ApiStatusBanner({ status, onRetry }) {
  if (status !== "offline") return null;
  return (
    <div className="mb-5 flex flex-col gap-2 rounded-xl2 border border-watermelon/25 bg-watermelon-50 px-4 py-3 text-sm sm:flex-row sm:items-center sm:justify-between">
      <div className="flex items-start gap-3">
        <span className="mt-0.5 grid h-6 w-6 flex-none place-items-center rounded-full bg-watermelon text-white">
          !
        </span>
        <div>
          <div className="font-semibold text-watermelon-700">Backend not reachable</div>
          <div className="text-ink/70">
            Start the API in a terminal:{" "}
            <code className="rounded bg-white px-1.5 py-0.5 font-mono text-[12px] text-watermelon-700">
              python scripts/run_api.py
            </code>{" "}
            — or run both together with{" "}
            <code className="rounded bg-white px-1.5 py-0.5 font-mono text-[12px] text-watermelon-700">
              .\dev.ps1
            </code>
            .
          </div>
        </div>
      </div>
      <button onClick={onRetry} className="btn-ghost shrink-0 self-start sm:self-auto">
        Retry
      </button>
    </div>
  );
}
