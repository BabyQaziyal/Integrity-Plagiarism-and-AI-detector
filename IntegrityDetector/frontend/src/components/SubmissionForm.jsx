import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import { Spinner } from "./ui";

const MIN_CHARS = 20;

export default function SubmissionForm({ onAnalyzed }) {
  const [students, setStudents] = useState([]);
  const [studentId, setStudentId] = useState("");
  const [newName, setNewName] = useState("");
  const [title, setTitle] = useState("");
  const [course, setCourse] = useState("");
  const [mode, setMode] = useState("text"); // 'text' | 'file'
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const fileInput = useRef(null);

  const loadStudents = () =>
    api.listStudents().then(setStudents).catch(() => {});
  useEffect(() => {
    loadStudents();
  }, []);

  async function ensureStudent() {
    if (studentId) return Number(studentId);
    const name = newName.trim();
    if (!name) throw new Error("Pick an existing student or enter a new name.");
    const s = await api.createStudent({ name });
    await loadStudents();
    setStudentId(String(s.id));
    return s.id;
  }

  function pickFile(f) {
    if (!f) return;
    setFile(f);
    if (!title) setTitle(f.name.replace(/\.[^.]+$/, ""));
  }

  async function submit(e) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const sid = await ensureStudent();
      let result;
      if (mode === "file") {
        if (!file) throw new Error("Choose a file (.pdf, .docx, .txt).");
        const fd = new FormData();
        fd.append("file", file);
        fd.append("student_id", sid);
        if (title) fd.append("title", title);
        if (course) fd.append("course", course);
        result = await api.submitFile(fd);
      } else {
        if (text.trim().length < MIN_CHARS)
          throw new Error("Paste at least a short paragraph (20+ characters).");
        result = await api.submitText({ student_id: sid, text, title: title || "Untitled", course });
      }
      onAnalyzed?.(result);
    } catch (e) {
      setError(e.offline ? "Backend not reachable — start it with `python scripts/run_api.py`." : e.message);
    } finally {
      setBusy(false);
    }
  }

  const charCount = text.trim().length;

  return (
    <form onSubmit={submit} className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2">
        <div>
          <label className="label">Student</label>
          <select className="input" value={studentId} onChange={(e) => setStudentId(e.target.value)}>
            <option value="">— new student —</option>
            {students.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name} ({s.submission_count})
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="label">{studentId ? "Selected student" : "New student name"}</label>
          <input
            className="input"
            placeholder="e.g. Ada Lovelace"
            value={studentId ? students.find((s) => String(s.id) === studentId)?.name || "" : newName}
            disabled={!!studentId}
            onChange={(e) => setNewName(e.target.value)}
          />
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <div>
          <label className="label">Title</label>
          <input
            className="input"
            placeholder="Assignment title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
          />
        </div>
        <div>
          <label className="label">Course (optional)</label>
          <input
            className="input"
            placeholder="e.g. ENG-204"
            value={course}
            onChange={(e) => setCourse(e.target.value)}
          />
        </div>
      </div>

      <div className="inline-flex rounded-full bg-lemon-100 p-1">
        <button
          type="button"
          onClick={() => setMode("text")}
          className={`rounded-full px-4 py-1.5 text-sm font-semibold transition ${
            mode === "text" ? "bg-white text-ink shadow-card" : "text-muted hover:text-ink"
          }`}
        >
          Paste text
        </button>
        <button
          type="button"
          onClick={() => setMode("file")}
          className={`rounded-full px-4 py-1.5 text-sm font-semibold transition ${
            mode === "file" ? "bg-white text-ink shadow-card" : "text-muted hover:text-ink"
          }`}
        >
          Upload file
        </button>
      </div>

      {mode === "text" ? (
        <div>
          <textarea
            className="input min-h-[12rem] font-mono text-[13px] leading-6"
            placeholder="Paste the student's submission here…"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <div className="mt-1 text-right text-xs text-muted">
            {charCount < MIN_CHARS ? (
              <span className="text-watermelon-600">{MIN_CHARS - charCount} more characters needed</span>
            ) : (
              <span>{charCount.toLocaleString()} characters</span>
            )}
          </div>
        </div>
      ) : (
        <div>
          <div
            role="button"
            tabIndex={0}
            onClick={() => fileInput.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              pickFile(e.dataTransfer.files?.[0]);
            }}
            className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-10 text-center transition ${
              dragOver ? "border-watermelon bg-watermelon-50" : "border-black/15 hover:border-watermelon/50 hover:bg-lemon-50"
            }`}
          >
            <div className="text-2xl">{file ? "📄" : "⬆️"}</div>
            {file ? (
              <div className="mt-1">
                <div className="font-semibold">{file.name}</div>
                <div className="text-xs text-muted">{(file.size / 1024).toFixed(0)} KB · click to replace</div>
              </div>
            ) : (
              <div className="mt-1 text-sm text-muted">
                <span className="font-semibold text-ink">Drop a file</span> or click to browse
                <div className="text-xs">PDF, DOCX, or TXT · up to 25 MB</div>
              </div>
            )}
          </div>
          <input
            ref={fileInput}
            className="hidden"
            type="file"
            accept=".pdf,.docx,.txt,.md"
            onChange={(e) => pickFile(e.target.files?.[0])}
          />
        </div>
      )}

      {error && (
        <div className="rounded-xl bg-watermelon-50 px-4 py-2.5 text-sm text-watermelon-700">{error}</div>
      )}

      <div className="flex flex-wrap items-center gap-4">
        <button className="btn-primary" disabled={busy}>
          {busy ? "Analyzing…" : "Analyze submission"}
        </button>
        {busy && <Spinner label="Running detectors — the first run loads models (~30s)…" />}
      </div>
    </form>
  );
}
