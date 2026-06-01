// Thin fetch wrapper around the Flask API. In dev, Vite proxies /api -> backend.
const BASE = import.meta.env.VITE_API_BASE || "";

export class ApiError extends Error {
  constructor(message, { status = 0, offline = false } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.offline = offline;
  }
}

async function req(path, opts = {}) {
  let res;
  try {
    res = await fetch(`${BASE}/api${path}`, opts);
  } catch (e) {
    // Network-level failure (backend down, DNS, CORS) — never a parseable body.
    throw new ApiError(
      "Can’t reach the backend. Is the API running? (python scripts/run_api.py)",
      { offline: true }
    );
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = j.error || detail;
    } catch (_) {
      /* non-JSON error body */
    }
    // 502/503/504 from the dev proxy mean the backend isn't answering.
    const offline = res.status === 502 || res.status === 503 || res.status === 504;
    throw new ApiError(detail || `Request failed (${res.status})`, {
      status: res.status,
      offline,
    });
  }
  return res.json();
}

const json = (method, body) => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => req("/health"),

  listStudents: () => req("/students"),
  getStudent: (id) => req(`/students/${id}`),
  createStudent: (body) => req("/students", json("POST", body)),

  listSubmissions: (studentId) =>
    req(`/submissions${studentId ? `?student_id=${studentId}` : ""}`),
  getSubmission: (id) => req(`/submissions/${id}`),
  submitText: (body) => req("/submissions", json("POST", body)),
  submitFile: (formData) => req("/submissions", { method: "POST", body: formData }),

  reportUrl: (id) => `${BASE}/api/submissions/${id}/report`,
  fileUrl: (id) => `${BASE}/api/submissions/${id}/file`,
};
