import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";

// Polls /api/health so the whole app knows whether the backend is reachable.
// status: "checking" | "online" | "offline"
export function useHealth(intervalMs = 8000) {
  const [status, setStatus] = useState("checking");
  const [info, setInfo] = useState(null);
  const timer = useRef(null);

  const check = useCallback(async () => {
    try {
      const h = await api.health();
      setInfo(h);
      setStatus("online");
    } catch (_) {
      setStatus("offline");
    }
  }, []);

  useEffect(() => {
    check();
    timer.current = setInterval(check, intervalMs);
    return () => clearInterval(timer.current);
  }, [check, intervalMs]);

  return { status, info, refresh: check };
}
