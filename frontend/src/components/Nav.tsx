import { NavLink } from "react-router-dom";
import { useEffect, useState } from "react";
import { getHealth, type HealthResponse } from "../api";

export function Nav() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [unreachable, setUnreachable] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    const poll = async () => {
      try {
        const h = await getHealth();
        if (cancelled) return;
        setHealth(h);
        setUnreachable(false);
        if (h.status !== "ok") timer = window.setTimeout(poll, 2500);
      } catch {
        if (cancelled) return;
        setUnreachable(true);
        timer = window.setTimeout(poll, 4000);
      }
    };
    poll();

    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, []);

  const status = unreachable
    ? { cls: "down", text: "backend unreachable" }
    : health?.status === "ok"
      ? { cls: "up", text: `${health.device.toUpperCase()} · ready` }
      : { cls: "loading", text: "loading models…" };

  return (
    <header className="nav">
      <div className="nav-inner">
        <NavLink to="/" className="brand">
          <span className="brand-mark" aria-hidden="true" />
          Speculative Decoding
        </NavLink>
        <nav className="nav-links">
          <NavLink to="/" end>
            Home
          </NavLink>
          <NavLink to="/theory">Theory</NavLink>
          <NavLink to="/playground">Playground</NavLink>
        </nav>
        <span className={`status status-${status.cls}`}>
          <span className="status-dot" aria-hidden="true" />
          {status.text}
        </span>
      </div>
    </header>
  );
}
