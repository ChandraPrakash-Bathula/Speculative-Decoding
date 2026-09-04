import { Routes, Route, Navigate } from "react-router-dom";
import { Nav } from "./components/Nav";
import { Home } from "./pages/Home";
import { Theory } from "./pages/Theory";
import { Playground } from "./pages/Playground";

export function App() {
  return (
    <>
      <Nav />
      <main className="main">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/theory" element={<Theory />} />
          <Route path="/playground" element={<Playground />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      <footer className="footer">
        <span>
          Qwen2.5-0.5B-Instruct drafting for Qwen2.5-1.5B-Instruct via Transformers{" "}
          <code>assistant_model</code>. All figures are measured at run time.
        </span>
      </footer>
    </>
  );
}
