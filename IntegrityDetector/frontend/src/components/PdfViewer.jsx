import { useEffect, useRef, useState } from "react";
import * as pdfjsLib from "pdfjs-dist";
import workerSrc from "pdfjs-dist/build/pdf.worker.min.mjs?url";

pdfjsLib.GlobalWorkerOptions.workerSrc = workerSrc;

// Minimal PDF.js viewer: renders every page to a canvas in a scroll container.
export default function PdfViewer({ url }) {
  const containerRef = useRef(null);
  const [error, setError] = useState(null);
  const [pages, setPages] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const container = containerRef.current;
    if (!url || !container) return;
    container.innerHTML = "";

    (async () => {
      try {
        const pdf = await pdfjsLib.getDocument(url).promise;
        if (cancelled) return;
        setPages(pdf.numPages);
        for (let n = 1; n <= pdf.numPages; n++) {
          const page = await pdf.getPage(n);
          const viewport = page.getViewport({ scale: 1.25 });
          const canvas = document.createElement("canvas");
          canvas.className = "mx-auto mb-3 rounded-lg shadow-card";
          canvas.width = viewport.width;
          canvas.height = viewport.height;
          container.appendChild(canvas);
          await page.render({ canvasContext: canvas.getContext("2d"), viewport }).promise;
          if (cancelled) return;
        }
      } catch (e) {
        setError(e.message || "Could not load PDF");
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [url]);

  if (error) return <div className="text-sm text-watermelon">PDF preview unavailable: {error}</div>;
  return (
    <div>
      {pages > 0 && <div className="mb-2 text-xs text-muted">{pages} page(s)</div>}
      <div ref={containerRef} className="max-h-[36rem] overflow-auto rounded-xl bg-lemon-50 p-3" />
    </div>
  );
}
