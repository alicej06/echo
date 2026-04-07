"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import { Radio, StopCircle, Play, Trash2, Volume2 } from "lucide-react";
import { useMyoWs } from "@/hooks/use-myo-ws";

const CONFIDENCE_THRESHOLDS = [
  { label: "High", value: 0.9, color: "#22c55e" },
  { label: "Med", value: 0.75, color: "#eab308" },
  { label: "Low", value: 0.6, color: "#f97316" },
];

function ConfidenceBar({ confidence }: { confidence: number }) {
  const active = [...CONFIDENCE_THRESHOLDS]
    .reverse()
    .find((t) => confidence >= t.value);
  return (
    <div className="flex flex-col gap-1.5">
      {CONFIDENCE_THRESHOLDS.map((t) => {
        const hit = confidence >= t.value;
        return (
          <div key={t.label} className="flex items-center gap-2">
            <span className="text-xs w-8" style={{ color: "#52525b" }}>
              {t.label}
            </span>
            <div
              className="flex-1 h-1.5 rounded-full"
              style={{ backgroundColor: "rgba(255,255,255,0.06)" }}
            >
              <div
                className="h-full rounded-full transition-all duration-300"
                style={{
                  width: hit ? `${Math.min(confidence * 100, 100)}%` : "0%",
                  backgroundColor: hit ? t.color : "transparent",
                  boxShadow: hit ? `0 0 6px ${t.color}60` : "none",
                }}
              />
            </div>
            <span
              className="text-xs w-8 text-right"
              style={{ color: hit ? t.color : "#3f3f46" }}
            >
              {t.value * 100}%
            </span>
          </div>
        );
      })}
      {active && (
        <p className="text-xs mt-1" style={{ color: active.color }}>
          {(confidence * 100).toFixed(0)}% confidence
        </p>
      )}
    </div>
  );
}

export default function TranslatePage() {
  const {
    status,
    currentLetter,
    confidence,
    letterStream,
    sentence,
    connect,
    disconnect,
    startDemo,
    clearStream,
  } = useMyoWs();
  const [wsUrl, setWsUrl] = useState("ws://localhost:8765");
  const [letterKey, setLetterKey] = useState(0);
  const prevLetterRef = useRef("");
  const prevSentenceRef = useRef("");
  const streamEndRef = useRef<HTMLDivElement>(null);
  const sessionStartRef = useRef<number>(Date.now());

  // Save session to localStorage on disconnect
  useEffect(() => {
    if (status === "connected" || status === "demo") {
      sessionStartRef.current = Date.now();
    }
    if (status === "disconnected" && letterStream.length > 0) {
      try {
        const raw = localStorage.getItem("maia_sessions");
        const sessions = raw ? (JSON.parse(raw) as object[]) : [];
        sessions.push({
          id: Date.now().toString(),
          date: new Date().toISOString(),
          letters: letterStream,
          sentences: sentence ? [sentence] : [],
          duration: Date.now() - sessionStartRef.current,
        });
        localStorage.setItem(
          "maia_sessions",
          JSON.stringify(sessions.slice(-50)),
        );
      } catch {
        // ignore
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  // Trigger animation on new letter
  useEffect(() => {
    if (currentLetter && currentLetter !== prevLetterRef.current) {
      setLetterKey((k) => k + 1);
      prevLetterRef.current = currentLetter;
    }
  }, [currentLetter]);

  // TTS on new sentence
  useEffect(() => {
    if (!sentence || sentence === prevSentenceRef.current) return;
    prevSentenceRef.current = sentence;
    if (typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.cancel();
      const utt = new SpeechSynthesisUtterance(sentence);
      utt.rate = 0.95;
      window.speechSynthesis.speak(utt);
    }
  }, [sentence]);

  // Auto-scroll stream
  useEffect(() => {
    streamEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [letterStream.length]);

  const handleConnect = useCallback(() => {
    connect(wsUrl);
  }, [connect, wsUrl]);

  const isActive = status === "connected" || status === "demo";
  const statusColor = {
    disconnected: "#52525b",
    connecting: "#eab308",
    connected: "#22c55e",
    demo: "#06b6d4",
  }[status];

  return (
    <main
      className="min-h-screen pt-16 pb-20 px-4"
      style={{ backgroundColor: "#0a0a0a" }}
    >
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="py-8 flex items-center justify-between flex-wrap gap-4">
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">
              Live Translation
            </h1>
            <p className="text-sm mt-1" style={{ color: "#52525b" }}>
              Real-time ASL to text via sEMG
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: statusColor }}
            />
            <span className="text-sm capitalize" style={{ color: statusColor }}>
              {status}
            </span>
            {status === "demo" && (
              <span
                className="text-xs px-2 py-0.5 rounded-full"
                style={{
                  background: "rgba(6,182,212,0.1)",
                  color: "#22d3ee",
                  border: "1px solid rgba(6,182,212,0.2)",
                }}
              >
                Demo
              </span>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Left: Letter display */}
          <div className="lg:col-span-1 flex flex-col gap-4">
            {/* Big letter */}
            <div
              className="rounded-2xl p-8 flex flex-col items-center justify-center aspect-square"
              style={{
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.08)",
              }}
            >
              {currentLetter ? (
                <div
                  key={`${currentLetter}-${letterKey}`}
                  className="letter-pop"
                >
                  <span
                    className="text-9xl font-bold leading-none select-none"
                    style={{
                      backgroundImage:
                        "linear-gradient(135deg, #06b6d4, #0ea5e9)",
                      WebkitBackgroundClip: "text",
                      WebkitTextFillColor: "transparent",
                    }}
                  >
                    {currentLetter}
                  </span>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-3">
                  <div
                    className="w-16 h-16 rounded-2xl flex items-center justify-center"
                    style={{ background: "rgba(255,255,255,0.04)" }}
                  >
                    <Radio className="w-7 h-7" style={{ color: "#3f3f46" }} />
                  </div>
                  <p className="text-sm" style={{ color: "#3f3f46" }}>
                    {status === "connecting"
                      ? "Connecting..."
                      : "Waiting for signal"}
                  </p>
                </div>
              )}
            </div>

            {/* Confidence */}
            <div
              className="rounded-2xl p-5"
              style={{
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.08)",
              }}
            >
              <p
                className="text-xs font-medium uppercase tracking-widest mb-3"
                style={{ color: "#52525b" }}
              >
                Confidence
              </p>
              <ConfidenceBar confidence={confidence} />
            </div>
          </div>

          {/* Right: Sentence + Stream */}
          <div className="lg:col-span-2 flex flex-col gap-4">
            {/* Claude output */}
            <div
              className="rounded-2xl p-6 min-h-32 flex flex-col"
              style={{
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.08)",
              }}
            >
              <div className="flex items-center gap-2 mb-4">
                <div
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: "#06b6d4" }}
                />
                <p
                  className="text-xs font-medium uppercase tracking-widest"
                  style={{ color: "#52525b" }}
                >
                  Claude Haiku output
                </p>
                {sentence && (
                  <button
                    onClick={() => {
                      window.speechSynthesis?.cancel();
                      const utt = new SpeechSynthesisUtterance(sentence);
                      window.speechSynthesis?.speak(utt);
                    }}
                    className="ml-auto p-1.5 rounded-lg transition-all duration-200 cursor-pointer"
                    style={{ color: "#52525b" }}
                    aria-label="Speak sentence"
                  >
                    <Volume2 className="w-4 h-4" />
                  </button>
                )}
              </div>
              {sentence ? (
                <p className="text-xl font-medium text-white leading-relaxed">
                  {sentence}
                </p>
              ) : (
                <p className="text-sm" style={{ color: "#3f3f46" }}>
                  Reconstructed sentence will appear here once enough letters
                  are recognized.
                </p>
              )}
            </div>

            {/* Letter stream */}
            <div
              className="rounded-2xl p-5 flex-1"
              style={{
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.08)",
              }}
            >
              <div className="flex items-center justify-between mb-3">
                <p
                  className="text-xs font-medium uppercase tracking-widest"
                  style={{ color: "#52525b" }}
                >
                  Letter stream ({letterStream.length})
                </p>
                {letterStream.length > 0 && (
                  <button
                    onClick={clearStream}
                    className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-lg transition-all duration-200 cursor-pointer"
                    style={{
                      color: "#52525b",
                      background: "rgba(255,255,255,0.04)",
                    }}
                    aria-label="Clear stream"
                  >
                    <Trash2 className="w-3 h-3" />
                    Clear
                  </button>
                )}
              </div>
              <div className="flex flex-wrap gap-1 max-h-40 overflow-y-auto">
                {letterStream.map((l, i) => (
                  <span
                    key={i}
                    className="inline-flex items-center justify-center w-7 h-7 rounded-lg text-xs font-bold transition-all duration-100"
                    style={{
                      backgroundColor:
                        i === letterStream.length - 1
                          ? "rgba(6,182,212,0.2)"
                          : "rgba(255,255,255,0.06)",
                      color:
                        i === letterStream.length - 1 ? "#22d3ee" : "#71717a",
                      border:
                        i === letterStream.length - 1
                          ? "1px solid rgba(6,182,212,0.3)"
                          : "1px solid transparent",
                    }}
                  >
                    {l}
                  </span>
                ))}
                <div ref={streamEndRef} />
              </div>
            </div>

            {/* Controls */}
            <div
              className="rounded-2xl p-5"
              style={{
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.08)",
              }}
            >
              <div className="flex gap-2 mb-4 flex-wrap">
                {!isActive ? (
                  <>
                    <div className="flex-1 min-w-0">
                      <input
                        type="text"
                        value={wsUrl}
                        onChange={(e) => setWsUrl(e.target.value)}
                        placeholder="ws://localhost:8765"
                        className="w-full px-3 py-2 rounded-xl text-sm transition-all duration-200 outline-none"
                        style={{
                          backgroundColor: "rgba(255,255,255,0.06)",
                          border: "1px solid rgba(255,255,255,0.1)",
                          color: "#e4e4e7",
                        }}
                        aria-label="WebSocket URL"
                      />
                    </div>
                    <button
                      onClick={handleConnect}
                      className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium text-white transition-all duration-200 cursor-pointer"
                      style={{ background: "rgba(6,182,212,0.7)" }}
                    >
                      <Radio className="w-4 h-4" />
                      Connect
                    </button>
                    <button
                      onClick={startDemo}
                      className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 cursor-pointer"
                      style={{
                        background: "rgba(255,255,255,0.06)",
                        border: "1px solid rgba(255,255,255,0.1)",
                        color: "#a1a1aa",
                      }}
                    >
                      <Play className="w-4 h-4" />
                      Demo
                    </button>
                  </>
                ) : (
                  <button
                    onClick={disconnect}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 cursor-pointer"
                    style={{
                      background: "rgba(239,68,68,0.1)",
                      border: "1px solid rgba(239,68,68,0.2)",
                      color: "#f87171",
                    }}
                  >
                    <StopCircle className="w-4 h-4" />
                    Disconnect
                  </button>
                )}
              </div>

              {!isActive && (
                <div
                  className="rounded-xl p-4 text-sm font-mono"
                  style={{
                    background: "rgba(0,0,0,0.4)",
                    border: "1px solid rgba(255,255,255,0.07)",
                    color: "#22c55e",
                  }}
                >
                  <span style={{ color: "#52525b" }}>$ </span>
                  python live_translate.py --ws-port 8765
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
