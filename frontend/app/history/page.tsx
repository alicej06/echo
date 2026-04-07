"use client";
import { useState, useEffect } from "react";
import {
  History,
  ChevronDown,
  ChevronUp,
  Trash2,
  Clock,
  Hash,
} from "lucide-react";

interface SessionRecord {
  id: string;
  date: string;
  letters: string[];
  sentences: string[];
  duration: number;
}

function fmtDuration(ms: number) {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  if (m === 0) return `${s}s`;
  return `${m}m ${s % 60}s`;
}

export default function HistoryPage() {
  const [sessions, setSessions] = useState<SessionRecord[]>([]);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  useEffect(() => {
    try {
      const raw = localStorage.getItem("maia_sessions");
      if (raw) setSessions(JSON.parse(raw) as SessionRecord[]);
    } catch {
      // ignore
    }
  }, []);

  const clearAll = () => {
    localStorage.removeItem("maia_sessions");
    setSessions([]);
  };

  const toggle = (id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  return (
    <main
      className="min-h-screen pt-16 pb-20 px-4"
      style={{ backgroundColor: "#0a0a0a" }}
    >
      <div className="max-w-3xl mx-auto">
        <div className="py-8 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">
              History
            </h1>
            <p className="text-sm mt-1" style={{ color: "#52525b" }}>
              {sessions.length === 0
                ? "No sessions recorded yet"
                : `${sessions.length} session${sessions.length === 1 ? "" : "s"}`}
            </p>
          </div>
          {sessions.length > 0 && (
            <button
              onClick={clearAll}
              className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 cursor-pointer"
              style={{
                background: "rgba(239,68,68,0.08)",
                border: "1px solid rgba(239,68,68,0.15)",
                color: "#f87171",
              }}
            >
              <Trash2 className="w-4 h-4" />
              Clear all
            </button>
          )}
        </div>

        {sessions.length === 0 ? (
          <div
            className="rounded-2xl p-16 flex flex-col items-center gap-4 text-center"
            style={{
              background: "rgba(255,255,255,0.03)",
              border: "1px solid rgba(255,255,255,0.07)",
            }}
          >
            <div
              className="w-14 h-14 rounded-2xl flex items-center justify-center"
              style={{ background: "rgba(255,255,255,0.05)" }}
            >
              <History className="w-6 h-6" style={{ color: "#3f3f46" }} />
            </div>
            <div>
              <p className="text-white font-medium mb-1">No sessions yet</p>
              <p className="text-sm max-w-xs" style={{ color: "#52525b" }}>
                Start a live translation session and your letter streams will be
                saved here automatically.
              </p>
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            {[...sessions].reverse().map((s) => {
              const isOpen = expanded.has(s.id);
              return (
                <div
                  key={s.id}
                  className="rounded-2xl overflow-hidden transition-all duration-200"
                  style={{
                    background: "rgba(255,255,255,0.04)",
                    border: "1px solid rgba(255,255,255,0.08)",
                  }}
                >
                  <button
                    onClick={() => toggle(s.id)}
                    className="w-full px-5 py-4 flex items-center justify-between text-left cursor-pointer transition-all duration-200 hover:bg-white/5"
                  >
                    <div className="flex items-center gap-4">
                      <div className="flex flex-col">
                        <span className="text-sm font-medium text-white">
                          {new Date(s.date).toLocaleDateString([], {
                            weekday: "short",
                            month: "short",
                            day: "numeric",
                          })}
                        </span>
                        <span className="text-xs" style={{ color: "#52525b" }}>
                          {new Date(s.date).toLocaleTimeString([], {
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span
                          className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full"
                          style={{
                            background: "rgba(255,255,255,0.06)",
                            color: "#71717a",
                          }}
                        >
                          <Hash className="w-3 h-3" />
                          {s.letters.length} letters
                        </span>
                        <span
                          className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full"
                          style={{
                            background: "rgba(255,255,255,0.06)",
                            color: "#71717a",
                          }}
                        >
                          <Clock className="w-3 h-3" />
                          {fmtDuration(s.duration)}
                        </span>
                      </div>
                    </div>
                    {isOpen ? (
                      <ChevronUp
                        className="w-4 h-4 flex-shrink-0"
                        style={{ color: "#52525b" }}
                      />
                    ) : (
                      <ChevronDown
                        className="w-4 h-4 flex-shrink-0"
                        style={{ color: "#52525b" }}
                      />
                    )}
                  </button>

                  {isOpen && (
                    <div
                      className="px-5 pb-5 flex flex-col gap-4 border-t"
                      style={{ borderColor: "rgba(255,255,255,0.06)" }}
                    >
                      {s.sentences.length > 0 && (
                        <div>
                          <p
                            className="text-xs font-medium uppercase tracking-widest mb-2 mt-4"
                            style={{ color: "#52525b" }}
                          >
                            Sentences
                          </p>
                          <div className="flex flex-col gap-2">
                            {s.sentences.map((sent, i) => (
                              <p
                                key={i}
                                className="text-sm rounded-xl px-3 py-2"
                                style={{
                                  background: "rgba(6,182,212,0.06)",
                                  border: "1px solid rgba(6,182,212,0.1)",
                                  color: "#e4e4e7",
                                }}
                              >
                                {sent}
                              </p>
                            ))}
                          </div>
                        </div>
                      )}
                      <div>
                        <p
                          className="text-xs font-medium uppercase tracking-widest mb-2"
                          style={{ color: "#52525b" }}
                        >
                          Letter transcript
                        </p>
                        <p
                          className="text-sm font-mono rounded-xl px-3 py-2 leading-relaxed break-all"
                          style={{
                            background: "rgba(0,0,0,0.3)",
                            border: "1px solid rgba(255,255,255,0.07)",
                            color: "#71717a",
                          }}
                        >
                          {s.letters.join("")}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </main>
  );
}
