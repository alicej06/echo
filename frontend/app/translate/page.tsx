"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import { Play, Square, RotateCcw, Bookmark } from "lucide-react";
import { useMyoWs } from "@/hooks/use-myo-ws";

const BG = "#F0EFF8";
const CARD = "#FFFFFF";
const PURPLE = "#7C6FE0";
const PURPLE_LIGHT = "rgba(124,111,224,0.12)";
const TEXT = "#1C1C1E";
const TEXT2 = "#6C6C70";
const TEXT3 = "#8E8E93";
const SHADOW = "0 1px 4px rgba(0,0,0,0.07)";

const EMG_CLASSES = [
  "emg-bar-1","emg-bar-2","emg-bar-3","emg-bar-4","emg-bar-5",
  "emg-bar-2","emg-bar-4","emg-bar-1","emg-bar-3","emg-bar-5",
  "emg-bar-4","emg-bar-2","emg-bar-1","emg-bar-5","emg-bar-3",
  "emg-bar-1","emg-bar-4","emg-bar-2","emg-bar-5","emg-bar-3",
];

export default function TranslatePage() {
  const { status, currentLetter, confidence, letterStream, sentence, connect, disconnect, startDemo, clearStream } = useMyoWs();
  const [wsUrl] = useState("ws://localhost:8765");
  const [letterKey, setLetterKey] = useState(0);
  const [saved, setSaved] = useState(false);
  const prevLetterRef = useRef("");
  const prevSentenceRef = useRef("");
  const sessionStartRef = useRef<number>(Date.now());

  const isActive = status === "connected" || status === "demo";

  // Save session on disconnect
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
        localStorage.setItem("maia_sessions", JSON.stringify(sessions.slice(-50)));
      } catch { /* ignore */ }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  // Letter animation
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

  const handleStart = useCallback(() => {
    startDemo();
  }, [startDemo]);

  const handleReplay = useCallback(() => {
    if (!sentence) return;
    window.speechSynthesis?.cancel();
    const utt = new SpeechSynthesisUtterance(sentence);
    utt.rate = 0.95;
    window.speechSynthesis?.speak(utt);
  }, [sentence]);

  const handleSave = useCallback(() => {
    if (!letterStream.length && !sentence) return;
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
      localStorage.setItem("maia_sessions", JSON.stringify(sessions.slice(-50)));
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch { /* ignore */ }
  }, [letterStream, sentence]);

  const statusLabel = {
    disconnected: "Ready to translate",
    connecting: "Connecting...",
    connected: "Translating",
    demo: "Demo mode",
  }[status];

  const statusDotColor = {
    disconnected: TEXT3,
    connecting: "#F59E0B",
    connected: "#34C759",
    demo: PURPLE,
  }[status];

  return (
    <main className="min-h-screen pb-24 px-4" style={{ backgroundColor: BG }}>
      <div className="max-w-sm mx-auto pt-12">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-2xl font-bold" style={{ color: TEXT }}>Live Translation</h1>
          <p className="text-sm mt-1" style={{ color: TEXT2 }}>ASL gestures to spoken words</p>
        </div>

        {/* Main output card */}
        <div
          className="rounded-2xl p-6 mb-4 min-h-40 flex flex-col items-center justify-center"
          style={{ backgroundColor: CARD, boxShadow: SHADOW }}
        >
          <div className="flex items-center gap-2 mb-4 self-start">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: statusDotColor }} />
            <span className="text-sm" style={{ color: TEXT3 }}>{statusLabel}</span>
          </div>

          {currentLetter && isActive ? (
            <div key={`${currentLetter}-${letterKey}`} className="letter-pop flex flex-col items-center gap-2">
              <span
                className="font-bold leading-none"
                style={{ fontSize: 80, color: PURPLE }}
              >
                {currentLetter}
              </span>
              <div className="w-full h-1 rounded-full overflow-hidden" style={{ backgroundColor: "rgba(124,111,224,0.15)" }}>
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{ width: `${confidence * 100}%`, backgroundColor: PURPLE }}
                />
              </div>
              <span className="text-xs" style={{ color: TEXT3 }}>{(confidence * 100).toFixed(0)}% confidence</span>
            </div>
          ) : sentence ? (
            <p className="text-lg font-medium text-center leading-relaxed" style={{ color: TEXT }}>
              {sentence}
            </p>
          ) : (
            <p className="text-base text-center" style={{ color: TEXT3 }}>
              Press Start to begin translation
            </p>
          )}

          {/* Letter stream */}
          {letterStream.length > 0 && (
            <div className="mt-4 self-start w-full">
              <div className="flex flex-wrap gap-1 max-h-16 overflow-hidden">
                {letterStream.slice(-30).map((l, i, arr) => (
                  <span
                    key={i}
                    className="inline-flex items-center justify-center w-6 h-6 rounded text-xs font-bold"
                    style={{
                      backgroundColor: i === arr.length - 1 ? PURPLE_LIGHT : "rgba(0,0,0,0.04)",
                      color: i === arr.length - 1 ? PURPLE : TEXT3,
                    }}
                  >
                    {l}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* EMG Signal card */}
        <div
          className="rounded-2xl p-4 mb-4"
          style={{ backgroundColor: CARD, boxShadow: SHADOW }}
        >
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-semibold uppercase tracking-wider" style={{ color: TEXT3 }}>
              EMG Signal
            </span>
            <div className="flex items-center gap-1.5">
              <span style={{ fontSize: 12 }}>🎙</span>
              <span className="text-xs" style={{ color: isActive ? "#34C759" : TEXT3 }}>
                {isActive ? "Active" : "Inactive"}
              </span>
            </div>
          </div>

          {/* Waveform bars */}
          <div className="flex items-center gap-1" style={{ height: 40 }}>
            {EMG_CLASSES.map((cls, i) => (
              <div
                key={i}
                className={isActive ? cls : ""}
                style={{
                  flex: 1,
                  height: isActive ? undefined : "20%",
                  backgroundColor: isActive ? PURPLE : "rgba(0,0,0,0.12)",
                  borderRadius: 2,
                  alignSelf: "center",
                  transition: "background-color 0.3s",
                }}
              />
            ))}
          </div>
        </div>

        {/* Start / Stop button */}
        {!isActive ? (
          <button
            onClick={handleStart}
            className="w-full flex items-center justify-center gap-2 rounded-2xl py-4 text-white font-semibold mb-3 cursor-pointer"
            style={{ backgroundColor: PURPLE, fontSize: 16 }}
          >
            <Play size={18} fill="white" />
            Start Translation
          </button>
        ) : (
          <button
            onClick={disconnect}
            className="w-full flex items-center justify-center gap-2 rounded-2xl py-4 font-semibold mb-3 cursor-pointer"
            style={{ backgroundColor: "#FF3B30", color: "#fff", fontSize: 16 }}
          >
            <Square size={16} fill="white" />
            Stop Translation
          </button>
        )}

        {/* Secondary buttons */}
        <div className="flex gap-3">
          <button
            onClick={handleReplay}
            disabled={!sentence}
            className="flex-1 flex items-center justify-center gap-2 rounded-2xl py-3 font-medium cursor-pointer"
            style={{
              backgroundColor: CARD,
              boxShadow: SHADOW,
              color: sentence ? TEXT2 : TEXT3,
              border: "1px solid rgba(0,0,0,0.06)",
              fontSize: 14,
              opacity: sentence ? 1 : 0.5,
            }}
          >
            <RotateCcw size={15} />
            Replay
          </button>
          <button
            onClick={handleSave}
            disabled={!letterStream.length && !sentence}
            className="flex-1 flex items-center justify-center gap-2 rounded-2xl py-3 font-medium cursor-pointer"
            style={{
              backgroundColor: CARD,
              boxShadow: SHADOW,
              color: saved ? "#34C759" : (!letterStream.length && !sentence ? TEXT3 : TEXT2),
              border: "1px solid rgba(0,0,0,0.06)",
              fontSize: 14,
              opacity: (!letterStream.length && !sentence) ? 0.5 : 1,
            }}
          >
            <Bookmark size={15} />
            {saved ? "Saved!" : "Save"}
          </button>
        </div>

        {/* WS connect (tucked away, small) */}
        {!isActive && (
          <div className="mt-4">
            <button
              onClick={() => connect(wsUrl)}
              className="w-full text-center text-xs py-2 rounded-xl cursor-pointer"
              style={{ color: TEXT3, backgroundColor: "rgba(0,0,0,0.04)" }}
            >
              Connect to Myo wristband (ws://localhost:8765)
            </button>
          </div>
        )}
      </div>
    </main>
  );
}
