"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import { Play, Square, RotateCcw, Trash2 } from "lucide-react";
import { useMyoWs } from "@/hooks/use-myo-ws";

const BG = "#F0EFF8";
const CARD = "#FFFFFF";
const PURPLE = "#7C6FE0";
const PURPLE_LIGHT = "rgba(124,111,224,0.12)";
const TEXT = "#1C1C1E";
const TEXT2 = "#6C6C70";
const TEXT3 = "#8E8E93";
const SHADOW = "0 1px 4px rgba(0,0,0,0.07)";
const GREEN = "#34C759";
const AMBER = "#F59E0B";

// DTW_MAX_GESTURE from backend — used to size the capture progress bar
const MAX_GESTURE_FRAMES = 1200;
// DTW_ONSET_RMS — threshold line position on the RMS bar
const ONSET_RMS = 18.0;
// Rough max RMS for full bar display
const MAX_RMS = 60.0;

export default function TranslatePage() {
  const {
    status,
    currentPhrase,
    phraseConfidence,
    phraseStream,
    deviceName,
    gestureState,
    gestureRms,
    gestureFrames,
    connect,
    disconnect,
    clearStream,
  } = useMyoWs();

  const [phraseKey, setPhraseKey] = useState(0);
  const prevPhraseRef = useRef("");
  const sessionStartRef = useRef<number>(Date.now());

  const isActive = status === "connected" || status === "demo";

  // Save session to localStorage on disconnect
  useEffect(() => {
    if (status === "connected" || status === "demo") {
      sessionStartRef.current = Date.now();
    }
    if (status === "disconnected" && phraseStream.length > 0) {
      try {
        const raw = localStorage.getItem("maia_sessions");
        const sessions = raw ? (JSON.parse(raw) as object[]) : [];
        sessions.push({
          id: Date.now().toString(),
          date: new Date().toISOString(),
          phrases: phraseStream,
          duration: Date.now() - sessionStartRef.current,
        });
        localStorage.setItem("maia_sessions", JSON.stringify(sessions.slice(-50)));
      } catch { /* ignore */ }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  // Animate on new phrase
  useEffect(() => {
    if (currentPhrase && currentPhrase !== prevPhraseRef.current) {
      setPhraseKey((k) => k + 1);
      prevPhraseRef.current = currentPhrase;
    }
  }, [currentPhrase]);

  // TTS: speak new phrase automatically
  useEffect(() => {
    if (!currentPhrase || currentPhrase === prevPhraseRef.current) return;
    if (typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.cancel();
      const utt = new SpeechSynthesisUtterance(currentPhrase);
      utt.rate = 0.95;
      window.speechSynthesis.speak(utt);
    }
  }, [currentPhrase]);

  const handleReplay = useCallback(() => {
    if (!currentPhrase) return;
    window.speechSynthesis?.cancel();
    const utt = new SpeechSynthesisUtterance(currentPhrase);
    utt.rate = 0.95;
    window.speechSynthesis?.speak(utt);
  }, [currentPhrase]);

  const statusLabel = {
    disconnected: "Disconnected",
    connecting: "Connecting...",
    connected: "Connected",
    demo: "Demo mode",
  }[status];

  const statusDotColor = {
    disconnected: TEXT3,
    connecting: AMBER,
    connected: GREEN,
    demo: PURPLE,
  }[status];

  const recentPhrases = phraseStream.slice(-8);

  // Gesture status card content
  const rmsBarPct = Math.min(gestureRms / MAX_RMS, 1);
  const onsetLinePct = ONSET_RMS / MAX_RMS;
  const captureBarPct = Math.min(gestureFrames / MAX_GESTURE_FRAMES, 1);

  const gestureLabel =
    !isActive ? "Inactive" :
    gestureState === "capturing" ? `Capturing · ${gestureFrames} frames` :
    gestureState === "thinking"  ? "Recognizing..." :
    "Listening";

  const gestureColor =
    gestureState === "capturing" ? GREEN :
    gestureState === "thinking"  ? PURPLE :
    TEXT3;

  return (
    <main className="min-h-screen pb-24 px-4" style={{ backgroundColor: BG }}>
      <div className="max-w-sm mx-auto pt-12">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-2xl font-bold" style={{ color: TEXT }}>
            Live Translation
          </h1>
          <p className="text-sm mt-1" style={{ color: TEXT2 }}>
            ASL gestures → spoken words
          </p>
        </div>

        {/* Main output card */}
        <div
          className="rounded-2xl p-6 mb-4 flex flex-col"
          style={{ backgroundColor: CARD, boxShadow: SHADOW, minHeight: 200 }}
        >
          {/* Status row */}
          <div className="flex items-center gap-2 mb-4">
            <div
              className="w-2 h-2 rounded-full flex-shrink-0"
              style={{ backgroundColor: statusDotColor }}
            />
            <span className="text-sm" style={{ color: TEXT3 }}>
              {statusLabel}
              {deviceName && status === "connected" ? ` · ${deviceName}` : ""}
            </span>
          </div>

          {/* Phrase display */}
          <div className="flex-1 flex flex-col items-center justify-center py-2">
            {currentPhrase && isActive ? (
              <div
                key={`${currentPhrase}-${phraseKey}`}
                className="letter-pop flex flex-col items-center gap-3 w-full"
              >
                <span
                  className="font-bold leading-tight text-center"
                  style={{ fontSize: 52, color: PURPLE }}
                >
                  {currentPhrase}
                </span>
                <div className="w-full flex flex-col items-center gap-1">
                  <div
                    className="w-full h-1.5 rounded-full overflow-hidden"
                    style={{ backgroundColor: "rgba(124,111,224,0.15)" }}
                  >
                    <div
                      className="h-full rounded-full transition-all duration-300"
                      style={{
                        width: `${phraseConfidence * 100}%`,
                        backgroundColor: PURPLE,
                      }}
                    />
                  </div>
                  <span className="text-xs" style={{ color: TEXT3 }}>
                    {(phraseConfidence * 100).toFixed(0)}% confidence
                  </span>
                </div>
              </div>
            ) : (
              <p
                className="text-base text-center leading-relaxed"
                style={{ color: TEXT3 }}
              >
                {isActive
                  ? "Start signing — phrases will appear here"
                  : "Press Start to begin translation"}
              </p>
            )}
          </div>

          {/* Phrase history chips */}
          {recentPhrases.length > 0 && (
            <div className="mt-4 pt-4" style={{ borderTop: "1px solid rgba(0,0,0,0.06)" }}>
              <div className="flex flex-wrap gap-1.5">
                {recentPhrases.map((phrase, i, arr) => {
                  const isLatest = i === arr.length - 1;
                  return (
                    <span
                      key={i}
                      className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium"
                      style={{
                        backgroundColor: isLatest ? PURPLE_LIGHT : "rgba(0,0,0,0.05)",
                        color: isLatest ? PURPLE : TEXT3,
                      }}
                    >
                      {phrase}
                    </span>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Gesture status card */}
        <div
          className="rounded-2xl p-4 mb-4"
          style={{ backgroundColor: CARD, boxShadow: SHADOW }}
        >
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-semibold uppercase tracking-wider" style={{ color: TEXT3 }}>
              Signal
            </span>
            <span className="text-xs font-medium" style={{ color: gestureColor }}>
              {gestureLabel}
            </span>
          </div>

          {gestureState === "thinking" && isActive ? (
            /* Thinking: animated dots */
            <div className="flex items-center justify-center gap-2" style={{ height: 36 }}>
              {[0, 1, 2].map((i) => (
                <div
                  key={i}
                  className="rounded-full"
                  style={{
                    width: 8, height: 8,
                    backgroundColor: PURPLE,
                    animation: `thinking-dot 1.2s ease-in-out ${i * 0.2}s infinite`,
                  }}
                />
              ))}
            </div>
          ) : gestureState === "capturing" && isActive ? (
            /* Capturing: growing progress bar */
            <div style={{ height: 36 }} className="flex flex-col justify-center gap-1.5">
              <div
                className="w-full rounded-full overflow-hidden"
                style={{ height: 8, backgroundColor: "rgba(0,0,0,0.07)" }}
              >
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${captureBarPct * 100}%`,
                    backgroundColor: GREEN,
                    transition: "width 0.1s linear",
                  }}
                />
              </div>
              <div className="flex justify-between">
                <span className="text-xs" style={{ color: TEXT3 }}>
                  {gestureFrames} frames
                </span>
                <span className="text-xs" style={{ color: TEXT3 }}>
                  {MAX_GESTURE_FRAMES}
                </span>
              </div>
            </div>
          ) : (
            /* Idle / inactive: RMS bar with onset threshold line */
            <div style={{ height: 36 }} className="flex flex-col justify-center gap-1.5">
              <div
                className="w-full rounded-full overflow-hidden relative"
                style={{ height: 8, backgroundColor: "rgba(0,0,0,0.07)" }}
              >
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${rmsBarPct * 100}%`,
                    backgroundColor: isActive
                      ? (gestureRms >= ONSET_RMS ? GREEN : AMBER)
                      : "rgba(0,0,0,0.15)",
                    transition: "width 0.1s linear, background-color 0.2s",
                  }}
                />
                {/* Onset threshold marker */}
                {isActive && (
                  <div
                    className="absolute top-0 bottom-0 w-px"
                    style={{
                      left: `${onsetLinePct * 100}%`,
                      backgroundColor: "rgba(0,0,0,0.2)",
                    }}
                  />
                )}
              </div>
              {isActive && (
                <div className="flex justify-between">
                  <span className="text-xs" style={{ color: TEXT3 }}>RMS {gestureRms.toFixed(1)}</span>
                  <span className="text-xs" style={{ color: TEXT3 }}>threshold {ONSET_RMS}</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Start / Stop button */}
        {!isActive ? (
          <button
            onClick={() => connect("ws://localhost:8765")}
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
        <div className="flex gap-3 mb-4">
          <button
            onClick={handleReplay}
            disabled={!currentPhrase}
            className="flex-1 flex items-center justify-center gap-2 rounded-2xl py-3 font-medium cursor-pointer"
            style={{
              backgroundColor: CARD,
              boxShadow: SHADOW,
              color: currentPhrase ? TEXT2 : TEXT3,
              border: "1px solid rgba(0,0,0,0.06)",
              fontSize: 14,
              opacity: currentPhrase ? 1 : 0.5,
            }}
          >
            <RotateCcw size={15} />
            Replay
          </button>
          <button
            onClick={clearStream}
            disabled={phraseStream.length === 0}
            className="flex-1 flex items-center justify-center gap-2 rounded-2xl py-3 font-medium cursor-pointer"
            style={{
              backgroundColor: CARD,
              boxShadow: SHADOW,
              color: phraseStream.length > 0 ? TEXT2 : TEXT3,
              border: "1px solid rgba(0,0,0,0.06)",
              fontSize: 14,
              opacity: phraseStream.length > 0 ? 1 : 0.5,
            }}
          >
            <Trash2 size={15} />
            Clear
          </button>
        </div>
      </div>
    </main>
  );
}
