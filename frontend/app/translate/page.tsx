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
const GREEN = "#34C759";
const AMBER = "#F59E0B";
const SHADOW = "0 1px 4px rgba(0,0,0,0.07)";

// Must match backend constants
const DTW_ONSET_RMS = 18.0;
const MAX_RMS = 60.0;
const MAX_GESTURE_FRAMES = 1200;

export default function TranslatePage() {
  const {
    status,
    currentPhrase,
    phraseConfidence,
    phraseStream,
    sentence,
    sentenceBuilding,
    sentencePhrases,
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

  // Animate + TTS on new phrase
  useEffect(() => {
    if (!currentPhrase || currentPhrase === prevPhraseRef.current) return;
    setPhraseKey((k) => k + 1);
    prevPhraseRef.current = currentPhrase;
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

  const statusDotColor =
    status === "connected" ? GREEN :
    status === "connecting" ? AMBER :
    status === "demo" ? PURPLE :
    TEXT3;

  const statusLabel =
    status === "connected" ? `Connected${deviceName ? ` · ${deviceName}` : ""}` :
    status === "connecting" ? "Connecting..." :
    status === "demo" ? "Demo" :
    "Disconnected";

  const recentPhrases = phraseStream.slice(-8);

  // RMS bar — clamp and compute threshold marker
  const rmsBarPct = Math.min(gestureRms / MAX_RMS, 1);
  const thresholdPct = DTW_ONSET_RMS / MAX_RMS;
  const rmsBelowThresh = gestureRms < DTW_ONSET_RMS;
  const capturePct = Math.min(gestureFrames / MAX_GESTURE_FRAMES, 1);

  return (
    <main className="min-h-screen pb-24 px-4" style={{ backgroundColor: BG }}>
      <div className="max-w-sm mx-auto pt-12">

        {/* Header */}
        <div className="mb-6">
          <h1 className="text-2xl font-bold" style={{ color: TEXT }}>Live Translation</h1>
          <p className="text-sm mt-1" style={{ color: TEXT2 }}>ASL gestures → spoken words</p>
        </div>

        {/* ── Status + gesture state card ── */}
        <div
          className="rounded-2xl mb-4 overflow-hidden"
          style={{ backgroundColor: CARD, boxShadow: SHADOW }}
        >
          {/* Top bar: connection + gesture state */}
          <div
            className="flex items-center justify-between px-4 py-3"
            style={{ borderBottom: "1px solid rgba(0,0,0,0.05)" }}
          >
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: statusDotColor }} />
              <span className="text-xs" style={{ color: TEXT3 }}>{statusLabel}</span>
            </div>

            {/* Gesture state badge */}
            {isActive && (
              <div
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium"
                style={{
                  backgroundColor:
                    gestureState === "capturing" ? "rgba(52,199,89,0.12)" :
                    gestureState === "thinking"  ? PURPLE_LIGHT :
                    "rgba(0,0,0,0.05)",
                  color:
                    gestureState === "capturing" ? GREEN :
                    gestureState === "thinking"  ? PURPLE :
                    TEXT3,
                }}
              >
                {gestureState === "capturing" && (
                  <span
                    className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                    style={{ backgroundColor: GREEN, animation: "pulse-dot 1s ease-in-out infinite" }}
                  />
                )}
                {gestureState === "thinking" && (
                  <span
                    className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                    style={{ backgroundColor: PURPLE, animation: "pulse-dot 0.8s ease-in-out infinite" }}
                  />
                )}
                {gestureState === "capturing" ? "Capturing" :
                 gestureState === "thinking"  ? "Recognizing" :
                 "Listening"}
              </div>
            )}
          </div>

          {/* Signal visualiser */}
          {isActive && (
            <div className="px-4 py-3" style={{ borderBottom: "1px solid rgba(0,0,0,0.05)" }}>
              {gestureState === "thinking" ? (
                /* Thinking: pulsing dots */
                <div className="flex items-center gap-2 h-8">
                  <span className="text-xs mr-1" style={{ color: TEXT3 }}>Recognizing</span>
                  {[0, 1, 2].map((i) => (
                    <div
                      key={i}
                      className="rounded-full"
                      style={{
                        width: 7, height: 7,
                        backgroundColor: PURPLE,
                        animation: `thinking-dot 1.2s ease-in-out ${i * 0.2}s infinite`,
                      }}
                    />
                  ))}
                </div>
              ) : gestureState === "capturing" ? (
                /* Capturing: growing bar + frame count */
                <div className="flex flex-col gap-1.5">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium" style={{ color: GREEN }}>Signing detected</span>
                    <span className="text-xs font-mono" style={{ color: TEXT3 }}>
                      {gestureFrames} / {MAX_GESTURE_FRAMES} frames
                    </span>
                  </div>
                  <div
                    className="w-full rounded-full overflow-hidden"
                    style={{ height: 6, backgroundColor: "rgba(52,199,89,0.15)" }}
                  >
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${capturePct * 100}%`,
                        backgroundColor: GREEN,
                        transition: "width 0.08s linear",
                      }}
                    />
                  </div>
                </div>
              ) : (
                /* Idle: RMS bar with threshold line */
                <div className="flex flex-col gap-1.5">
                  <div className="flex items-center justify-between">
                    <span className="text-xs" style={{ color: TEXT3 }}>Signal strength</span>
                    <span
                      className="text-xs font-mono"
                      style={{ color: rmsBelowThresh ? TEXT3 : GREEN }}
                    >
                      {gestureRms.toFixed(1)} / {DTW_ONSET_RMS} threshold
                    </span>
                  </div>
                  <div className="relative w-full rounded-full overflow-visible" style={{ height: 6 }}>
                    {/* Track */}
                    <div
                      className="absolute inset-0 rounded-full"
                      style={{ backgroundColor: "rgba(0,0,0,0.08)" }}
                    />
                    {/* Fill */}
                    <div
                      className="absolute top-0 left-0 h-full rounded-full"
                      style={{
                        width: `${rmsBarPct * 100}%`,
                        backgroundColor: rmsBelowThresh ? AMBER : GREEN,
                        transition: "width 0.08s linear, background-color 0.15s",
                      }}
                    />
                    {/* Threshold tick */}
                    <div
                      className="absolute top-0 bottom-0 w-0.5 rounded-full"
                      style={{
                        left: `${thresholdPct * 100}%`,
                        backgroundColor: "rgba(0,0,0,0.25)",
                        transform: "translateX(-50%)",
                        height: 10,
                        top: -2,
                      }}
                    />
                  </div>
                  <p className="text-xs" style={{ color: TEXT3 }}>
                    {rmsBelowThresh
                      ? "Sign a gesture to begin — keep above the threshold"
                      : "Threshold reached — hold your sign"}
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Phrase output */}
          <div className="px-6 py-6 flex flex-col items-center justify-center" style={{ minHeight: 160 }}>
            {currentPhrase && isActive ? (
              <div
                key={`${currentPhrase}-${phraseKey}`}
                className="letter-pop flex flex-col items-center gap-3 w-full"
              >
                <span
                  className="font-bold leading-tight text-center"
                  style={{ fontSize: 48, color: PURPLE }}
                >
                  {currentPhrase}
                </span>
                <div className="w-full flex flex-col items-center gap-1">
                  <div
                    className="w-full h-1.5 rounded-full overflow-hidden"
                    style={{ backgroundColor: PURPLE_LIGHT }}
                  >
                    <div
                      className="h-full rounded-full transition-all duration-300"
                      style={{ width: `${phraseConfidence * 100}%`, backgroundColor: PURPLE }}
                    />
                  </div>
                  <span className="text-xs" style={{ color: TEXT3 }}>
                    {(phraseConfidence * 100).toFixed(0)}% confidence
                  </span>
                </div>
              </div>
            ) : (
              <p className="text-sm text-center" style={{ color: TEXT3 }}>
                {isActive
                  ? gestureState === "idle"
                    ? "Waiting for a gesture…"
                    : "\u00a0"
                  : "Press Start to begin"}
              </p>
            )}
          </div>

          {/* Phrase history */}
          {recentPhrases.length > 0 && (
            <div
              className="px-4 py-3 flex flex-wrap gap-1.5"
              style={{ borderTop: "1px solid rgba(0,0,0,0.05)" }}
            >
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
          )}
        </div>

        {/* Sentence construction card — shown when building or sentence exists */}
        {isActive && (sentence || sentenceBuilding) && (
          <div
            className="rounded-2xl p-4 mb-4"
            style={{ backgroundColor: CARD, boxShadow: SHADOW }}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold uppercase tracking-wider" style={{ color: TEXT3 }}>
                Sentence
              </span>
              {sentenceBuilding && (
                <div className="flex items-center gap-1.5">
                  <span className="text-xs" style={{ color: PURPLE }}>
                    {sentencePhrases.join(" · ")}
                  </span>
                  <div className="flex gap-1">
                    {[0, 1, 2].map((i) => (
                      <div
                        key={i}
                        className="rounded-full"
                        style={{
                          width: 4, height: 4,
                          backgroundColor: PURPLE,
                          animation: `thinking-dot 1.2s ease-in-out ${i * 0.2}s infinite`,
                        }}
                      />
                    ))}
                  </div>
                </div>
              )}
            </div>
            {sentence && !sentenceBuilding && (
              <p
                className="text-base font-medium leading-snug"
                style={{ color: TEXT }}
              >
                {sentence}
              </p>
            )}
          </div>
        )}

        {/* Start / Stop */}
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
            Stop
          </button>
        )}

        {/* Secondary */}
        <div className="flex gap-3">
          <button
            onClick={handleReplay}
            disabled={!currentPhrase}
            className="flex-1 flex items-center justify-center gap-2 rounded-2xl py-3 font-medium cursor-pointer"
            style={{
              backgroundColor: CARD, boxShadow: SHADOW,
              color: currentPhrase ? TEXT2 : TEXT3,
              border: "1px solid rgba(0,0,0,0.06)", fontSize: 14,
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
              backgroundColor: CARD, boxShadow: SHADOW,
              color: phraseStream.length > 0 ? TEXT2 : TEXT3,
              border: "1px solid rgba(0,0,0,0.06)", fontSize: 14,
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
