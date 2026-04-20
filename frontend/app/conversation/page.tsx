"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import { Mic, Square, Play, Hand, Volume2 } from "lucide-react";
import { useMyoWs } from "@/hooks/use-myo-ws";
import { useDeepgram } from "@/hooks/use-deepgram";
import { useElevenLabs } from "@/hooks/use-elevenlabs";

const BG = "#F0EFF8";
const CARD = "#FFFFFF";
const PURPLE = "#7C6FE0";
const PURPLE_LIGHT = "rgba(124,111,224,0.12)";
const PURPLE_MED = "rgba(124,111,224,0.20)";
const TEXT = "#1C1C1E";
const TEXT2 = "#6C6C70";
const TEXT3 = "#8E8E93";
const GREEN = "#34C759";
const SHADOW = "0 1px 4px rgba(0,0,0,0.07)";

interface Message {
  id: string;
  side: "asl" | "speech";
  text: string;
  timestamp: string;
}

function fmt(d: Date) {
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// ── Myo signal bars (animated when capturing) ─────────────────────────────
function SignalBars({ active }: { active: boolean }) {
  const bars = [
    { height: 6, anim: "bar1" },
    { height: 10, anim: "bar2" },
    { height: 14, anim: "bar3" },
    { height: 10, anim: "bar4" },
    { height: 6, anim: "bar5" },
  ];
  return (
    <div className="flex items-end gap-0.5" style={{ height: 16 }}>
      {bars.map((b, i) => (
        <div
          key={i}
          className="rounded-full"
          style={{
            width: 3,
            height: active ? b.height : 4,
            backgroundColor: active ? PURPLE : "rgba(124,111,224,0.35)",
            animation: active ? `${b.anim} 1s ease-in-out infinite` : "none",
            transition: "height 0.2s",
          }}
        />
      ))}
    </div>
  );
}

// ── Source badge ──────────────────────────────────────────────────────────
function Badge({ label, isAsl }: { label: string; isAsl: boolean }) {
  return (
    <span
      className="text-xs px-2 py-0.5 rounded-full font-medium"
      style={{
        backgroundColor: isAsl ? PURPLE_MED : "rgba(0,0,0,0.06)",
        color: isAsl ? PURPLE : TEXT3,
      }}
    >
      {label}
    </span>
  );
}

export default function ConversationPage() {
  const {
    status,
    sentence,
    sentenceBuilding,
    sentencePhrases,
    currentPhrase,
    gestureState,
    connect,
    disconnect,
  } = useMyoWs();

  const {
    isListening,
    interimTranscript,
    finalTranscript,
    micPermission,
    startListening,
    stopListening,
  } = useDeepgram();

  const { speak, isSpeaking } = useElevenLabs();

  const [messages, setMessages] = useState<Message[]>([]);
  const prevSentenceRef = useRef("");
  const prevFinalRef = useRef("");
  const bottomRef = useRef<HTMLDivElement>(null);

  const isActive = status === "connected";
  const isCapturing = gestureState === "capturing";
  const isThinking = gestureState === "thinking" || sentenceBuilding;
  const showSigningIndicator = isActive && (isCapturing || isThinking || !!currentPhrase);

  // Commit ASL sentence → chat bubble + speak aloud via ElevenLabs
  useEffect(() => {
    if (!sentence || sentence === prevSentenceRef.current) return;
    prevSentenceRef.current = sentence;
    setMessages((m) => [
      ...m,
      { id: Date.now().toString(), side: "asl", text: sentence, timestamp: fmt(new Date()) },
    ]);
    speak(sentence);
  }, [sentence, speak]);

  // Commit Deepgram final transcript → chat bubble
  useEffect(() => {
    if (!finalTranscript || finalTranscript === prevFinalRef.current) return;
    prevFinalRef.current = finalTranscript;
    setMessages((m) => [
      ...m,
      { id: Date.now().toString(), side: "speech", text: finalTranscript, timestamp: fmt(new Date()) },
    ]);
  }, [finalTranscript]);

  // Auto-scroll on new content
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, interimTranscript, showSigningIndicator]);

  const handleMicDown = useCallback(() => {
    if (isListening) return;
    startListening();
  }, [isListening, startListening]);

  const handleMicUp = useCallback(() => {
    stopListening();
  }, [stopListening]);

  const pendingLabel = sentenceBuilding
    ? sentencePhrases.length > 0
      ? sentencePhrases.join(" · ")
      : "Building sentence…"
    : isThinking
      ? "Recognizing…"
      : isCapturing
        ? "Signing…"
        : currentPhrase || "";

  return (
    <main className="min-h-screen flex flex-col" style={{ backgroundColor: BG }}>
      <div className="max-w-sm mx-auto w-full flex flex-col pt-12" style={{ minHeight: "100dvh" }}>

        {/* ── Header ──────────────────────────────────────────────────── */}
        <div className="px-4 mb-3">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h1 className="text-2xl font-bold" style={{ color: TEXT }}>Conversation</h1>
              <p className="text-sm mt-0.5" style={{ color: TEXT2 }}>ASL ↔ Voice</p>
            </div>
            {/* ElevenLabs speaking indicator */}
            {isSpeaking && (
              <div
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-full"
                style={{ backgroundColor: PURPLE_MED }}
              >
                <Volume2 size={13} style={{ color: PURPLE }} />
                <span className="text-xs font-medium" style={{ color: PURPLE }}>Speaking</span>
              </div>
            )}
          </div>

          {/* Myo status bar */}
          <div
            className="flex items-center justify-between px-3 py-2.5 rounded-2xl"
            style={{
              background: isActive
                ? "linear-gradient(135deg, rgba(124,111,224,0.14) 0%, rgba(124,111,224,0.07) 100%)"
                : "rgba(0,0,0,0.04)",
              border: isActive
                ? "1px solid rgba(124,111,224,0.2)"
                : "1px solid rgba(0,0,0,0.06)",
            }}
          >
            <div className="flex items-center gap-2">
              <Hand size={14} style={{ color: isActive ? PURPLE : TEXT3 }} />
              <span className="text-xs font-medium" style={{ color: isActive ? PURPLE : TEXT3 }}>
                {isActive
                  ? isCapturing
                    ? "Reading gesture…"
                    : isThinking
                      ? "Processing…"
                      : "Myo ready"
                  : status === "connecting"
                    ? "Connecting to Myo…"
                    : "Myo disconnected"}
              </span>
            </div>
            <SignalBars active={isCapturing} />
          </div>
        </div>

        {/* ── Chat area ───────────────────────────────────────────────── */}
        <div className="flex-1 overflow-y-auto px-4 pb-2" style={{ minHeight: 0 }}>
          {messages.length === 0 && !showSigningIndicator ? (
            <div className="flex flex-col items-center justify-center h-full gap-3 pb-12">
              <div
                className="w-14 h-14 rounded-2xl flex items-center justify-center mb-2"
                style={{ backgroundColor: PURPLE_LIGHT }}
              >
                <Hand size={28} style={{ color: PURPLE }} />
              </div>
              <p className="text-sm text-center max-w-xs" style={{ color: TEXT3 }}>
                {isActive
                  ? "Start signing — your words will appear here.\nYour partner can reply by holding the mic button."
                  : "Connect to begin the conversation."}
              </p>
            </div>
          ) : (
            <div className="flex flex-col gap-3 py-2">
              {messages.map((msg) => {
                const isAsl = msg.side === "asl";
                return (
                  <div
                    key={msg.id}
                    className={`flex ${isAsl ? "justify-end" : "justify-start"} items-end gap-2`}
                  >
                    {/* Hearing user avatar */}
                    {!isAsl && (
                      <div
                        className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0"
                        style={{ backgroundColor: "rgba(0,0,0,0.07)" }}
                      >
                        <Mic size={13} style={{ color: TEXT3 }} />
                      </div>
                    )}

                    <div
                      className="flex flex-col gap-1.5"
                      style={{ maxWidth: "75%", alignItems: isAsl ? "flex-end" : "flex-start" }}
                    >
                      <div
                        className="px-4 py-3 text-sm leading-relaxed"
                        style={
                          isAsl
                            ? {
                                background: `linear-gradient(135deg, ${PURPLE} 0%, #6A5FCC 100%)`,
                                color: "#fff",
                                borderRadius: "18px 18px 4px 18px",
                                boxShadow: "0 2px 8px rgba(124,111,224,0.3)",
                              }
                            : {
                                backgroundColor: CARD,
                                color: TEXT,
                                borderRadius: "18px 18px 18px 4px",
                                boxShadow: SHADOW,
                              }
                        }
                      >
                        {msg.text}
                      </div>
                      <div className="flex items-center gap-1.5 px-1">
                        <Badge
                          label={isAsl ? "ASL → text" : "voice → text"}
                          isAsl={isAsl}
                        />
                        <span className="text-xs" style={{ color: TEXT3 }}>
                          {msg.timestamp}
                        </span>
                      </div>
                    </div>

                    {/* ASL user avatar */}
                    {isAsl && (
                      <div
                        className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0"
                        style={{ backgroundColor: PURPLE_LIGHT }}
                      >
                        <Hand size={13} style={{ color: PURPLE }} />
                      </div>
                    )}
                  </div>
                );
              })}

              {/* ASL signing / sentence-building indicator */}
              {showSigningIndicator && (
                <div className="flex justify-end items-end gap-2">
                  <div
                    className="px-4 py-3 text-sm"
                    style={{
                      background: sentenceBuilding
                        ? `linear-gradient(135deg, ${PURPLE_MED} 0%, rgba(124,111,224,0.08) 100%)`
                        : "rgba(124,111,224,0.06)",
                      color: sentenceBuilding ? PURPLE : TEXT3,
                      borderRadius: "18px 18px 4px 18px",
                      border: `1px solid ${sentenceBuilding ? "rgba(124,111,224,0.25)" : "rgba(0,0,0,0.06)"}`,
                      maxWidth: "75%",
                    }}
                  >
                    <div className="flex items-center gap-2">
                      <span>{pendingLabel}</span>
                      <div className="flex gap-1">
                        {[0, 1, 2].map((i) => (
                          <div
                            key={i}
                            className="rounded-full"
                            style={{
                              width: 5,
                              height: 5,
                              backgroundColor: sentenceBuilding ? PURPLE : TEXT3,
                              animation: `thinking-dot 1.2s ease-in-out ${i * 0.2}s infinite`,
                            }}
                          />
                        ))}
                      </div>
                    </div>
                  </div>
                  <div
                    className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0"
                    style={{ backgroundColor: PURPLE_LIGHT }}
                  >
                    <Hand size={13} style={{ color: PURPLE }} />
                  </div>
                </div>
              )}

              <div ref={bottomRef} />
            </div>
          )}
        </div>

        {/* ── Bottom controls ──────────────────────────────────────────── */}
        <div
          className="px-4 pb-24 pt-3 flex flex-col gap-2"
          style={{ borderTop: "1px solid rgba(0,0,0,0.06)" }}
        >
          {/* Live preview box — Deepgram interim */}
          {isActive && (isListening || interimTranscript) && (
            <div
              className="rounded-2xl px-4 py-3 flex items-start gap-3"
              style={{
                backgroundColor: "rgba(52,199,89,0.08)",
                border: "1px solid rgba(52,199,89,0.2)",
              }}
            >
              {/* Pulsing green dot */}
              <div className="flex-shrink-0 mt-1">
                <div
                  className="w-2 h-2 rounded-full"
                  style={{
                    backgroundColor: GREEN,
                    animation: isListening ? "pulse-dot 1s ease-in-out infinite" : "none",
                  }}
                />
              </div>
              <p
                className="text-sm italic leading-relaxed flex-1"
                style={{ color: interimTranscript ? TEXT2 : TEXT3 }}
              >
                {interimTranscript || "Listening…"}
              </p>
            </div>
          )}

          {/* Mic permission denied notice */}
          {micPermission === "denied" && (
            <div
              className="rounded-2xl px-4 py-3 text-sm text-center"
              style={{
                backgroundColor: "rgba(255,59,48,0.08)",
                color: "#FF3B30",
                border: "1px solid rgba(255,59,48,0.15)",
              }}
            >
              Microphone access was denied. Please allow mic access in your browser settings.
            </div>
          )}

          {/* Hold-to-speak mic button */}
          {isActive && micPermission !== "denied" && (
            <button
              onPointerDown={handleMicDown}
              onPointerUp={handleMicUp}
              onPointerLeave={handleMicUp}
              className="w-full flex items-center justify-center gap-2 rounded-2xl py-4 font-semibold select-none cursor-pointer"
              style={{
                background: isListening
                  ? `linear-gradient(135deg, ${GREEN} 0%, #28A745 100%)`
                  : CARD,
                color: isListening ? "#fff" : TEXT2,
                boxShadow: isListening
                  ? "0 4px 14px rgba(52,199,89,0.35)"
                  : SHADOW,
                fontSize: 15,
                border: isListening ? "none" : "1px solid rgba(0,0,0,0.06)",
                transition: "background 0.15s, color 0.15s, box-shadow 0.15s",
              }}
            >
              {isListening ? (
                <>
                  <Volume2 size={18} />
                  Listening…
                </>
              ) : (
                <>
                  <Mic size={18} />
                  Hold to Speak
                </>
              )}
            </button>
          )}

          {/* Connect / End */}
          {!isActive ? (
            <button
              onClick={() =>
                connect(
                  process.env.NEXT_PUBLIC_MAIA_WS_URL ?? "ws://localhost:8765"
                )
              }
              className="w-full flex items-center justify-center gap-2 rounded-2xl py-4 font-semibold cursor-pointer text-white"
              style={{
                background: `linear-gradient(135deg, ${PURPLE} 0%, #6A5FCC 100%)`,
                boxShadow: "0 4px 14px rgba(124,111,224,0.35)",
                fontSize: 15,
              }}
            >
              <Play size={16} fill="white" />
              {status === "connecting" ? "Connecting…" : "Start Conversation"}
            </button>
          ) : (
            <button
              onClick={disconnect}
              className="w-full flex items-center justify-center gap-2 rounded-2xl py-3 font-medium cursor-pointer"
              style={{
                backgroundColor: "rgba(255,59,48,0.08)",
                color: "#FF3B30",
                fontSize: 14,
                border: "1px solid rgba(255,59,48,0.15)",
              }}
            >
              <Square size={14} fill="#FF3B30" />
              End Conversation
            </button>
          )}
        </div>
      </div>
    </main>
  );
}
