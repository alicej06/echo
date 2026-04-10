"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import { Mic, Square, Play, Hand, Volume2 } from "lucide-react";
import { useMyoWs } from "@/hooks/use-myo-ws";

const BG = "#F0EFF8";
const CARD = "#FFFFFF";
const PURPLE = "#7C6FE0";
const PURPLE_LIGHT = "rgba(124,111,224,0.12)";
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
  pending?: boolean;
}

function fmt(d: Date) {
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
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

  const [messages, setMessages] = useState<Message[]>([]);
  const [isListening, setIsListening] = useState(false);
  const [interimText, setInterimText] = useState("");
  const prevSentenceRef = useRef("");
  const recognizerRef = useRef<SpeechRecognition | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const isActive = status === "connected";

  // Push completed ASL sentence into chat
  useEffect(() => {
    if (!sentence || sentence === prevSentenceRef.current) return;
    prevSentenceRef.current = sentence;
    setMessages((m) => [
      ...m,
      {
        id: Date.now().toString(),
        side: "asl",
        text: sentence,
        timestamp: fmt(new Date()),
      },
    ]);
    // Speak ASL user's sentence aloud for the hearing partner
    if (typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.cancel();
      const utt = new SpeechSynthesisUtterance(sentence);
      utt.rate = 0.95;
      window.speechSynthesis.speak(utt);
    }
  }, [sentence]);

  // Auto-scroll on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, sentenceBuilding, gestureState]);

  // Speech recognition for hearing user
  const startListening = useCallback(() => {
    const Ctor =
      typeof window !== "undefined"
        ? window.SpeechRecognition ?? window.webkitSpeechRecognition
        : null;
    if (!Ctor) return;
    const rec = new Ctor();
    rec.continuous = false;
    rec.interimResults = true;
    rec.lang = "en-US";
    recognizerRef.current = rec;

    rec.onresult = (ev: SpeechRecognitionEvent) => {
      let interim = "";
      let final = "";
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const t = ev.results[i][0].transcript;
        if (ev.results[i].isFinal) final += t;
        else interim += t;
      }
      setInterimText(interim);
      if (final.trim()) {
        setInterimText("");
        setMessages((m) => [
          ...m,
          {
            id: Date.now().toString(),
            side: "speech",
            text: final.trim(),
            timestamp: fmt(new Date()),
          },
        ]);
      }
    };
    rec.onend = () => {
      setIsListening(false);
      setInterimText("");
    };
    rec.onerror = () => {
      setIsListening(false);
      setInterimText("");
    };
    rec.start();
    setIsListening(true);
  }, []);

  const stopListening = useCallback(() => {
    recognizerRef.current?.stop();
  }, []);

  // Signing state: what to show in the ASL "pending" bubble
  const isCapturing = gestureState === "capturing";
  const isThinking = gestureState === "thinking" || sentenceBuilding;
  const showSigningIndicator = isActive && (isCapturing || isThinking || currentPhrase);

  const pendingLabel =
    sentenceBuilding
      ? sentencePhrases.length > 0
        ? sentencePhrases.join(" · ")
        : "Building sentence…"
      : isThinking
        ? "Recognizing…"
        : isCapturing
          ? "Signing…"
          : currentPhrase || "";

  return (
    <main
      className="min-h-screen flex flex-col"
      style={{ backgroundColor: BG }}
    >
      <div className="max-w-sm mx-auto w-full flex flex-col pt-12" style={{ minHeight: "100dvh" }}>
        {/* Header */}
        <div className="px-4 mb-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold" style={{ color: TEXT }}>
              Conversation
            </h1>
            <p className="text-sm mt-0.5" style={{ color: TEXT2 }}>
              ASL ↔ Voice
            </p>
          </div>
          {/* Connection status dot */}
          <div className="flex items-center gap-2">
            <div
              className="w-2 h-2 rounded-full"
              style={{
                backgroundColor:
                  status === "connected"
                    ? GREEN
                    : status === "connecting"
                      ? "#F59E0B"
                      : TEXT3,
              }}
            />
            <span className="text-xs" style={{ color: TEXT3 }}>
              {status === "connected"
                ? "Live"
                : status === "connecting"
                  ? "Connecting…"
                  : "Offline"}
            </span>
          </div>
        </div>

        {/* Chat area — fills available space, scrollable */}
        <div className="flex-1 overflow-y-auto px-4 pb-2" style={{ minHeight: 0 }}>
          {messages.length === 0 && !showSigningIndicator ? (
            <div className="flex flex-col items-center justify-center h-full gap-3 pb-12">
              <div
                className="w-14 h-14 rounded-2xl flex items-center justify-center mb-2"
                style={{ backgroundColor: PURPLE_LIGHT }}
              >
                <Hand size={28} style={{ color: PURPLE }} />
              </div>
              <p
                className="text-sm text-center max-w-xs"
                style={{ color: TEXT3 }}
              >
                {isActive
                  ? "Start signing — your words will appear here.\nYour partner can reply using the mic below."
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
                      className="flex flex-col gap-1"
                      style={{ maxWidth: "75%", alignItems: isAsl ? "flex-end" : "flex-start" }}
                    >
                      <div
                        className="px-4 py-3 text-sm leading-relaxed"
                        style={
                          isAsl
                            ? {
                                backgroundColor: PURPLE,
                                color: "#fff",
                                borderRadius: "18px 18px 4px 18px",
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
                      <span
                        className="text-xs px-1"
                        style={{ color: TEXT3 }}
                      >
                        {isAsl ? "🤚 ASL" : "🎤 Voice"} · {msg.timestamp}
                      </span>
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

              {/* Live: interim speech bubble (hearing user typing) */}
              {interimText && (
                <div className="flex justify-start items-end gap-2">
                  <div
                    className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0"
                    style={{ backgroundColor: "rgba(0,0,0,0.07)" }}
                  >
                    <Mic size={13} style={{ color: TEXT3 }} />
                  </div>
                  <div
                    className="px-4 py-3 text-sm leading-relaxed italic"
                    style={{
                      backgroundColor: CARD,
                      color: TEXT3,
                      borderRadius: "18px 18px 18px 4px",
                      boxShadow: SHADOW,
                      maxWidth: "75%",
                    }}
                  >
                    {interimText}
                  </div>
                </div>
              )}

              {/* Live: ASL signing / sentence-building indicator */}
              {showSigningIndicator && (
                <div className="flex justify-end items-end gap-2">
                  <div
                    className="px-4 py-3 text-sm"
                    style={{
                      backgroundColor: sentenceBuilding ? PURPLE_LIGHT : "rgba(124,111,224,0.06)",
                      color: sentenceBuilding ? PURPLE : TEXT3,
                      borderRadius: "18px 18px 4px 18px",
                      border: `1px solid ${sentenceBuilding ? "rgba(124,111,224,0.25)" : "rgba(0,0,0,0.06)"}`,
                      maxWidth: "75%",
                    }}
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-sm">{pendingLabel}</span>
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

        {/* Bottom controls — pb-24 clears the fixed bottom nav */}
        <div className="px-4 pb-24 pt-2 flex flex-col gap-2" style={{ borderTop: "1px solid rgba(0,0,0,0.06)" }}>

          {/* Mic — hold to speak (hearing user) */}
          {isActive && (
            <button
              onPointerDown={startListening}
              onPointerUp={stopListening}
              onPointerLeave={stopListening}
              className="w-full flex items-center justify-center gap-2 rounded-2xl py-4 font-semibold select-none cursor-pointer"
              style={{
                backgroundColor: isListening ? PURPLE : CARD,
                color: isListening ? "#fff" : TEXT2,
                boxShadow: SHADOW,
                fontSize: 15,
                border: isListening ? "none" : "1px solid rgba(0,0,0,0.06)",
                transition: "background-color 0.15s, color 0.15s",
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
              onClick={() => connect("ws://localhost:8765")}
              className="w-full flex items-center justify-center gap-2 rounded-2xl py-4 font-semibold cursor-pointer text-white"
              style={{ backgroundColor: PURPLE, fontSize: 15 }}
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
