"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import { Mic, Square } from "lucide-react";
import { useMyoWs } from "@/hooks/use-myo-ws";

const BG = "#F0EFF8";
const CARD = "#FFFFFF";
const PURPLE = "#7C6FE0";
const PURPLE_LIGHT = "rgba(124,111,224,0.12)";
const TEXT = "#1C1C1E";
const TEXT2 = "#6C6C70";
const TEXT3 = "#8E8E93";
const SHADOW = "0 1px 4px rgba(0,0,0,0.07)";

type Mode = "two-way" | "asl-speech" | "speech-text";

interface Message {
  id: string;
  side: "asl" | "speech";
  text: string;
  timestamp: string;
}

function fmt(d: Date) {
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export default function ConversationPage() {
  const { status, sentence, letterStream, connect, disconnect, startDemo } = useMyoWs();
  const [messages, setMessages] = useState<Message[]>([]);
  const [isListening, setIsListening] = useState(false);
  const [mode, setMode] = useState<Mode>("two-way");
  const prevSentenceRef = useRef("");
  const recognizerRef = useRef<SpeechRecognition | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const isActive = status === "connected" || status === "demo";

  // Auto-start demo when page loads
  useEffect(() => {
    if (status === "disconnected") {
      startDemo();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Push ASL sentence as right-side message
  useEffect(() => {
    if (!sentence || sentence === prevSentenceRef.current) return;
    prevSentenceRef.current = sentence;
    setMessages((m) => [
      ...m,
      { id: Date.now().toString(), side: "asl", text: sentence, timestamp: fmt(new Date()) },
    ]);
    if (typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.cancel();
      const utt = new SpeechSynthesisUtterance(sentence);
      utt.rate = 0.95;
      window.speechSynthesis.speak(utt);
    }
  }, [sentence]);

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length]);

  const startListening = useCallback(() => {
    const Ctor = typeof window !== "undefined"
      ? (window.SpeechRecognition ?? window.webkitSpeechRecognition)
      : null;
    if (!Ctor) return;
    const rec = new Ctor();
    rec.continuous = false;
    rec.interimResults = false;
    rec.lang = "en-US";
    recognizerRef.current = rec;
    rec.onresult = (ev: SpeechRecognitionEvent) => {
      const text = ev.results[ev.resultIndex]?.[0]?.transcript ?? "";
      if (text.trim()) {
        setMessages((m) => [
          ...m,
          { id: Date.now().toString(), side: "speech", text: text.trim(), timestamp: fmt(new Date()) },
        ]);
      }
    };
    rec.onend = () => setIsListening(false);
    rec.start();
    setIsListening(true);
  }, []);

  const stopListening = useCallback(() => {
    recognizerRef.current?.stop();
    setIsListening(false);
  }, []);

  const MODES: { key: Mode; label: string }[] = [
    { key: "two-way", label: "Two-Way" },
    { key: "asl-speech", label: "ASL → Speech" },
    { key: "speech-text", label: "Speech → Text" },
  ];

  return (
    <main className="min-h-screen flex flex-col pb-20" style={{ backgroundColor: BG }}>
      <div className="max-w-sm mx-auto w-full flex flex-col flex-1 pt-12 px-4">
        {/* Header */}
        <div className="mb-5">
          <h1 className="text-2xl font-bold" style={{ color: TEXT }}>Conversation</h1>
          <p className="text-sm mt-1" style={{ color: TEXT2 }}>Two-way communication mode</p>
        </div>

        {/* Mode tabs */}
        <div className="flex gap-2 mb-5 overflow-x-auto pb-1" style={{ scrollbarWidth: "none" }}>
          {MODES.map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setMode(key)}
              className="flex-shrink-0 px-4 py-2 rounded-full text-sm font-medium cursor-pointer"
              style={
                mode === key
                  ? { backgroundColor: PURPLE, color: "#fff" }
                  : { backgroundColor: CARD, color: TEXT3, border: "1px solid rgba(0,0,0,0.08)" }
              }
            >
              {label}
            </button>
          ))}
        </div>

        {/* Chat area */}
        <div className="flex-1 overflow-y-auto mb-4" style={{ minHeight: 200 }}>
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-48 gap-3">
              <p className="text-sm text-center max-w-xs" style={{ color: TEXT3 }}>
                Start signing and your partner can reply by holding the mic button.
              </p>
            </div>
          ) : (
            <div className="flex flex-col gap-3 py-2">
              {messages.map((msg) => {
                const isAsl = msg.side === "asl";
                return (
                  <div key={msg.id} className={`flex ${isAsl ? "justify-end" : "justify-start"} gap-2`}>
                    {!isAsl && (
                      <div
                        className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 mt-1"
                        style={{ backgroundColor: "rgba(0,0,0,0.06)" }}
                      >
                        <Mic size={13} style={{ color: TEXT3 }} />
                      </div>
                    )}
                    <div className="flex flex-col gap-1 max-w-[75%]">
                      <p
                        className="text-xs font-medium"
                        style={{ color: TEXT3, textAlign: isAsl ? "right" : "left" }}
                      >
                        {isAsl ? "🤚 ASL User" : "Hearing User"}
                      </p>
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
                        style={{ color: TEXT3, textAlign: isAsl ? "right" : "left" }}
                      >
                        {msg.timestamp}
                      </span>
                    </div>
                    {isAsl && (
                      <div
                        className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 mt-6"
                        style={{ backgroundColor: PURPLE_LIGHT }}
                      >
                        <span style={{ fontSize: 12 }}>🤚</span>
                      </div>
                    )}
                  </div>
                );
              })}
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        {/* Active letter strip */}
        {isActive && letterStream.length > 0 && mode !== "speech-text" && (
          <div
            className="rounded-xl px-3 py-2 mb-3 flex gap-1 overflow-hidden"
            style={{ backgroundColor: CARD, boxShadow: SHADOW }}
          >
            {letterStream.slice(-24).map((l, i, arr) => (
              <span
                key={i}
                className="text-xs font-bold w-5 h-5 flex items-center justify-center rounded"
                style={{
                  color: i === arr.length - 1 ? PURPLE : TEXT3,
                  backgroundColor: i === arr.length - 1 ? PURPLE_LIGHT : "transparent",
                }}
              >
                {l}
              </span>
            ))}
          </div>
        )}

        {/* Mic button (speech → text) */}
        {mode !== "asl-speech" && (
          <button
            onMouseDown={startListening}
            onMouseUp={stopListening}
            onTouchStart={startListening}
            onTouchEnd={stopListening}
            className="w-full flex items-center justify-center gap-2 rounded-2xl py-4 font-semibold mb-3 cursor-pointer select-none"
            style={{
              backgroundColor: isListening ? PURPLE : CARD,
              color: isListening ? "#fff" : TEXT2,
              boxShadow: SHADOW,
              fontSize: 15,
              border: isListening ? "none" : "1px solid rgba(0,0,0,0.06)",
            }}
          >
            <Mic size={18} />
            {isListening ? "Listening..." : "Hold to Speak"}
          </button>
        )}

        {/* End / Start */}
        {isActive ? (
          <button
            onClick={disconnect}
            className="w-full flex items-center justify-center gap-2 rounded-2xl py-4 font-semibold cursor-pointer"
            style={{ backgroundColor: "#FF3B30", color: "#fff", fontSize: 15 }}
          >
            <Square size={16} fill="white" />
            End Conversation
          </button>
        ) : (
          <button
            onClick={() => connect("ws://localhost:8765")}
            className="w-full flex items-center justify-center gap-2 rounded-2xl py-4 font-semibold cursor-pointer"
            style={{ backgroundColor: PURPLE, color: "#fff", fontSize: 15 }}
          >
            Start Conversation
          </button>
        )}
      </div>
    </main>
  );
}
