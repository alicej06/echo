"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import { Mic, MicOff, Radio, Play, StopCircle } from "lucide-react";
import { useMyoWs } from "@/hooks/use-myo-ws";
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
  const { status, sentence, letterStream, connect, disconnect, startDemo } =
    useMyoWs();
  const [messages, setMessages] = useState<Message[]>([]);
  const [wsUrl] = useState("ws://localhost:8765");
  const [isListening, setIsListening] = useState(false);
  const prevSentenceRef = useRef("");
  const recognizerRef = useRef<SpeechRecognition | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const isActive = status === "connected" || status === "demo";

  // Push ASL sentence to chat
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
    const SpeechRecognitionCtor =
      typeof window !== "undefined"
        ? (window.SpeechRecognition ?? window.webkitSpeechRecognition)
        : null;
    if (!SpeechRecognitionCtor) return;

    const rec = new SpeechRecognitionCtor();
    rec.continuous = false;
    rec.interimResults = false;
    rec.lang = "en-US";
    recognizerRef.current = rec;

    rec.onresult = (ev: SpeechRecognitionEvent) => {
      const transcript = ev.results[ev.resultIndex]?.[0]?.transcript ?? "";
      if (transcript.trim()) {
        setMessages((m) => [
          ...m,
          {
            id: Date.now().toString(),
            side: "speech",
            text: transcript.trim(),
            timestamp: fmt(new Date()),
          },
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

  const statusColor = {
    disconnected: "#52525b",
    connecting: "#eab308",
    connected: "#22c55e",
    demo: "#06b6d4",
  }[status];

  return (
    <main
      className="min-h-screen pt-16 flex flex-col"
      style={{ backgroundColor: "#0a0a0a" }}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-5 max-w-3xl mx-auto w-full">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">
            Conversation
          </h1>
          <p className="text-sm mt-0.5" style={{ color: "#52525b" }}>
            ASL signing meets spoken reply
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
        </div>
      </div>

      {/* Chat area */}
      <div className="flex-1 overflow-y-auto px-4 pb-4 max-w-3xl mx-auto w-full">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 gap-3">
            <div
              className="w-14 h-14 rounded-2xl flex items-center justify-center"
              style={{ background: "rgba(255,255,255,0.04)" }}
            >
              <Radio className="w-6 h-6" style={{ color: "#3f3f46" }} />
            </div>
            <p
              className="text-sm text-center max-w-xs"
              style={{ color: "#3f3f46" }}
            >
              Connect Myo and start signing. Your partner can respond by holding
              the mic button.
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-3 py-4">
            {messages.map((msg) => {
              const isAsl = msg.side === "asl";
              return (
                <div
                  key={msg.id}
                  className={`flex ${isAsl ? "justify-start" : "justify-end"} gap-2`}
                >
                  {isAsl && (
                    <div
                      className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 mt-1 text-xs font-bold"
                      style={{
                        background: "rgba(6,182,212,0.2)",
                        color: "#22d3ee",
                      }}
                    >
                      M
                    </div>
                  )}
                  <div className="flex flex-col gap-1 max-w-xs md:max-w-sm">
                    <div
                      className="px-4 py-2.5 rounded-2xl text-sm leading-relaxed"
                      style={
                        isAsl
                          ? {
                              backgroundColor: "rgba(6,182,212,0.15)",
                              color: "#e4e4e7",
                              borderRadius: "4px 18px 18px 18px",
                              border: "1px solid rgba(6,182,212,0.2)",
                            }
                          : {
                              backgroundColor: "#6d28d9",
                              color: "#ffffff",
                              borderRadius: "18px 4px 18px 18px",
                            }
                      }
                    >
                      {msg.text}
                    </div>
                    <span
                      className={`text-xs px-1 ${isAsl ? "text-left" : "text-right"}`}
                      style={{ color: "#3f3f46" }}
                    >
                      {msg.timestamp}
                    </span>
                  </div>
                  {!isAsl && (
                    <div
                      className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 mt-1 text-xs font-bold"
                      style={{
                        background: "rgba(109,40,217,0.2)",
                        color: "#a78bfa",
                      }}
                    >
                      Y
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
      {isActive && letterStream.length > 0 && (
        <div className="px-4 pb-2 max-w-3xl mx-auto w-full">
          <div
            className="rounded-xl px-3 py-2 flex gap-1 overflow-hidden"
            style={{
              background: "rgba(255,255,255,0.03)",
              border: "1px solid rgba(255,255,255,0.06)",
            }}
          >
            {letterStream.slice(-20).map((l, i, arr) => (
              <span
                key={i}
                className="text-xs font-bold w-5 h-5 flex items-center justify-center rounded"
                style={{
                  color: i === arr.length - 1 ? "#22d3ee" : "#3f3f46",
                  backgroundColor:
                    i === arr.length - 1
                      ? "rgba(6,182,212,0.15)"
                      : "transparent",
                }}
              >
                {l}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Controls */}
      <div
        className="border-t px-4 py-4 max-w-3xl mx-auto w-full"
        style={{ borderColor: "rgba(255,255,255,0.07)" }}
      >
        <div className="flex items-center gap-3">
          {/* Connection */}
          {!isActive ? (
            <>
              <button
                onClick={() => connect(wsUrl)}
                className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium text-white transition-all duration-200 cursor-pointer"
                style={{ background: "rgba(6,182,212,0.7)" }}
              >
                <Radio className="w-4 h-4" />
                Connect
              </button>
              <button
                onClick={startDemo}
                className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 cursor-pointer"
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
              className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 cursor-pointer"
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

          {/* Mic hold-to-speak */}
          <button
            onMouseDown={startListening}
            onMouseUp={stopListening}
            onTouchStart={startListening}
            onTouchEnd={stopListening}
            className="ml-auto flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium text-sm transition-all duration-200 cursor-pointer select-none"
            style={
              isListening
                ? {
                    background: "rgba(109,40,217,0.7)",
                    color: "#fff",
                    border: "1px solid rgba(109,40,217,0.5)",
                    boxShadow: "0 0 20px rgba(109,40,217,0.3)",
                  }
                : {
                    background: "rgba(255,255,255,0.06)",
                    border: "1px solid rgba(255,255,255,0.1)",
                    color: "#a1a1aa",
                  }
            }
            aria-label="Hold to speak"
          >
            {isListening ? (
              <Mic className="w-4 h-4" />
            ) : (
              <MicOff className="w-4 h-4" />
            )}
            {isListening ? "Listening..." : "Hold to speak"}
          </button>
        </div>
      </div>
    </main>
  );
}
