"use client";
import { useCallback, useEffect, useRef, useState } from "react";

export type MyoStatus = "disconnected" | "connecting" | "connected" | "demo";

export interface MyoState {
  status: MyoStatus;
  currentLetter: string;
  confidence: number;
  letterStream: string[];
  sentence: string;
  deviceName: string;
}

const DEMO_LETTERS = "HELLO WORLD MAIA ASL".split("").filter((c) => c !== " ");
const DEMO_SENTENCES = [
  "Hello, nice to meet you.",
  "How are you today?",
  "My name is MAIA.",
  "I use ASL to communicate.",
  "Can we meet tomorrow?",
  "Thank you for your help.",
  "I need a moment please.",
  "That sounds good to me.",
];
const WS_URL_DEFAULT: string =
  (typeof process !== "undefined" &&
  typeof process.env?.NEXT_PUBLIC_MAIA_WS_URL === "string"
    ? process.env.NEXT_PUBLIC_MAIA_WS_URL
    : null) ?? "ws://localhost:8765";

const MAX_STREAM = 100;

export function useMyoWs() {
  const [state, setState] = useState<MyoState>({
    status: "disconnected",
    currentLetter: "",
    confidence: 0,
    letterStream: [],
    sentence: "",
    deviceName: "",
  });

  const wsRef = useRef<WebSocket | null>(null);
  const demoIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const demoIndexRef = useRef(0);

  const stopDemo = useCallback(() => {
    if (demoIntervalRef.current) {
      clearInterval(demoIntervalRef.current);
      demoIntervalRef.current = null;
    }
  }, []);

  const disconnect = useCallback(() => {
    stopDemo();
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
      wsRef.current = null;
    }
    setState((s) => ({ ...s, status: "disconnected" }));
  }, [stopDemo]);

  const startDemo = useCallback(() => {
    disconnect();
    demoIndexRef.current = 0;
    setState((s) => ({
      ...s,
      status: "demo",
      currentLetter: "",
      confidence: 0,
      letterStream: [],
      sentence: "",
      deviceName: "Demo",
    }));

    demoIntervalRef.current = setInterval(() => {
      const letter = DEMO_LETTERS[demoIndexRef.current % DEMO_LETTERS.length];
      demoIndexRef.current += 1;
      const conf = 0.7 + Math.random() * 0.29;
      // Emit a realistic demo sentence every full cycle through the letters
      const sentenceIdx =
        Math.floor(demoIndexRef.current / DEMO_LETTERS.length) %
        DEMO_SENTENCES.length;
      const emitSentence = demoIndexRef.current % DEMO_LETTERS.length === 0;

      setState((s) => {
        const nextStream = [...s.letterStream, letter].slice(-MAX_STREAM);
        return {
          ...s,
          currentLetter: letter,
          confidence: conf,
          letterStream: nextStream,
          sentence: emitSentence ? DEMO_SENTENCES[sentenceIdx] : s.sentence,
        };
      });
    }, 600);
  }, [disconnect]);

  const connect = useCallback(
    (wsUrl?: string) => {
      disconnect();
      const url = wsUrl || WS_URL_DEFAULT;
      setState((s) => ({
        ...s,
        status: "connecting",
        currentLetter: "",
        letterStream: [],
        sentence: "",
      }));

      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setState((s) => ({ ...s, status: "connected" }));
      };

      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data as string) as Record<string, unknown>;
          if (msg.type === "letter" || ("label" in msg && !msg.type)) {
            const letter = String(msg.letter ?? msg.label ?? "").toUpperCase();
            const confidence = Number(msg.confidence ?? 0);
            setState((s) => {
              const nextStream = [...s.letterStream, letter].slice(-MAX_STREAM);
              return {
                ...s,
                currentLetter: letter,
                confidence,
                letterStream: nextStream,
              };
            });
          } else if (msg.type === "sentence") {
            const text = String(msg.text ?? "");
            setState((s) => ({ ...s, sentence: text }));
          } else if (msg.type === "status") {
            const deviceName = String(msg.device ?? "Myo");
            setState((s) => ({
              ...s,
              deviceName,
              status: msg.connected ? "connected" : "disconnected",
            }));
          }
        } catch {
          // ignore malformed frames
        }
      };

      ws.onerror = () => {
        setState((s) => ({ ...s, status: "disconnected" }));
      };

      ws.onclose = () => {
        setState((s) => {
          if (s.status !== "demo") return { ...s, status: "disconnected" };
          return s;
        });
      };
    },
    [disconnect],
  );

  const clearStream = useCallback(() => {
    setState((s) => ({
      ...s,
      currentLetter: "",
      letterStream: [],
      sentence: "",
    }));
  }, []);

  useEffect(() => {
    return () => {
      stopDemo();
      wsRef.current?.close();
    };
  }, [stopDemo]);

  return { ...state, connect, disconnect, startDemo, clearStream };
}
