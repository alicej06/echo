"use client";
import { useCallback, useEffect, useRef, useState } from "react";

export type MyoStatus = "disconnected" | "connecting" | "connected" | "demo";

export interface TopKEntry {
  letter: string;
  score: number;
}

export interface TrainStatus {
  [letter: string]: number; // letter -> count of recordings (0-5)
}

export interface MyoState {
  status: MyoStatus;
  currentLetter: string;
  confidence: number;
  topK: TopKEntry[];
  letterStream: string[];
  sentence: string;
  deviceName: string;
  trainStatus: TrainStatus;
  modelReady: boolean;
}

const DEMO_LETTERS = "HELLO WORLD ECHO ASL".split("").filter((c) => c !== " ");
const DEMO_SENTENCES = [
  "Hello, nice to meet you.",
  "How are you today?",
  "My name is Echo.",
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
const ASL_LETTERS = "abcdefghijklmnopqrstuvwxyz".split("");

function emptyTrainStatus(): TrainStatus {
  return Object.fromEntries(ASL_LETTERS.map((l) => [l, 0]));
}

export function useMyoWs() {
  const [state, setState] = useState<MyoState>({
    status: "disconnected",
    currentLetter: "",
    confidence: 0,
    topK: [],
    letterStream: [],
    sentence: "",
    deviceName: "",
    trainStatus: emptyTrainStatus(),
    modelReady: false,
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
      topK: [],
      letterStream: [],
      sentence: "",
      deviceName: "Demo",
    }));

    demoIntervalRef.current = setInterval(() => {
      const letter = DEMO_LETTERS[demoIndexRef.current % DEMO_LETTERS.length];
      demoIndexRef.current += 1;
      const conf = 0.4 + Math.random() * 0.55;
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
          topK: [{ letter, score: conf }],
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
        trainStatus: emptyTrainStatus(),
        modelReady: false,
      }));

      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setState((s) => ({ ...s, status: "connected" }));
      };

      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data as string) as Record<string, unknown>;

          if (msg.type === "letter") {
            const letter = String(msg.letter ?? "").toUpperCase();
            const confidence = Number(msg.confidence ?? 0);
            const topK = ((msg.top_k as [string, number][]) ?? []).map(
              ([l, s]) => ({ letter: l, score: s }),
            );
            setState((s) => ({
              ...s,
              currentLetter: letter,
              confidence,
              topK,
              letterStream: [...s.letterStream, letter].slice(-MAX_STREAM),
            }));
          } else if (msg.type === "sentence") {
            setState((s) => ({ ...s, sentence: String(msg.text ?? "") }));
          } else if (msg.type === "status") {
            setState((s) => ({
              ...s,
              deviceName: String(msg.device ?? "Myo"),
              status: msg.connected ? "connected" : "disconnected",
            }));
          } else if (msg.type === "train_ack") {
            const letter = String(msg.letter ?? "").toLowerCase();
            const count = Number(msg.count ?? 0);
            setState((s) => ({
              ...s,
              trainStatus: { ...s.trainStatus, [letter]: count },
            }));
          } else if (msg.type === "model_ready") {
            setState((s) => ({ ...s, modelReady: true }));
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

  // Send a message to the server
  const send = useCallback((payload: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(payload));
    }
  }, []);

  // Training commands
  const trainRecord = useCallback(
    (letter: string) => {
      send({ type: "train_record", letter: letter.toLowerCase() });
    },
    [send],
  );

  const trainModel = useCallback(() => {
    send({ type: "train_model" });
  }, [send]);

  const sendCorrection = useCallback(
    (letter: string) => {
      send({ type: "correction", letter: letter.toLowerCase() });
    },
    [send],
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

  return {
    ...state,
    connect,
    disconnect,
    startDemo,
    clearStream,
    trainRecord,
    trainModel,
    sendCorrection,
  };
}
