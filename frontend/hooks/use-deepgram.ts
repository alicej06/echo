"use client";
import { useState, useRef, useCallback, useEffect } from "react";

// Requires NEXT_PUBLIC_DEEPGRAM_API_KEY in .env.local
const API_KEY = process.env.NEXT_PUBLIC_DEEPGRAM_API_KEY ?? "";
const WS_URL = "wss://api.deepgram.com/v1/listen";

export interface UseDeepgramReturn {
  isListening: boolean;
  interimTranscript: string;
  finalTranscript: string;
  micPermission: "prompt" | "granted" | "denied";
  startListening: () => Promise<void>;
  stopListening: () => void;
}

export function useDeepgram(): UseDeepgramReturn {
  const [isListening, setIsListening] = useState(false);
  const [interimTranscript, setInterimTranscript] = useState("");
  const [finalTranscript, setFinalTranscript] = useState("");
  const [micPermission, setMicPermission] = useState<"prompt" | "granted" | "denied">("prompt");

  const wsRef = useRef<WebSocket | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const stopListening = useCallback(() => {
    recorderRef.current?.stop();
    wsRef.current?.close();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    recorderRef.current = null;
    wsRef.current = null;
    streamRef.current = null;
    setIsListening(false);
    setInterimTranscript("");
  }, []);

  const startListening = useCallback(async () => {
    if (!API_KEY) {
      console.warn("[useDeepgram] NEXT_PUBLIC_DEEPGRAM_API_KEY is not set");
      return;
    }

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      streamRef.current = stream;
      setMicPermission("granted");
    } catch (err: unknown) {
      if (err instanceof Error && err.name === "NotAllowedError") {
        setMicPermission("denied");
      }
      return;
    }

    const params = new URLSearchParams({
      model: "nova-2",
      interim_results: "true",
      punctuate: "true",
      endpointing: "400",
    });

    // Deepgram browser auth: pass token as a subprotocol header
    const ws = new WebSocket(`${WS_URL}?${params}`, ["token", API_KEY]);
    wsRef.current = ws;

    ws.onopen = () => {
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";

      const recorder = new MediaRecorder(stream, { mimeType });
      recorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0 && ws.readyState === WebSocket.OPEN) {
          ws.send(e.data);
        }
      };

      recorder.start(250); // send chunks every 250 ms
      setIsListening(true);
    };

    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data as string);
        if (data.type !== "Results") return;
        const transcript: string = data.channel?.alternatives?.[0]?.transcript ?? "";

        if (data.speech_final && data.is_final) {
          // End of utterance — commit as a final result
          setInterimTranscript("");
          if (transcript.trim()) setFinalTranscript(transcript.trim());
        } else {
          // Still speaking — show as live preview
          setInterimTranscript(transcript);
        }
      } catch {
        // ignore malformed frames
      }
    };

    ws.onerror = () => stopListening();
    ws.onclose = () => {
      setIsListening(false);
      setInterimTranscript("");
    };
  }, [stopListening]);

  // Clean up on unmount
  useEffect(() => () => stopListening(), [stopListening]);

  return { isListening, interimTranscript, finalTranscript, micPermission, startListening, stopListening };
}
