"use client";
import { useState, useRef, useCallback, useEffect } from "react";

// Requires NEXT_PUBLIC_ELEVENLABS_API_KEY and NEXT_PUBLIC_ELEVENLABS_VOICE_ID in .env.local
const API_KEY = process.env.NEXT_PUBLIC_ELEVENLABS_API_KEY ?? "";
const VOICE_ID =
  process.env.NEXT_PUBLIC_ELEVENLABS_VOICE_ID ?? "21m00Tcm4TlvDq8ikWAM"; // Rachel

export interface UseElevenLabsReturn {
  speak: (text: string) => Promise<void>;
  isSpeaking: boolean;
  muted: boolean;
  toggleMute: () => void;
}

export function useElevenLabs(): UseElevenLabsReturn {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [muted, setMuted] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const blobUrlRef = useRef<string | null>(null);

  const cancelCurrent = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.onended = null;
      audioRef.current.onerror = null;
      audioRef.current = null;
    }
    if (blobUrlRef.current) {
      URL.revokeObjectURL(blobUrlRef.current);
      blobUrlRef.current = null;
    }
  }, []);

  const toggleMute = useCallback(() => {
    setMuted((m) => {
      if (!m) cancelCurrent();
      return !m;
    });
  }, [cancelCurrent]);

  const speak = useCallback(
    async (text: string) => {
      if (!text.trim() || muted) return;

      if (!API_KEY) {
        console.warn("[useElevenLabs] NEXT_PUBLIC_ELEVENLABS_API_KEY is not set");
        return;
      }

      cancelCurrent();

      try {
        setIsSpeaking(true);

        const res = await fetch(
          `https://api.elevenlabs.io/v1/text-to-speech/${VOICE_ID}`,
          {
            method: "POST",
            headers: {
              "xi-api-key": API_KEY,
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              text,
              model_id: "eleven_turbo_v2_5",
              voice_settings: {
                stability: 0.5,
                similarity_boost: 0.75,
              },
            }),
          }
        );

        if (!res.ok) {
          console.error("[useElevenLabs] API error:", res.status, await res.text());
          setIsSpeaking(false);
          return;
        }

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        blobUrlRef.current = url;

        const audio = new Audio(url);
        audioRef.current = audio;

        audio.onended = () => {
          setIsSpeaking(false);
          URL.revokeObjectURL(url);
          blobUrlRef.current = null;
          audioRef.current = null;
        };

        audio.onerror = () => {
          setIsSpeaking(false);
        };

        await audio.play();
      } catch (err) {
        console.error("[useElevenLabs] Playback error:", err);
        setIsSpeaking(false);
      }
    },
    [cancelCurrent, muted]
  );

  // Clean up on unmount
  useEffect(
    () => () => {
      cancelCurrent();
    },
    [cancelCurrent]
  );

  return { speak, isSpeaking, muted, toggleMute };
}
