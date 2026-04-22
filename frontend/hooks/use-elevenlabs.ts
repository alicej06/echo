"use client";
import { useState, useRef, useCallback, useEffect } from "react";

// Requires NEXT_PUBLIC_ELEVENLABS_API_KEY in .env.local
// Voice ID is read dynamically from localStorage (maia_prefs.selectedVoiceId) so
// the user's choice on the Profile page takes effect immediately.
const API_KEY = process.env.NEXT_PUBLIC_ELEVENLABS_API_KEY ?? "";
const FALLBACK_VOICE_ID =
  process.env.NEXT_PUBLIC_ELEVENLABS_VOICE_ID ?? "l4Coq6695JDX9xtLqXDE"; // Lauren

function getVoiceId(): string {
  if (typeof window === "undefined") return FALLBACK_VOICE_ID;
  try {
    const prefs = JSON.parse(localStorage.getItem("maia_prefs") ?? "{}") as Record<string, unknown>;
    return (prefs.selectedVoiceId as string) ?? FALLBACK_VOICE_ID;
  } catch {
    return FALLBACK_VOICE_ID;
  }
}

export interface UseElevenLabsReturn {
  speak: (text: string) => Promise<void>;
  isSpeaking: boolean;
}

export function useElevenLabs(): UseElevenLabsReturn {
  const [isSpeaking, setIsSpeaking] = useState(false);
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

  const speak = useCallback(
    async (text: string) => {
      if (!text.trim()) return;

      if (!API_KEY) {
        console.warn("[useElevenLabs] NEXT_PUBLIC_ELEVENLABS_API_KEY is not set");
        return;
      }

      cancelCurrent();

      try {
        setIsSpeaking(true);

        const res = await fetch(
          `https://api.elevenlabs.io/v1/text-to-speech/${getVoiceId()}`,
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
    [cancelCurrent]
  );

  // Clean up on unmount
  useEffect(
    () => () => {
      cancelCurrent();
    },
    [cancelCurrent]
  );

  return { speak, isSpeaking };
}
