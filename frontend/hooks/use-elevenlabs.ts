"use client";
import { useState, useRef, useCallback, useEffect } from "react";

// Requires NEXT_PUBLIC_ELEVENLABS_API_KEY in .env.local
// Voice ID is read dynamically from localStorage (maia_prefs.selectedVoiceId) so
// the user's choice on the Profile page takes effect immediately.
const API_KEY = process.env.NEXT_PUBLIC_ELEVENLABS_API_KEY ?? "";
const FALLBACK_VOICE_ID =
  process.env.NEXT_PUBLIC_ELEVENLABS_VOICE_ID ?? "l4Coq6695JDX9xtLqXDE"; // Lauren

interface AudioPrefs {
  voiceId: string;
  speed: number;   // 0.6 – 1.5 (maps from the 60–150 slider)
  volume: number;  // 0 – 1   (maps from the 0–100 slider)
}

function getAudioPrefs(): AudioPrefs {
  if (typeof window === "undefined") {
    return { voiceId: FALLBACK_VOICE_ID, speed: 1.0, volume: 1.0 };
  }
  try {
    const prefs = JSON.parse(localStorage.getItem("maia_prefs") ?? "{}") as Record<string, unknown>;
    return {
      voiceId: (prefs.selectedVoiceId as string) ?? FALLBACK_VOICE_ID,
      speed:   typeof prefs.voiceRate === "number" ? prefs.voiceRate : 1.0,
      volume:  typeof prefs.volume    === "number" ? prefs.volume / 100 : 1.0,
    };
  } catch {
    return { voiceId: FALLBACK_VOICE_ID, speed: 1.0, volume: 1.0 };
  }
}

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

        // Read prefs fresh on every speak() call so slider changes take effect immediately
        const { voiceId, speed, volume } = getAudioPrefs();

        const res = await fetch(
          `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`,
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
                speed,          // passed to ElevenLabs (0.7–1.2 range)
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
        audio.volume       = volume;        // 0–1 applied to the audio element
        audio.playbackRate = speed;         // also applied locally for instant feel
        audioRef.current   = audio;

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
