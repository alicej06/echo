"use client";
import { useState, useRef, useCallback, useEffect } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft, Mic, Square, Play, Pause, Check, Loader2, Sparkles, AlertCircle } from "lucide-react";

const BG     = "linear-gradient(180deg, #9147C8 0%, #A066D8 30%, #C49AEE 65%, #DDD0F8 85%, #EDE8FF 100%)";
const CARD   = "rgba(255,255,255,0.82)";
const PURPLE = "#7C6FE0";
const GREEN  = "#34C759";
const TEXT   = "#1C1C1E";
const TEXT2  = "#6C6C70";
const TEXT3  = "#8E8E93";
const SHADOW = "0 2px 12px rgba(80,0,150,0.1)";
const ON_BG  = "#fff";
const ON_BG2 = "rgba(255,255,255,0.8)";
const ON_BG3 = "rgba(255,255,255,0.55)";

const API_KEY      = process.env.NEXT_PUBLIC_ELEVENLABS_API_KEY ?? "";
const MIN_RECORD_S = 30;
const MAX_RECORD_S = 120;

type Tab   = "clone" | "design";
type Stage = "idle" | "recording" | "recorded" | "loading" | "preview" | "saved";

// ── Helpers ────────────────────────────────────────────────────────────────

function saveVoiceToProfile(voiceId: string, voiceName: string) {
  try {
    const raw   = localStorage.getItem("maia_prefs");
    const prefs = raw ? (JSON.parse(raw) as Record<string, unknown>) : {};
    prefs.selectedVoiceId = voiceId;
    prefs.selectedVoice   = voiceName;
    localStorage.setItem("maia_prefs", JSON.stringify(prefs));

    const libRaw  = localStorage.getItem("maia_custom_voices");
    const library = libRaw ? (JSON.parse(libRaw) as { name: string; voiceId: string }[]) : [];
    if (!library.some((v) => v.voiceId === voiceId)) {
      library.push({ name: voiceName, voiceId });
      localStorage.setItem("maia_custom_voices", JSON.stringify(library));
    }
  } catch { /* ignore */ }
}

function fmt(s: number) {
  const safe = isNaN(s) || !isFinite(s) ? 0 : Math.floor(s);
  return `${Math.floor(safe / 60)}:${String(safe % 60).padStart(2, "0")}`;
}

// ── Custom audio player (replaces glitchy native <audio controls>) ─────────

function AudioPlayer({ src }: { src: string }) {
  const audioRef            = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying]   = useState(false);
  const [current, setCurrent]   = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    const a = new Audio(src);
    audioRef.current = a;
    a.onloadedmetadata = () => setDuration(a.duration);
    a.ontimeupdate     = () => setCurrent(a.currentTime);
    a.onended          = () => { setPlaying(false); setCurrent(0); };
    a.onerror          = () => setPlaying(false);
    return () => { a.pause(); a.src = ""; audioRef.current = null; };
  }, [src]);

  const toggle = () => {
    const a = audioRef.current;
    if (!a) return;
    if (playing) {
      a.pause();
      setPlaying(false);
    } else {
      a.play().then(() => setPlaying(true)).catch(() => {});
    }
  };

  const seek = (e: React.MouseEvent<HTMLDivElement>) => {
    const a = audioRef.current;
    if (!a || !duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    a.currentTime = ((e.clientX - rect.left) / rect.width) * duration;
  };

  const pct = duration > 0 ? Math.min((current / duration) * 100, 100) : 0;

  return (
    <div className="flex items-center gap-3 p-3 rounded-xl"
      style={{ backgroundColor: "rgba(0,0,0,0.06)" }}>
      {/* Play / Pause */}
      <button
        onClick={toggle}
        className="w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0 cursor-pointer"
        style={{ backgroundColor: PURPLE, border: "none" }}
      >
        {playing
          ? <Pause size={14} fill={ON_BG} style={{ color: ON_BG }} />
          : <Play  size={14} fill={ON_BG} style={{ color: ON_BG, marginLeft: 1 }} />}
      </button>

      {/* Track + times */}
      <div className="flex-1 flex flex-col gap-1.5">
        <div
          onClick={seek}
          className="w-full rounded-full cursor-pointer"
          style={{ height: 4, backgroundColor: "rgba(0,0,0,0.12)", position: "relative" }}
        >
          <div style={{
            position: "absolute", left: 0, top: 0, bottom: 0,
            width: `${pct}%`,
            backgroundColor: PURPLE,
            borderRadius: 99,
            transition: "width 0.1s linear",
          }} />
        </div>
        <div className="flex justify-between" style={{ fontSize: 10, color: TEXT3 }}>
          <span>{fmt(current)}</span>
          <span>{fmt(duration)}</span>
        </div>
      </div>
    </div>
  );
}

// ── Clone tab ──────────────────────────────────────────────────────────────

function CloneTab() {
  const [stage,       setStage]       = useState<Stage>("idle");
  const [elapsed,     setElapsed]     = useState(0);
  const [audioUrl,    setAudioUrl]    = useState<string | null>(null);
  const [voiceName,   setVoiceName]   = useState("");
  const [submitError, setSubmitError] = useState("");   // separate so errors don't wipe the recording
  const [savedName,   setSavedName]   = useState("");

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef   = useRef<MediaStream | null>(null);
  const chunksRef   = useRef<Blob[]>([]);
  const blobRef     = useRef<Blob | null>(null);
  const timerRef    = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopTimer = () => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
  };

  const doStop = useCallback(() => {
    stopTimer();
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const doStopRef = useRef(doStop);
  doStopRef.current = doStop;

  const startRecording = useCallback(async () => {
    setSubmitError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      chunksRef.current = [];
      setElapsed(0);

      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus" : "audio/webm";
      const rec = new MediaRecorder(stream, { mimeType });
      recorderRef.current = rec;

      rec.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      rec.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType });
        blobRef.current = blob;
        const url = URL.createObjectURL(blob);
        setAudioUrl(url);
        setStage("recorded");
        streamRef.current?.getTracks().forEach((t) => t.stop());
      };

      rec.start(250);
      setStage("recording");

      let secs = 0;
      timerRef.current = setInterval(() => {
        secs += 1;
        setElapsed(secs);
        if (secs >= MAX_RECORD_S) doStopRef.current();
      }, 1000);
    } catch {
      setSubmitError("Microphone access denied. Allow mic in browser settings.");
    }
  }, []);

  const stopRecording = useCallback(() => { doStopRef.current(); }, []);

  const submitClone = useCallback(async () => {
    if (!blobRef.current || !voiceName.trim()) return;
    if (!API_KEY) { setSubmitError("NEXT_PUBLIC_ELEVENLABS_API_KEY not set"); return; }

    setStage("loading");
    setSubmitError("");
    try {
      const form = new FormData();
      form.append("name",  voiceName.trim());
      form.append("files", blobRef.current, "echo_voice.webm");

      const res = await fetch("https://api.elevenlabs.io/v1/voices/add", {
        method: "POST",
        headers: { "xi-api-key": API_KEY },
        body: form,
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({})) as Record<string, unknown>;
        const detail = (body.detail as Record<string, unknown>)?.message
          ?? (body.detail as string) ?? `Error ${res.status}`;
        throw new Error(String(detail));
      }

      const { voice_id } = await res.json() as { voice_id: string };
      saveVoiceToProfile(voice_id, voiceName.trim());
      setSavedName(voiceName.trim());
      setStage("saved");
    } catch (err: unknown) {
      // Keep stage as "recorded" so user can fix name or retry — don't wipe their recording
      setSubmitError(err instanceof Error ? err.message : "Something went wrong. Try again.");
      setStage("recorded");
    }
  }, [voiceName]);

  useEffect(() => () => {
    stopTimer();
    streamRef.current?.getTracks().forEach((t) => t.stop());
  }, []);

  const hasEnough = elapsed >= MIN_RECORD_S;

  if (stage === "saved") {
    return (
      <div className="flex flex-col items-center gap-6 pt-4 text-center">
        <div className="w-20 h-20 rounded-2xl flex items-center justify-center"
          style={{ backgroundColor: "rgba(52,199,89,0.2)", border: "1px solid rgba(52,199,89,0.4)" }}>
          <Check size={40} style={{ color: GREEN }} />
        </div>
        <div>
          <p className="text-2xl font-bold mb-1" style={{ color: ON_BG }}>{savedName}</p>
          <p className="text-sm" style={{ color: ON_BG2 }}>
            Voice cloned and added to your voice library.
          </p>
          <p className="text-xs mt-1" style={{ color: ON_BG3 }}>
            Find it in Settings → Voice → Your Voices
          </p>
        </div>
        <button
          onClick={() => { setStage("idle"); setVoiceName(""); setAudioUrl(null); setElapsed(0); setSubmitError(""); }}
          className="w-full rounded-2xl py-3 text-sm font-semibold cursor-pointer"
          style={{ backgroundColor: "rgba(255,255,255,0.2)", color: ON_BG, border: "1px solid rgba(255,255,255,0.35)" }}>
          Clone another voice
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5">
      {/* Instructions */}
      <div className="rounded-2xl p-4" style={{ backgroundColor: CARD, boxShadow: SHADOW }}>
        <p className="text-sm font-semibold mb-1" style={{ color: TEXT }}>How it works</p>
        <p className="text-sm leading-relaxed" style={{ color: TEXT2 }}>
          Record at least <strong>30 seconds</strong> of natural speech — read a passage, tell a story,
          or just talk. The more you record the better the clone sounds.
        </p>
      </div>

      {/* Voice name input — always visible so user knows it's required */}
      <div>
        <input
          type="text"
          placeholder="Name this voice (e.g. My Voice)"
          value={voiceName}
          onChange={(e) => setVoiceName(e.target.value)}
          className="w-full rounded-2xl px-4 py-3 text-sm outline-none"
          style={{ backgroundColor: CARD, boxShadow: SHADOW, color: TEXT,
            border: `2px solid ${voiceName.trim() ? PURPLE : "transparent"}` }}
        />
        {!voiceName.trim() && (
          <p className="text-xs mt-1.5 ml-1" style={{ color: ON_BG3 }}>
            ↑ Enter a name — you&apos;ll need this to save
          </p>
        )}
      </div>

      {/* Record button — hide while showing the recorded playback */}
      {stage !== "recorded" && stage !== "loading" && (
        <div className="flex flex-col items-center gap-4">
          <div className="relative flex items-center justify-center">
            {stage === "recording" && (
              <div className="absolute rounded-full"
                style={{ width: 120, height: 120, backgroundColor: "rgba(255,59,48,0.12)",
                  animation: "pulse-ring 1s ease-out infinite" }} />
            )}
            <button
              onClick={stage === "recording" ? stopRecording : startRecording}
              className="w-24 h-24 rounded-full flex items-center justify-center cursor-pointer select-none"
              style={{
                backgroundColor: stage === "recording" ? "#FF3B30" : "rgba(255,255,255,0.2)",
                border: `3px solid ${stage === "recording" ? "#FF3B30" : ON_BG}`,
                boxShadow: stage === "recording" ? "0 0 0 6px rgba(255,59,48,0.2)" : SHADOW,
                transition: "all 0.2s",
              }}>
              {stage === "recording"
                ? <Square size={28} fill={ON_BG} style={{ color: ON_BG }} />
                : <Mic size={28} style={{ color: ON_BG }} />}
            </button>
          </div>

          {stage === "recording" && (
            <>
              <div className="flex flex-col items-center gap-1">
                <p className="text-2xl font-bold tabular-nums" style={{ color: ON_BG }}>{fmt(elapsed)}</p>
                <p className="text-xs" style={{ color: hasEnough ? GREEN : ON_BG3 }}>
                  {hasEnough ? "✓ Enough to clone — keep going for better quality" : `${MIN_RECORD_S - elapsed}s more needed`}
                </p>
              </div>
              <button
                onClick={stopRecording}
                className="w-full rounded-2xl py-3 text-sm font-semibold cursor-pointer"
                style={{ backgroundColor: "rgba(255,59,48,0.18)", color: "#FF3B30",
                  border: "1px solid rgba(255,59,48,0.35)" }}>
                Stop Recording
              </button>
            </>
          )}

          {stage === "idle" && (
            <p className="text-sm text-center" style={{ color: ON_BG2 }}>Tap to start recording</p>
          )}
        </div>
      )}

      {/* Playback + submit */}
      {stage === "recorded" && audioUrl && (
        <div className="flex flex-col gap-3">
          <div className="rounded-2xl p-4" style={{ backgroundColor: CARD, boxShadow: SHADOW }}>
            <p className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: TEXT3 }}>
              Preview your recording
            </p>
            <AudioPlayer src={audioUrl} />
          </div>

          <button
            onClick={() => { setAudioUrl(null); setStage("idle"); setElapsed(0); setSubmitError(""); }}
            className="w-full rounded-2xl py-3 text-sm font-medium cursor-pointer"
            style={{ backgroundColor: "rgba(255,255,255,0.15)", color: ON_BG,
              border: "1px solid rgba(255,255,255,0.3)" }}>
            Re-record
          </button>

          {/* Error from previous submit attempt */}
          {submitError && (
            <div className="flex items-start gap-3 rounded-2xl px-4 py-3"
              style={{ backgroundColor: "rgba(255,59,48,0.15)", border: "1px solid rgba(255,59,48,0.3)" }}>
              <AlertCircle size={16} style={{ color: "#FF3B30", flexShrink: 0, marginTop: 1 }} />
              <p className="text-sm" style={{ color: "#FF3B30" }}>{submitError}</p>
            </div>
          )}

          <button
            onClick={submitClone}
            disabled={!voiceName.trim()}
            className="w-full flex items-center justify-center gap-2 rounded-2xl py-4 font-semibold"
            style={{
              backgroundColor: voiceName.trim() ? GREEN : "rgba(255,255,255,0.2)",
              color: ON_BG,
              fontSize: 15,
              cursor: voiceName.trim() ? "pointer" : "not-allowed",
              transition: "background-color 0.2s",
              opacity: voiceName.trim() ? 1 : 0.6,
            }}>
            <Check size={18} />
            Clone &amp; Save Voice
          </button>
          {!voiceName.trim() && (
            <p className="text-xs text-center" style={{ color: ON_BG3 }}>
              Enter a voice name above to save
            </p>
          )}
        </div>
      )}

      {/* Loading */}
      {stage === "loading" && (
        <div className="flex flex-col items-center gap-3 py-4">
          <Loader2 size={32} style={{ color: ON_BG, animation: "spin 1s linear infinite" }} />
          <p className="text-sm" style={{ color: ON_BG2 }}>Cloning your voice with ElevenLabs…</p>
          <p className="text-xs text-center" style={{ color: ON_BG3 }}>This can take 30–60 seconds</p>
        </div>
      )}
    </div>
  );
}

// ── Design tab ─────────────────────────────────────────────────────────────

function DesignTab() {
  const [stage,       setStage]       = useState<Stage>("idle");
  const [description, setDescription] = useState("");
  const [voiceName,   setVoiceName]   = useState("");
  const [previewId,   setPreviewId]   = useState("");
  const [previewUrl,  setPreviewUrl]  = useState<string | null>(null);
  const [error,       setError]       = useState("");
  const [savedName,   setSavedName]   = useState("");

  const SAMPLE_TEXT = "Hey! My name is Echo. I'm here to help bridge the gap between ASL and spoken language. Let me know how I can help.";

  const generatePreview = useCallback(async () => {
    if (!description.trim() || !API_KEY) return;
    setStage("loading");
    setError("");
    try {
      const res = await fetch("https://api.elevenlabs.io/v1/text-to-voice/create-previews", {
        method: "POST",
        headers: { "xi-api-key": API_KEY, "Content-Type": "application/json" },
        body: JSON.stringify({ voice_description: description.trim(), text: SAMPLE_TEXT }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({})) as Record<string, unknown>;
        const detail = (body.detail as Record<string, unknown>)?.message
          ?? (body.detail as string) ?? `Error ${res.status}`;
        throw new Error(String(detail));
      }

      const data = await res.json() as { previews: { generated_voice_id: string; audio_base_64: string }[] };
      const first = data.previews?.[0];
      if (!first) throw new Error("No previews returned");

      const audioBytes = Uint8Array.from(atob(first.audio_base_64), (c) => c.charCodeAt(0));
      const blob = new Blob([audioBytes], { type: "audio/mpeg" });
      setPreviewUrl(URL.createObjectURL(blob));
      setPreviewId(first.generated_voice_id);
      setStage("preview");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
      setStage("idle");
    }
  }, [description]);

  const saveDesignedVoice = useCallback(async () => {
    if (!previewId || !voiceName.trim() || !API_KEY) return;
    setStage("loading");
    setError("");
    try {
      const res = await fetch("https://api.elevenlabs.io/v1/text-to-voice/create-voice-from-preview", {
        method: "POST",
        headers: { "xi-api-key": API_KEY, "Content-Type": "application/json" },
        body: JSON.stringify({
          voice_name:         voiceName.trim(),
          voice_description:  description.trim(),
          generated_voice_id: previewId,
        }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({})) as Record<string, unknown>;
        const detail = (body.detail as Record<string, unknown>)?.message
          ?? (body.detail as string) ?? `Error ${res.status}`;
        throw new Error(String(detail));
      }

      const { voice_id } = await res.json() as { voice_id: string };
      saveVoiceToProfile(voice_id, voiceName.trim());
      setSavedName(voiceName.trim());
      setStage("saved");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
      setStage("preview");  // stay on preview so they can retry
    }
  }, [previewId, voiceName, description]);

  if (stage === "saved") {
    return (
      <div className="flex flex-col items-center gap-6 pt-4 text-center">
        <div className="w-20 h-20 rounded-2xl flex items-center justify-center"
          style={{ backgroundColor: "rgba(52,199,89,0.2)", border: "1px solid rgba(52,199,89,0.4)" }}>
          <Check size={40} style={{ color: GREEN }} />
        </div>
        <div>
          <p className="text-2xl font-bold mb-1" style={{ color: ON_BG }}>{savedName}</p>
          <p className="text-sm" style={{ color: ON_BG2 }}>Voice saved and added to your voice library.</p>
          <p className="text-xs mt-1" style={{ color: ON_BG3 }}>
            Find it in Settings → Voice → Your Voices
          </p>
        </div>
        <button
          onClick={() => { setStage("idle"); setDescription(""); setVoiceName(""); setPreviewUrl(null); setPreviewId(""); setError(""); }}
          className="w-full rounded-2xl py-3 text-sm font-semibold cursor-pointer"
          style={{ backgroundColor: "rgba(255,255,255,0.2)", color: ON_BG, border: "1px solid rgba(255,255,255,0.35)" }}>
          Design another voice
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {/* How it works */}
      <div className="rounded-2xl p-4" style={{ backgroundColor: CARD, boxShadow: SHADOW }}>
        <p className="text-sm font-semibold mb-1" style={{ color: TEXT }}>How it works</p>
        <p className="text-sm leading-relaxed" style={{ color: TEXT2 }}>
          Describe the voice you want and ElevenLabs will generate a preview. Be specific —
          age, tone, accent, energy level all help.
        </p>
        <p className="text-xs mt-2 italic" style={{ color: TEXT3 }}>
          e.g. &quot;Young American woman, warm and conversational, college-age energy&quot;
        </p>
      </div>

      {/* Description input */}
      <textarea
        rows={3}
        placeholder="Describe the voice you want…"
        value={description}
        onChange={(e) => setDescription(e.target.value)}
        className="w-full rounded-2xl px-4 py-3 text-sm outline-none resize-none"
        style={{ backgroundColor: CARD, boxShadow: SHADOW, color: TEXT,
          border: `2px solid ${description.trim() ? PURPLE : "transparent"}` }}
      />

      {/* Generate button */}
      {stage !== "preview" && (
        <button
          onClick={generatePreview}
          disabled={!description.trim() || stage === "loading"}
          className="w-full flex items-center justify-center gap-2 rounded-2xl py-4 font-semibold"
          style={{
            backgroundColor: description.trim() && stage !== "loading"
              ? "rgba(255,255,255,0.9)" : "rgba(255,255,255,0.2)",
            color: description.trim() && stage !== "loading" ? PURPLE : ON_BG,
            fontSize: 15,
            cursor: description.trim() && stage !== "loading" ? "pointer" : "not-allowed",
            transition: "all 0.2s",
          }}>
          {stage === "loading"
            ? <><Loader2 size={18} style={{ animation: "spin 1s linear infinite" }} /> Generating preview…</>
            : <><Sparkles size={18} /> Generate Preview</>}
        </button>
      )}

      {/* Error */}
      {error && (
        <div className="flex items-start gap-3 rounded-2xl px-4 py-3"
          style={{ backgroundColor: "rgba(255,59,48,0.15)", border: "1px solid rgba(255,59,48,0.3)" }}>
          <AlertCircle size={16} style={{ color: "#FF3B30", flexShrink: 0, marginTop: 1 }} />
          <p className="text-sm" style={{ color: "#FF3B30" }}>{error}</p>
        </div>
      )}

      {/* Preview */}
      {stage === "preview" && previewUrl && (
        <div className="flex flex-col gap-3">
          <div className="rounded-2xl p-4" style={{ backgroundColor: CARD, boxShadow: SHADOW }}>
            <p className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: TEXT3 }}>
              Voice preview
            </p>
            <AudioPlayer src={previewUrl} />
          </div>

          {/* Name input */}
          <div>
            <input
              type="text"
              placeholder="Name this voice (e.g. My AI Voice)"
              value={voiceName}
              onChange={(e) => setVoiceName(e.target.value)}
              className="w-full rounded-2xl px-4 py-3 text-sm outline-none"
              style={{ backgroundColor: CARD, boxShadow: SHADOW, color: TEXT,
                border: `2px solid ${voiceName.trim() ? PURPLE : "transparent"}` }}
            />
            {!voiceName.trim() && (
              <p className="text-xs mt-1.5 ml-1" style={{ color: ON_BG3 }}>
                ↑ Enter a name to save this voice
              </p>
            )}
          </div>

          <div className="flex gap-2">
            <button
              onClick={() => { setStage("idle"); setPreviewUrl(null); setPreviewId(""); setError(""); }}
              className="flex-1 rounded-2xl py-3 text-sm font-medium cursor-pointer"
              style={{ backgroundColor: "rgba(255,255,255,0.15)", color: ON_BG,
                border: "1px solid rgba(255,255,255,0.3)" }}>
              Try again
            </button>
            <button
              onClick={saveDesignedVoice}
              disabled={!voiceName.trim()}
              className="flex-1 flex items-center justify-center gap-1.5 rounded-2xl py-3 text-sm font-semibold"
              style={{
                backgroundColor: voiceName.trim() ? GREEN : "rgba(255,255,255,0.2)",
                color: ON_BG,
                cursor: voiceName.trim() ? "pointer" : "not-allowed",
                opacity: voiceName.trim() ? 1 : 0.6,
                transition: "background-color 0.2s",
              }}>
              <Check size={16} />
              Use this voice
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Page ───────────────────────────────────────────────────────────────────

export default function VoiceSettingsPage() {
  const router = useRouter();
  const [tab, setTab] = useState<Tab>("clone");

  const [currentVoice, setCurrentVoice] = useState("Lauren");
  useEffect(() => {
    try {
      const prefs = JSON.parse(localStorage.getItem("maia_prefs") ?? "{}") as Record<string, unknown>;
      if (prefs.selectedVoice) setCurrentVoice(prefs.selectedVoice as string);
    } catch { /* ignore */ }
  }, []);

  return (
    <main className="min-h-screen pb-24 px-4" style={{ background: BG }}>
      <div className="max-w-sm mx-auto pt-12">

        {/* Header */}
        <div className="flex items-center gap-3 mb-6">
          <button
            onClick={() => router.back()}
            className="w-9 h-9 rounded-xl flex items-center justify-center cursor-pointer flex-shrink-0"
            style={{ backgroundColor: "rgba(255,255,255,0.2)", border: "1px solid rgba(255,255,255,0.35)" }}>
            <ArrowLeft size={18} style={{ color: ON_BG }} />
          </button>
          <div>
            <h1 className="text-xl font-bold" style={{ color: ON_BG }}>Personalize Voice</h1>
            <p className="text-xs mt-0.5" style={{ color: ON_BG3 }}>
              Active: <span style={{ color: ON_BG2 }}>{currentVoice}</span>
            </p>
          </div>
        </div>

        {/* Tab switcher */}
        <div className="flex gap-2 mb-6 p-1 rounded-2xl" style={{ backgroundColor: "rgba(0,0,0,0.15)" }}>
          {(["clone", "design"] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className="flex-1 flex items-center justify-center gap-1.5 py-2.5 rounded-xl text-sm font-semibold cursor-pointer transition-all"
              style={{
                backgroundColor: tab === t ? "rgba(255,255,255,0.9)" : "transparent",
                color:           tab === t ? PURPLE : ON_BG2,
                boxShadow:       tab === t ? SHADOW : "none",
              }}>
              {t === "clone" ? <><Mic size={14} /> Clone my voice</> : <><Sparkles size={14} /> Design a voice</>}
            </button>
          ))}
        </div>

        {/* Tab content */}
        {tab === "clone" ? <CloneTab /> : <DesignTab />}

      </div>
    </main>
  );
}
