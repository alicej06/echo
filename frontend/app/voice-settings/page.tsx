"use client";
import { useState, useRef, useCallback, useEffect } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft, Mic, Square, Play, Check, Loader2, Sparkles, AlertCircle } from "lucide-react";

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
const MIN_RECORD_S = 30;  // ElevenLabs minimum for cloning
const MAX_RECORD_S = 120; // auto-stop safety net (2 minutes)

type Tab   = "clone" | "design";
type Stage = "idle" | "recording" | "recorded" | "loading" | "preview" | "saved" | "error";

// ── Helpers ────────────────────────────────────────────────────────────────

function saveVoiceToProfile(voiceId: string, voiceName: string) {
  try {
    // 1. Set as active voice in prefs
    const raw   = localStorage.getItem("maia_prefs");
    const prefs = raw ? (JSON.parse(raw) as Record<string, unknown>) : {};
    prefs.selectedVoiceId = voiceId;
    prefs.selectedVoice   = voiceName;
    localStorage.setItem("maia_prefs", JSON.stringify(prefs));

    // 2. Append to persistent custom voice library (deduplicated by voiceId)
    const libRaw  = localStorage.getItem("maia_custom_voices");
    const library = libRaw ? (JSON.parse(libRaw) as { name: string; voiceId: string }[]) : [];
    if (!library.some((v) => v.voiceId === voiceId)) {
      library.push({ name: voiceName, voiceId });
      localStorage.setItem("maia_custom_voices", JSON.stringify(library));
    }

    console.log(`[voice-settings] saved voice → ${voiceName} (${voiceId})`);
  } catch { /* ignore */ }
}

function fmt(s: number) {
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

// ── Clone tab ──────────────────────────────────────────────────────────────

function CloneTab() {
  const [stage,     setStage]     = useState<Stage>("idle");
  const [elapsed,   setElapsed]   = useState(0);
  const [audioUrl,  setAudioUrl]  = useState<string | null>(null);
  const [voiceName, setVoiceName] = useState("");
  const [error,     setError]     = useState("");
  const [savedName, setSavedName] = useState("");

  const recorderRef  = useRef<MediaRecorder | null>(null);
  const streamRef    = useRef<MediaStream | null>(null);
  const chunksRef    = useRef<Blob[]>([]);
  const blobRef      = useRef<Blob | null>(null);
  const timerRef     = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopTimer = () => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
  };

  // Stable stop function stored in a ref so the auto-stop timer can call it
  const doStop = useCallback(() => {
    stopTimer();
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();   // triggers onstop → sets stage to "recorded"
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const doStopRef = useRef(doStop);
  doStopRef.current = doStop;

  const startRecording = useCallback(async () => {
    setError("");
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
        // Auto-stop at MAX_RECORD_S
        if (secs >= MAX_RECORD_S) doStopRef.current();
      }, 1000);
    } catch {
      setError("Microphone access denied. Allow mic in browser settings.");
      setStage("error");
    }
  }, []);

  const stopRecording = useCallback(() => {
    doStopRef.current();
  }, []);

  const submitClone = useCallback(async () => {
    if (!blobRef.current || !voiceName.trim()) return;
    if (!API_KEY) { setError("NEXT_PUBLIC_ELEVENLABS_API_KEY not set"); return; }

    setStage("loading");
    setError("");
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
          ?? (body.detail as string)
          ?? `Error ${res.status}`;
        throw new Error(String(detail));
      }

      const { voice_id } = await res.json() as { voice_id: string };
      saveVoiceToProfile(voice_id, voiceName.trim());
      setSavedName(voiceName.trim());
      setStage("saved");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
      setStage("error");
    }
  }, [voiceName]);

  useEffect(() => () => { stopTimer(); streamRef.current?.getTracks().forEach((t) => t.stop()); }, []);

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
          <p className="text-sm" style={{ color: ON_BG2 }}>Voice cloned and set as your Echo voice.</p>
        </div>
        <button onClick={() => { setStage("idle"); setVoiceName(""); setAudioUrl(null); setElapsed(0); }}
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

      {/* Voice name input */}
      <input
        type="text"
        placeholder="Name this voice (e.g. My Voice)"
        value={voiceName}
        onChange={(e) => setVoiceName(e.target.value)}
        className="w-full rounded-2xl px-4 py-3 text-sm outline-none"
        style={{ backgroundColor: CARD, boxShadow: SHADOW, color: TEXT,
          border: `2px solid ${voiceName.trim() ? PURPLE : "transparent"}` }}
      />

      {/* Record button */}
      {stage !== "recorded" && (
        <div className="flex flex-col items-center gap-4">
          <div className="relative flex items-center justify-center">
            {stage === "recording" && (
              <div className="absolute rounded-full"
                style={{ width: 120, height: 120, backgroundColor: "rgba(255,59,48,0.12)",
                  animation: "pulse-ring 1s ease-out infinite" }} />
            )}
            <button
              onClick={stage === "recording" ? stopRecording : startRecording}
              disabled={stage === "loading"}
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
            <div className="flex flex-col items-center gap-1">
              <p className="text-2xl font-bold tabular-nums" style={{ color: ON_BG }}>{fmt(elapsed)}</p>
              <p className="text-xs" style={{ color: hasEnough ? GREEN : ON_BG3 }}>
                {hasEnough ? "✓ Enough to clone — keep going for better quality" : `${MIN_RECORD_S - elapsed}s more needed`}
              </p>
            </div>
          )}

          {stage === "recording" && (
            <button
              onClick={stopRecording}
              className="w-full rounded-2xl py-3 text-sm font-semibold cursor-pointer"
              style={{
                backgroundColor: "rgba(255,59,48,0.18)",
                color: "#FF3B30",
                border: "1px solid rgba(255,59,48,0.35)",
              }}>
              Stop Recording
            </button>
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
              Preview recording
            </p>
            <audio src={audioUrl} controls className="w-full" style={{ borderRadius: 8 }} />
          </div>

          <button onClick={() => { setAudioUrl(null); setStage("idle"); setElapsed(0); }}
            className="w-full rounded-2xl py-3 text-sm font-medium cursor-pointer"
            style={{ backgroundColor: "rgba(255,255,255,0.15)", color: ON_BG, border: "1px solid rgba(255,255,255,0.3)" }}>
            Re-record
          </button>

          <button
            onClick={submitClone}
            disabled={!voiceName.trim() || stage === "loading"}
            className="w-full flex items-center justify-center gap-2 rounded-2xl py-4 font-semibold cursor-pointer"
            style={{
              backgroundColor: voiceName.trim() ? GREEN : "rgba(255,255,255,0.2)",
              color: ON_BG,
              fontSize: 15,
              transition: "background-color 0.2s",
            }}>
            <Check size={18} />
            Clone &amp; Save Voice
          </button>
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

      {/* Error */}
      {(stage === "error") && error && (
        <div className="flex items-start gap-3 rounded-2xl px-4 py-3"
          style={{ backgroundColor: "rgba(255,59,48,0.15)", border: "1px solid rgba(255,59,48,0.3)" }}>
          <AlertCircle size={16} style={{ color: "#FF3B30", flexShrink: 0, marginTop: 1 }} />
          <p className="text-sm" style={{ color: "#FF3B30" }}>{error}</p>
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
      setStage("error");
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
          voice_name:        voiceName.trim(),
          voice_description: description.trim(),
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
      setStage("error");
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
          <p className="text-sm" style={{ color: ON_BG2 }}>Voice saved and set as your Echo voice.</p>
        </div>
        <button onClick={() => { setStage("idle"); setDescription(""); setVoiceName(""); setPreviewUrl(null); setPreviewId(""); }}
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
          e.g. "Young American woman, warm and conversational, college-age energy"
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
          className="w-full flex items-center justify-center gap-2 rounded-2xl py-4 font-semibold cursor-pointer"
          style={{
            backgroundColor: description.trim() && stage !== "loading"
              ? "rgba(255,255,255,0.9)" : "rgba(255,255,255,0.2)",
            color: description.trim() && stage !== "loading" ? PURPLE : ON_BG,
            fontSize: 15,
            transition: "all 0.2s",
          }}>
          {stage === "loading"
            ? <><Loader2 size={18} style={{ animation: "spin 1s linear infinite" }} /> Generating preview…</>
            : <><Sparkles size={18} /> Generate Preview</>}
        </button>
      )}

      {/* Preview */}
      {stage === "preview" && previewUrl && (
        <div className="flex flex-col gap-3">
          <div className="rounded-2xl p-4" style={{ backgroundColor: CARD, boxShadow: SHADOW }}>
            <p className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: TEXT3 }}>
              Voice preview
            </p>
            <audio src={previewUrl} controls className="w-full" style={{ borderRadius: 8 }} />
          </div>

          {/* Name input */}
          <input
            type="text"
            placeholder="Name this voice (e.g. My AI Voice)"
            value={voiceName}
            onChange={(e) => setVoiceName(e.target.value)}
            className="w-full rounded-2xl px-4 py-3 text-sm outline-none"
            style={{ backgroundColor: CARD, boxShadow: SHADOW, color: TEXT,
              border: `2px solid ${voiceName.trim() ? PURPLE : "transparent"}` }}
          />

          <div className="flex gap-2">
            <button onClick={() => { setStage("idle"); setPreviewUrl(null); setPreviewId(""); }}
              className="flex-1 rounded-2xl py-3 text-sm font-medium cursor-pointer"
              style={{ backgroundColor: "rgba(255,255,255,0.15)", color: ON_BG, border: "1px solid rgba(255,255,255,0.3)" }}>
              Try again
            </button>
            <button
              onClick={saveDesignedVoice}
              disabled={!voiceName.trim()}
              className="flex-1 flex items-center justify-center gap-1.5 rounded-2xl py-3 text-sm font-semibold cursor-pointer"
              style={{
                backgroundColor: voiceName.trim() ? GREEN : "rgba(255,255,255,0.2)",
                color: ON_BG, transition: "background-color 0.2s",
              }}>
              <Check size={16} />
              Use this voice
            </button>
          </div>
        </div>
      )}

      {/* Error */}
      {stage === "error" && error && (
        <div className="flex items-start gap-3 rounded-2xl px-4 py-3"
          style={{ backgroundColor: "rgba(255,59,48,0.15)", border: "1px solid rgba(255,59,48,0.3)" }}>
          <AlertCircle size={16} style={{ color: "#FF3B30", flexShrink: 0, marginTop: 1 }} />
          <div>
            <p className="text-sm" style={{ color: "#FF3B30" }}>{error}</p>
            <button onClick={() => setStage("idle")} className="text-xs mt-1 underline cursor-pointer"
              style={{ color: "#FF3B30" }}>Try again</button>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Page ───────────────────────────────────────────────────────────────────

export default function VoiceSettingsPage() {
  const router   = useRouter();
  const [tab, setTab] = useState<Tab>("clone");

  // Show currently active voice
  const [currentVoice, setCurrentVoice] = useState("Rachel (default)");
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
