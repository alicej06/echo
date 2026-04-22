"use client";
import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { Watch, Volume2, Check, Loader2, Mic } from "lucide-react";
import { useMyoWs } from "@/hooks/use-myo-ws";
import { useElevenLabs } from "@/hooks/use-elevenlabs";

const PURPLE = "#7C6FE0";
const GREEN  = "#34C759";

const VOICES = [
  { name: "Samantha", desc: "Natural, friendly" },
  { name: "Karen",    desc: "Clear, professional" },
  { name: "Alex",     desc: "Calm, neutral" },
];

const CAL_GESTURES = [
  { phrase: "hello",       hint: "Wave hand side to side" },
  { phrase: "thank you",   hint: "Flat hand from chin forward" },
  { phrase: "how are you", hint: "Bent fingers roll forward, then point" },
];

const REPS_NEEDED = 3;
const REP_PROMPTS = ["Hold your gesture and tap Record", "Great! One more time", "Last one"];

type OnboardStep = "wristband" | "calibration" | "voice" | "done";
const STEPS: OnboardStep[] = ["wristband", "calibration", "voice", "done"];

export default function OnboardingPage() {
  const router = useRouter();
  const myo    = useMyoWs();
  const { speak } = useElevenLabs();

  const [stepIdx,        setStepIdx]        = useState(0);
  const [selectedVoice,  setSelectedVoice]  = useState("Samantha");

  // Calibration state
  const [calGestureIdx,  setCalGestureIdx]  = useState(0);  // which gesture we're on
  const [calReps,        setCalReps]        = useState([0, 0, 0]); // reps per gesture
  const [listening,      setListening]      = useState(false);
  const prevCount = useRef(0);
  const listenTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Auto-advance wristband step on connect
  useEffect(() => {
    if (myo.status === "connected" && STEPS[stepIdx] === "wristband") {
      setTimeout(() => setStepIdx(1), 700);
    }
  }, [myo.status, stepIdx]);

  // Detect rep recorded for current gesture
  useEffect(() => {
    if (!listening) return;
    const phrase = CAL_GESTURES[calGestureIdx].phrase;
    const curr = myo.phraseTrainStatus[phrase] ?? 0;
    if (curr > prevCount.current) {
      prevCount.current = curr;
      setListening(false);
      if (listenTimeout.current) clearTimeout(listenTimeout.current);

      setCalReps((prev) => {
        const next = [...prev];
        next[calGestureIdx] = curr;
        return next;
      });

      // Auto-advance to next gesture after a short pause
      if (curr >= REPS_NEEDED && calGestureIdx < CAL_GESTURES.length - 1) {
        setTimeout(() => {
          setCalGestureIdx((i) => i + 1);
          prevCount.current = 0;
        }, 600);
      }
    }
  }, [myo.phraseTrainStatus, listening, calGestureIdx]);

  useEffect(() => {
    return () => { if (listenTimeout.current) clearTimeout(listenTimeout.current); };
  }, []);

  function handleRecord() {
    if (listening || myo.status !== "connected") return;
    const phrase = CAL_GESTURES[calGestureIdx].phrase;
    prevCount.current = myo.phraseTrainStatus[phrase] ?? 0;
    setListening(true);
    myo.trainPhrase(phrase);
    listenTimeout.current = setTimeout(() => setListening(false), 10000);
  }

  function finish() {
    try {
      const raw   = localStorage.getItem("maia_prefs");
      const prefs = raw ? JSON.parse(raw) : {};
      localStorage.setItem("maia_prefs", JSON.stringify({ ...prefs, selectedVoice }));
      localStorage.setItem("echo_onboarded", "1");
    } catch { /* ignore */ }
    router.push("/home");
  }

  const calAllDone = calReps.every((r) => r >= REPS_NEEDED);

  const canProceed =
    (STEPS[stepIdx] === "wristband"   && myo.status === "connected") ||
    (STEPS[stepIdx] === "calibration" && calAllDone) ||
     STEPS[stepIdx] === "voice" ||
     STEPS[stepIdx] === "done";

  function onContinue() {
    if (stepIdx < STEPS.length - 1) setStepIdx((s) => s + 1);
    else finish();
  }

  return (
    <main
      className="min-h-screen flex flex-col items-center justify-between px-6 py-12"
      style={{
        background:
          "linear-gradient(180deg, #9147C8 0%, #A066D8 30%, #C49AEE 65%, #DDD0F8 85%, #EDE8FF 100%)",
      }}
    >
      {/* Progress dots + skip */}
      <div className="w-full max-w-sm flex items-center justify-between">
        <div className="flex gap-2">
          {STEPS.map((_, i) => (
            <div
              key={i}
              style={{
                width: i === stepIdx ? 20 : 6,
                height: 6,
                borderRadius: 3,
                backgroundColor:
                  i <= stepIdx ? "rgba(255,255,255,0.9)" : "rgba(255,255,255,0.3)",
                transition: "all 0.3s",
              }}
            />
          ))}
        </div>
        {stepIdx < 3 && (
          <button
            onClick={finish}
            style={{
              fontSize: 13,
              color: "rgba(255,255,255,0.65)",
              background: "none",
              border: "none",
              cursor: "pointer",
              fontWeight: 500,
            }}
          >
            Skip
          </button>
        )}
      </div>

      {/* Card */}
      <div
        className="w-full max-w-sm rounded-3xl p-6 flex flex-col gap-6"
        style={{
          backgroundColor: "rgba(255,255,255,0.18)",
          backdropFilter: "blur(20px)",
          border: "1px solid rgba(255,255,255,0.35)",
        }}
      >
        {STEPS[stepIdx] === "wristband"   && <WristbandStep   status={myo.status} onConnect={() => myo.connect()} />}
        {STEPS[stepIdx] === "calibration" && (
          <CalibrationStep
            gestureIdx={calGestureIdx}
            reps={calReps}
            listening={listening}
            connected={myo.status === "connected"}
            gestureRms={myo.gestureRms}
            onRecord={handleRecord}
          />
        )}
        {STEPS[stepIdx] === "voice" && <VoiceStep selected={selectedVoice} onSelect={setSelectedVoice} speak={speak} />}
        {STEPS[stepIdx] === "done"  && <DoneStep />}
      </div>

      {/* CTA */}
      <div className="w-full max-w-sm" style={{ paddingBottom: "2rem" }}>
        <button
          onClick={onContinue}
          disabled={!canProceed}
          style={{
            width: "100%",
            padding: "15px",
            borderRadius: 100,
            background: canProceed ? "rgba(255,255,255,0.95)" : "rgba(255,255,255,0.25)",
            border: "none",
            color: canProceed ? PURPLE : "rgba(255,255,255,0.45)",
            fontSize: 13,
            fontWeight: 700,
            letterSpacing: "0.08em",
            cursor: canProceed ? "pointer" : "default",
            transition: "all 0.2s",
          }}
        >
          {STEPS[stepIdx] === "done" ? "GO TO ECHO" : "CONTINUE"}
        </button>
      </div>
    </main>
  );
}

// ── Step 1: Wristband ──────────────────────────────────────────────────────

function WristbandStep({ status, onConnect }: { status: string; onConnect: () => void }) {
  return (
    <>
      <div
        className="w-16 h-16 rounded-full flex items-center justify-center"
        style={{ backgroundColor: "rgba(255,255,255,0.25)" }}
      >
        <Watch size={30} color="#fff" />
      </div>
      <div>
        <h2 style={{ fontSize: 22, fontWeight: 700, color: "#fff", marginBottom: 6 }}>
          Put on your wristband
        </h2>
        <p style={{ fontSize: 14, color: "rgba(255,255,255,0.75)", lineHeight: 1.55 }}>
          Wear it snugly on your dominant forearm with the sensors facing inward against your skin.
        </p>
      </div>
      <button
        onClick={status === "disconnected" ? onConnect : undefined}
        style={{
          padding: "12px 20px",
          borderRadius: 100,
          border: "1.5px solid rgba(255,255,255,0.6)",
          background:
            status === "connected" ? "rgba(52,199,89,0.25)" : "rgba(255,255,255,0.15)",
          color: "#fff",
          fontSize: 13,
          fontWeight: 600,
          cursor: status === "disconnected" ? "pointer" : "default",
          display: "flex",
          alignItems: "center",
          gap: 8,
          width: "fit-content",
          transition: "all 0.3s",
        }}
      >
        {status === "disconnected" && "Connect wristband"}
        {status === "connecting"   && <><InlineSpinner /> Connecting...</>}
        {status === "connected"    && <><Check size={14} /> Connected</>}
      </button>
    </>
  );
}

// ── Step 2: Calibration ───────────────────────────────────────────────────

function CalibrationStep({
  gestureIdx,
  reps,
  listening,
  connected,
  gestureRms,
  onRecord,
}: {
  gestureIdx: number;
  reps: number[];
  listening: boolean;
  connected: boolean;
  gestureRms: number;
  onRecord: () => void;
}) {
  const gesture    = CAL_GESTURES[gestureIdx];
  const currentRep = reps[gestureIdx];
  const done       = currentRep >= REPS_NEEDED;

  // Animate EMG bars using gestureRms + per-channel offsets
  const [bars, setBars] = useState<number[]>(Array(8).fill(0.05));
  const offsets = useRef(
    Array.from({ length: 8 }, () => 0.5 + Math.random()),
  );
  useEffect(() => {
    const id = setInterval(() => {
      const base = Math.min(gestureRms / 40, 1);
      setBars(
        offsets.current.map((o) =>
          Math.max(0.04, Math.min(1, base * o + (Math.random() - 0.5) * 0.08)),
        ),
      );
    }, 60);
    return () => clearInterval(id);
  }, [gestureRms]);

  return (
    <>
      {/* Header */}
      <div>
        <p style={{ fontSize: 13, color: "rgba(255,255,255,0.6)", fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 4 }}>
          Gesture {gestureIdx + 1} of {CAL_GESTURES.length}
        </p>
        <h2 style={{ fontSize: 22, fontWeight: 700, color: "#fff", marginBottom: 4 }}>
          {gesture.phrase}
        </h2>
        <p style={{ fontSize: 14, color: "rgba(255,255,255,0.7)" }}>
          {gesture.hint}
        </p>
      </div>

      {/* Live EMG bars */}
      <div
        style={{
          borderRadius: 16,
          padding: "14px 16px",
          background: "rgba(0,0,0,0.15)",
          border: listening
            ? "1px solid rgba(255,255,255,0.4)"
            : "1px solid rgba(255,255,255,0.15)",
          transition: "border-color 0.3s",
        }}
      >
        <p style={{ fontSize: 11, color: "rgba(255,255,255,0.5)", fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 10 }}>
          EMG Signal {listening ? "· Recording" : "· Live"}
        </p>
        <div style={{ display: "flex", alignItems: "flex-end", gap: 4, height: 48 }}>
          {bars.map((h, i) => (
            <div
              key={i}
              style={{
                flex: 1,
                height: `${h * 100}%`,
                borderRadius: 3,
                backgroundColor: listening
                  ? i < 4 ? "rgba(255,255,255,0.9)" : "rgba(255,255,255,0.5)"
                  : "rgba(255,255,255,0.3)",
                transition: "height 0.06s linear, background-color 0.3s",
              }}
            />
          ))}
        </div>
      </div>

      {/* Rep dots */}
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        {Array.from({ length: REPS_NEEDED }).map((_, i) => (
          <div
            key={i}
            style={{
              width: i < currentRep ? 24 : 10,
              height: 10,
              borderRadius: 5,
              backgroundColor:
                i < currentRep
                  ? GREEN
                  : "rgba(255,255,255,0.25)",
              transition: "all 0.3s",
            }}
          />
        ))}
        <p style={{ fontSize: 13, color: "rgba(255,255,255,0.65)", marginLeft: 4 }}>
          {done
            ? "Done!"
            : currentRep === 0
              ? REP_PROMPTS[0]
              : REP_PROMPTS[currentRep]}
        </p>
      </div>

      {/* Record button */}
      <div style={{ position: "relative", display: "flex", alignItems: "center", justifyContent: "center" }}>
        {listening && (
          <div
            style={{
              position: "absolute",
              width: 100,
              height: 100,
              borderRadius: "50%",
              backgroundColor: "rgba(255,255,255,0.15)",
              animation: "pulse-ring 1s ease-out infinite",
            }}
          />
        )}
        <button
          onClick={!done ? onRecord : undefined}
          disabled={!connected || listening || done}
          style={{
            width: 80,
            height: 80,
            borderRadius: "50%",
            border: `3px solid ${done ? GREEN : "rgba(255,255,255,0.8)"}`,
            background: done
              ? "rgba(52,199,89,0.3)"
              : listening
                ? "rgba(255,255,255,0.35)"
                : "rgba(255,255,255,0.18)",
            cursor: connected && !listening && !done ? "pointer" : "default",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            transition: "all 0.2s",
          }}
        >
          {done
            ? <Check size={30} color={GREEN} />
            : listening
              ? <Loader2 size={26} color="#fff" className="animate-spin" />
              : <Mic size={26} color="#fff" />}
        </button>
      </div>

      {/* Gesture progress pills */}
      <div style={{ display: "flex", gap: 6 }}>
        {CAL_GESTURES.map((g, i) => (
          <div
            key={g.phrase}
            style={{
              flex: 1,
              padding: "5px 0",
              borderRadius: 8,
              background:
                reps[i] >= REPS_NEEDED
                  ? "rgba(52,199,89,0.3)"
                  : i === gestureIdx
                    ? "rgba(255,255,255,0.25)"
                    : "rgba(255,255,255,0.1)",
              border:
                i === gestureIdx
                  ? "1px solid rgba(255,255,255,0.5)"
                  : "1px solid transparent",
              textAlign: "center",
              fontSize: 11,
              color:
                reps[i] >= REPS_NEEDED
                  ? GREEN
                  : i === gestureIdx
                    ? "#fff"
                    : "rgba(255,255,255,0.4)",
              fontWeight: 600,
              transition: "all 0.3s",
            }}
          >
            {reps[i] >= REPS_NEEDED ? "✓ " : ""}{g.phrase}
          </div>
        ))}
      </div>
    </>
  );
}

// ── Step 3: Voice ─────────────────────────────────────────────────────────

function VoiceStep({ selected, onSelect, speak }: { selected: string; onSelect: (v: string) => void; speak: (t: string) => Promise<void> }) {
  return (
    <>
      <div
        className="w-16 h-16 rounded-full flex items-center justify-center"
        style={{ backgroundColor: "rgba(255,255,255,0.25)" }}
      >
        <Volume2 size={30} color="#fff" />
      </div>
      <div>
        <h2 style={{ fontSize: 22, fontWeight: 700, color: "#fff", marginBottom: 6 }}>
          Choose your voice
        </h2>
        <p style={{ fontSize: 14, color: "rgba(255,255,255,0.75)", lineHeight: 1.55 }}>
          Echo speaks your signs aloud. Tap to preview.
        </p>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {VOICES.map(({ name, desc }) => (
          <button
            key={name}
            onClick={() => {
              onSelect(name);
              speak(`Hi! I'm the Echo voice. Nice to meet you.`);
            }}
            style={{
              padding: "12px 16px",
              borderRadius: 14,
              border:
                selected === name
                  ? "2px solid rgba(255,255,255,0.9)"
                  : "1.5px solid rgba(255,255,255,0.3)",
              background:
                selected === name ? "rgba(255,255,255,0.25)" : "rgba(255,255,255,0.1)",
              color: "#fff",
              textAlign: "left",
              cursor: "pointer",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              transition: "all 0.2s",
            }}
          >
            <div>
              <p style={{ fontSize: 14, fontWeight: 600 }}>{name}</p>
              <p style={{ fontSize: 12, color: "rgba(255,255,255,0.65)", marginTop: 2 }}>{desc}</p>
            </div>
            {selected === name && <Check size={16} color="rgba(255,255,255,0.9)" />}
          </button>
        ))}
      </div>
    </>
  );
}

// ── Step 4: Done ──────────────────────────────────────────────────────────

function DoneStep() {
  return (
    <>
      <div
        className="w-16 h-16 rounded-full flex items-center justify-center"
        style={{ backgroundColor: "rgba(255,255,255,0.25)" }}
      >
        <Check size={30} color="#fff" strokeWidth={2.5} />
      </div>
      <div>
        <h2 style={{ fontSize: 22, fontWeight: 700, color: "#fff", marginBottom: 6 }}>
          You&apos;re all set
        </h2>
        <p style={{ fontSize: 14, color: "rgba(255,255,255,0.75)", lineHeight: 1.55 }}>
          Echo is ready. Start translating, have a conversation, or teach Echo something new.
        </p>
      </div>
    </>
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────

function InlineSpinner() {
  return (
    <div
      style={{
        width: 14,
        height: 14,
        border: "2px solid rgba(255,255,255,0.3)",
        borderTopColor: "#fff",
        borderRadius: "50%",
        animation: "spin 0.7s linear infinite",
      }}
    />
  );
}
