"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { ArrowLeft, ArrowRight, Check, Mic, RefreshCw } from "lucide-react";
import { useMyoWs } from "@/hooks/use-myo-ws";

const BG     = "linear-gradient(180deg, #9147C8 0%, #A066D8 30%, #C49AEE 65%, #DDD0F8 85%, #EDE8FF 100%)";
const CARD   = "rgba(255,255,255,0.82)";
const PURPLE = "#7C6FE0";
const TEXT   = "#1C1C1E";
const TEXT2  = "#6C6C70";
const TEXT3  = "#8E8E93";
const GREEN  = "#34C759";
const SHADOW = "0 2px 12px rgba(80,0,150,0.1)";

// Text colors for elements sitting directly on the gradient
const ON_BG       = "#fff";
const ON_BG_2     = "rgba(255,255,255,0.85)";
const ON_BG_3     = "rgba(255,255,255,0.6)";

const TOTAL_REPS = 3;
const REP_INSTRUCTIONS = [
  "Hold your gesture and tap Record",
  "Great! Do it again",
  "One last time!",
];

type Step = "name" | "record" | "confirm";

export default function TeachPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { status, teach, connect, teachRecord, teachTrain } = useMyoWs();
  const [step, setStep]       = useState<Step>("name");
  const [word, setWord]       = useState("");
  const [reps, setReps]       = useState(0);
  const [trainBusy, setTrainBusy] = useState(false);
  const [isCalibrate, setIsCalibrate] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const w = searchParams.get("word");
    const cal = searchParams.get("calibrate") === "true";
    if (w) {
      setWord(w);
      setIsCalibrate(cal);
      setStep("record");
    }
  }, [searchParams]);

  const isConnected = status === "connected";

  const prevTeachCountRef = useRef(0);

  useEffect(() => {
    if (
      teach.word === word.trim().toLowerCase() &&
      teach.count > prevTeachCountRef.current
    ) {
      prevTeachCountRef.current = teach.count;
      setReps((r) => r + 1);
    }
  }, [teach.count, teach.word, word]);

  useEffect(() => {
    if (teach.modelReady && teach.word === word.trim().toLowerCase()) {
      setStep("confirm");
      setTrainBusy(false);
    }
  }, [teach.modelReady, teach.word, word]);

  const handleRecord = useCallback(() => {
    if (!isConnected || teach.collecting) return;
    teachRecord(word.trim().toLowerCase());
  }, [isConnected, teach.collecting, teachRecord, word]);

  const handleSave = useCallback(() => {
    if (trainBusy) return;
    setTrainBusy(true);
    teachTrain(word.trim().toLowerCase());
  }, [trainBusy, teachTrain, word]);

  const handleTeachAnother = () => {
    setWord("");
    setReps(0);
    prevTeachCountRef.current = 0;
    setTrainBusy(false);
    setStep("name");
  };

  const emg = teach.emg;
  const emgMax = 128;

  return (
    <main className="min-h-screen pb-24 px-4" style={{ background: BG }}>
      <div className="max-w-sm mx-auto pt-12">

        {/* Header */}
        <div className="flex items-center gap-3 mb-8">
          <button
            onClick={() => router.back()}
            className="w-9 h-9 rounded-xl flex items-center justify-center cursor-pointer"
            style={{ backgroundColor: "rgba(255,255,255,0.2)", border: "1px solid rgba(255,255,255,0.35)" }}
          >
            <ArrowLeft size={18} style={{ color: ON_BG }} />
          </button>
          <div>
            <h1 className="text-xl font-bold" style={{ color: ON_BG }}>Teach Echo</h1>
            <p className="text-xs mt-0.5" style={{ color: ON_BG_3 }}>
              Step {step === "name" ? 1 : step === "record" ? 2 : 3} of 3
            </p>
          </div>
        </div>

        {/* Step dots */}
        <div className="flex items-center gap-2 mb-8">
          {(["name", "record", "confirm"] as Step[]).map((s) => (
            <div
              key={s}
              className="rounded-full transition-all duration-300"
              style={{
                width: step === s ? 24 : 8,
                height: 8,
                backgroundColor: step === s ? ON_BG : step > s ? GREEN : "rgba(255,255,255,0.3)",
              }}
            />
          ))}
        </div>

        {/* ── Step 1: Name ── */}
        {step === "name" && (
          <div className="flex flex-col gap-6">
            <div>
              <p className="text-2xl font-bold mb-1" style={{ color: ON_BG }}>
                What word is this gesture for?
              </p>
              <p className="text-sm" style={{ color: ON_BG_2 }}>
                It can be a slang term, a name, or any word you want to teach.
              </p>
            </div>

            <input
              ref={inputRef}
              autoFocus
              type="text"
              value={word}
              onChange={(e) => setWord(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && word.trim()) setStep("record");
              }}
              placeholder="e.g. lowkey, rizz, a friend's name…"
              className="w-full rounded-2xl px-5 py-4 text-lg outline-none"
              style={{
                backgroundColor: CARD,
                boxShadow: SHADOW,
                color: TEXT,
                border: `2px solid ${word.trim() ? PURPLE : "transparent"}`,
              }}
            />

            {/* Connect prompt if not connected */}
            {!isConnected && (
              <div
                className="rounded-2xl p-4 flex flex-col gap-3"
                style={{ backgroundColor: "rgba(255,255,255,0.15)", border: "1px solid rgba(255,255,255,0.3)" }}
              >
                <p className="text-sm font-medium" style={{ color: ON_BG }}>
                  Connect to your Myo to record gestures
                </p>
                <button
                  onClick={() => connect("ws://localhost:8765")}
                  className="px-4 py-2 rounded-xl text-sm font-semibold cursor-pointer"
                  style={{ backgroundColor: "rgba(255,255,255,0.25)", color: ON_BG, border: "1px solid rgba(255,255,255,0.4)" }}
                >
                  {status === "connecting" ? "Connecting…" : "Connect"}
                </button>
              </div>
            )}

            <button
              onClick={() => setStep("record")}
              disabled={!word.trim()}
              className="w-full flex items-center justify-center gap-2 rounded-2xl py-4 font-semibold cursor-pointer"
              style={{
                backgroundColor: word.trim() ? "rgba(255,255,255,0.9)" : "rgba(255,255,255,0.2)",
                color: word.trim() ? PURPLE : ON_BG_3,
                border: word.trim() ? "none" : "1px solid rgba(255,255,255,0.35)",
                fontSize: 16,
                transition: "background-color 0.2s, color 0.2s",
              }}
            >
              Next
              <ArrowRight size={18} />
            </button>
          </div>
        )}

        {/* ── Step 2: Record ── */}
        {step === "record" && (
          <div className="flex flex-col items-center gap-6">
            {/* Word reminder */}
            <div className="text-center">
              <p className="text-xs font-semibold uppercase tracking-wider mb-1" style={{ color: ON_BG_3 }}>
                Recording gesture for
              </p>
              <p className="text-3xl font-bold" style={{ color: ON_BG }}>
                {word.trim()}
              </p>
            </div>

            {/* Instruction */}
            <p className="text-sm text-center" style={{ color: ON_BG_2 }}>
              {reps < TOTAL_REPS
                ? REP_INSTRUCTIONS[reps]
                : `Echo learned "${word.trim()}"!`}
            </p>

            {/* Big record button */}
            <div className="relative flex items-center justify-center">
              {teach.collecting && (
                <div
                  className="absolute rounded-full"
                  style={{
                    width: 140, height: 140,
                    backgroundColor: "rgba(255,255,255,0.15)",
                    animation: "pulse-ring 1s ease-out infinite",
                  }}
                />
              )}

              <button
                onClick={reps < TOTAL_REPS ? handleRecord : undefined}
                disabled={!isConnected || teach.collecting || reps >= TOTAL_REPS}
                className="w-28 h-28 rounded-full flex items-center justify-center cursor-pointer select-none"
                style={{
                  backgroundColor:
                    reps >= TOTAL_REPS
                      ? GREEN
                      : teach.collecting
                        ? PURPLE
                        : "rgba(255,255,255,0.2)",
                  border: `3px solid ${reps >= TOTAL_REPS ? GREEN : ON_BG}`,
                  transition: "background-color 0.3s",
                  boxShadow: teach.collecting
                    ? "0 0 0 6px rgba(255,255,255,0.15)"
                    : SHADOW,
                }}
              >
                {reps >= TOTAL_REPS ? (
                  <Check size={40} style={{ color: "#fff" }} />
                ) : (
                  <Mic size={32} style={{ color: ON_BG }} />
                )}
              </button>
            </div>

            {/* Rep dots */}
            <div className="flex gap-3">
              {Array.from({ length: TOTAL_REPS }).map((_, i) => (
                <div
                  key={i}
                  className="rounded-full transition-all duration-500"
                  style={{
                    width: i < reps ? 28 : 12,
                    height: 12,
                    backgroundColor:
                      i < reps
                        ? i === reps - 1 ? GREEN : ON_BG
                        : "rgba(255,255,255,0.3)",
                  }}
                />
              ))}
            </div>

            {/* EMG live visualiser */}
            <div
              className="w-full rounded-2xl p-4"
              style={{ backgroundColor: CARD, boxShadow: SHADOW }}
            >
              <p className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: TEXT3 }}>
                EMG Signal {teach.collecting ? "· Recording" : ""}
              </p>
              <div className="flex items-end gap-1" style={{ height: 48 }}>
                {emg.map((val, i) => {
                  const pct = Math.min(Math.abs(val) / emgMax, 1);
                  return (
                    <div
                      key={i}
                      className="flex-1 rounded-sm"
                      style={{
                        height: `${Math.max(pct * 100, 4)}%`,
                        backgroundColor: teach.collecting
                          ? i < 4 ? PURPLE : "rgba(124,111,224,0.5)"
                          : "rgba(0,0,0,0.1)",
                        transition: "height 0.05s linear, background-color 0.3s",
                      }}
                    />
                  );
                })}
              </div>
            </div>

            {/* Save button — appears after all reps */}
            {reps >= TOTAL_REPS && (
              <button
                onClick={handleSave}
                disabled={trainBusy || !isConnected}
                className="w-full flex items-center justify-center gap-2 rounded-2xl py-4 font-semibold cursor-pointer"
                style={{
                  backgroundColor: trainBusy ? "rgba(255,255,255,0.2)" : GREEN,
                  color: ON_BG,
                  fontSize: 16,
                  transition: "background-color 0.2s",
                }}
              >
                {trainBusy ? (
                  <>
                    <RefreshCw size={18} className="animate-spin" />
                    Training model…
                  </>
                ) : (
                  <>
                    <Check size={18} />
                    Save "{word.trim()}"
                  </>
                )}
              </button>
            )}
          </div>
        )}

        {/* ── Step 3: Confirm ── */}
        {step === "confirm" && (
          <div className="flex flex-col items-center gap-8 pt-8 text-center">
            <div
              className="w-20 h-20 rounded-2xl flex items-center justify-center"
              style={{ backgroundColor: "rgba(52,199,89,0.2)", border: "1px solid rgba(52,199,89,0.4)" }}
            >
              <Check size={40} style={{ color: GREEN }} />
            </div>

            <div>
              <p className="text-5xl font-bold mb-3" style={{ color: ON_BG }}>
                {word.trim()}
              </p>
              <p className="text-base" style={{ color: ON_BG_2 }}>
                {isCalibrate
                  ? "Model updated with your new reps."
                  : "Echo will recognise this from now on."}
              </p>
              {teach.cvAccuracy != null && (
                <p className="text-sm mt-1" style={{ color: ON_BG_3 }}>
                  Model accuracy: {(teach.cvAccuracy * 100).toFixed(0)}%
                </p>
              )}
            </div>

            <div className="flex flex-col gap-3 w-full">
              <button
                onClick={() => router.push("/translate")}
                className="w-full flex items-center justify-center gap-2 rounded-2xl py-4 font-semibold cursor-pointer"
                style={{ backgroundColor: "rgba(255,255,255,0.9)", color: PURPLE, fontSize: 16 }}
              >
                Try it now
              </button>
              <button
                onClick={handleTeachAnother}
                className="w-full flex items-center justify-center gap-2 rounded-2xl py-4 font-semibold cursor-pointer"
                style={{
                  backgroundColor: "rgba(255,255,255,0.15)",
                  color: ON_BG,
                  fontSize: 16,
                  border: "1px solid rgba(255,255,255,0.35)",
                }}
              >
                <RefreshCw size={16} />
                Teach another
              </button>
              <button
                onClick={() => router.back()}
                className="text-sm cursor-pointer py-2"
                style={{ color: ON_BG_2 }}
              >
                Done
              </button>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
