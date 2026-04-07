"use client";
import { useState, useCallback, useEffect, useRef } from "react";
import {
  Mic,
  CheckCircle2,
  Circle,
  Radio,
  Play,
  Loader2,
  AlertCircle,
  ChevronRight,
} from "lucide-react";
import { useMyoWs } from "@/hooks/use-myo-ws";

const ASL_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const REPS_NEEDED = 5;

// ASL fingerspelling descriptions (from the paper's Table 1)
const LETTER_HINTS: Record<string, string> = {
  A: "Fist with thumb to side",
  B: "Flat open hand, thumb in",
  C: "Curved fingers and thumb",
  D: "Index up, others curved to thumb",
  E: "Fingers bent/retracted to palm",
  F: "Thumb + index touch, others spread",
  G: "Index + thumb point sideways",
  H: "Index + middle extended sideways",
  I: "Pinky extended from fist",
  J: "Pinky extended, trace J",
  K: "Index + middle + thumb form K",
  L: "Thumb + index right angle",
  M: "Three fingers over thumb",
  N: "Two fingers over thumb",
  O: "Fingers + thumb form O",
  P: "K-shape pointing down",
  Q: "G-shape pointing down",
  R: "Index + middle crossed",
  S: "Fist with thumb over fingers",
  T: "Thumb between index + middle",
  U: "Index + middle together up",
  V: "Index + middle spread (peace)",
  W: "First three fingers spread",
  X: "Index hooked",
  Y: "Thumb + pinky spread",
  Z: "Index traces Z",
};

function LetterCard({
  letter,
  count,
  isActive,
  isRecording,
  onRecord,
  disabled,
}: {
  letter: string;
  count: number;
  isActive: boolean;
  isRecording: boolean;
  onRecord: () => void;
  disabled: boolean;
}) {
  const done = count >= REPS_NEEDED;
  const pct = Math.min(count / REPS_NEEDED, 1);

  return (
    <div
      className="rounded-2xl p-4 flex flex-col gap-3 transition-all duration-200"
      style={{
        background: isActive
          ? "rgba(6,182,212,0.08)"
          : done
            ? "rgba(34,197,94,0.06)"
            : "rgba(255,255,255,0.04)",
        border: isActive
          ? "1px solid rgba(6,182,212,0.4)"
          : done
            ? "1px solid rgba(34,197,94,0.2)"
            : "1px solid rgba(255,255,255,0.08)",
      }}
    >
      {/* Letter + count */}
      <div className="flex items-start justify-between">
        <span
          className="text-3xl font-bold leading-none"
          style={{
            color: done ? "#22c55e" : isActive ? "#22d3ee" : "#e4e4e7",
          }}
        >
          {letter}
        </span>
        <div className="flex items-center gap-1 mt-1">
          {Array.from({ length: REPS_NEEDED }).map((_, i) => (
            <div
              key={i}
              className="w-2 h-2 rounded-full transition-all duration-300"
              style={{
                backgroundColor:
                  i < count
                    ? done
                      ? "#22c55e"
                      : "#06b6d4"
                    : "rgba(255,255,255,0.12)",
              }}
            />
          ))}
        </div>
      </div>

      {/* Progress bar */}
      <div
        className="h-1 rounded-full overflow-hidden"
        style={{ background: "rgba(255,255,255,0.08)" }}
      >
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{
            width: `${pct * 100}%`,
            background: done
              ? "#22c55e"
              : "linear-gradient(90deg, #06b6d4, #0ea5e9)",
          }}
        />
      </div>

      {/* Hint */}
      <p className="text-xs leading-snug" style={{ color: "#52525b" }}>
        {LETTER_HINTS[letter]}
      </p>

      {/* Record button */}
      {!done && (
        <button
          onClick={onRecord}
          disabled={disabled || isRecording}
          className="flex items-center justify-center gap-1.5 py-2 rounded-xl text-xs font-medium transition-all duration-200"
          style={{
            background: isActive
              ? "rgba(6,182,212,0.2)"
              : "rgba(255,255,255,0.06)",
            border: isActive
              ? "1px solid rgba(6,182,212,0.4)"
              : "1px solid rgba(255,255,255,0.1)",
            color: isActive ? "#22d3ee" : "#a1a1aa",
            cursor: disabled ? "not-allowed" : "pointer",
            opacity: disabled && !isActive ? 0.5 : 1,
          }}
        >
          {isRecording && isActive ? (
            <>
              <Loader2 className="w-3 h-3 animate-spin" />
              Recording...
            </>
          ) : (
            <>
              <Mic className="w-3 h-3" />
              {count === 0 ? "Record" : `Record (${count}/${REPS_NEEDED})`}
            </>
          )}
        </button>
      )}

      {done && (
        <div
          className="flex items-center gap-1.5 py-2 justify-center rounded-xl text-xs font-medium"
          style={{
            background: "rgba(34,197,94,0.1)",
            color: "#22c55e",
          }}
        >
          <CheckCircle2 className="w-3 h-3" />
          Complete
        </div>
      )}
    </div>
  );
}

export default function TrainPage() {
  const {
    status,
    trainStatus,
    modelReady,
    connect,
    disconnect,
    trainRecord,
    trainModel,
  } = useMyoWs();

  const [wsUrl, setWsUrl] = useState("ws://localhost:8765");
  const [activeLetter, setActiveLetter] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [step, setStep] = useState<"connect" | "train" | "done">("connect");
  const prevTrainRef = useRef<typeof trainStatus>({});

  const isConnected = status === "connected";
  const totalDone = ASL_LETTERS.filter(
    (l) => (trainStatus[l.toLowerCase()] ?? 0) >= REPS_NEEDED,
  ).length;
  const allDone = totalDone === 26;

  // Detect recording completion (count incremented)
  useEffect(() => {
    if (!activeLetter) return;
    const lc = activeLetter.toLowerCase();
    const prev = prevTrainRef.current[lc] ?? 0;
    const curr = trainStatus[lc] ?? 0;
    if (curr > prev) {
      setIsRecording(false);
      setActiveLetter(null);
    }
    prevTrainRef.current = { ...trainStatus };
  }, [trainStatus, activeLetter]);

  // Move to train step on connect
  useEffect(() => {
    if (isConnected && step === "connect") setStep("train");
    if (status === "disconnected" && step !== "connect") setStep("connect");
  }, [isConnected, status, step]);

  // Move to done on model ready
  useEffect(() => {
    if (modelReady) {
      setIsTraining(false);
      setStep("done");
    }
  }, [modelReady]);

  const handleRecord = useCallback(
    (letter: string) => {
      if (!isConnected || isRecording) return;
      setActiveLetter(letter);
      setIsRecording(true);
      trainRecord(letter);
    },
    [isConnected, isRecording, trainRecord],
  );

  const handleTrainModel = useCallback(() => {
    if (!allDone || isTraining) return;
    setIsTraining(true);
    trainModel();
  }, [allDone, isTraining, trainModel]);

  return (
    <main
      className="min-h-screen pt-16 pb-20 px-4"
      style={{ backgroundColor: "#0a0a0a" }}
    >
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="py-8">
          <h1 className="text-2xl font-bold text-white tracking-tight">
            Training
          </h1>
          <p className="text-sm mt-1" style={{ color: "#52525b" }}>
            Record 5 samples of each ASL letter to build your personal model
          </p>
        </div>

        {/* Steps indicator */}
        <div className="flex items-center gap-2 mb-8">
          {(["connect", "train", "done"] as const).map((s, i) => {
            const active = step === s;
            const past =
              (s === "connect" && step !== "connect") ||
              (s === "train" && step === "done");
            return (
              <div key={s} className="flex items-center gap-2">
                {i > 0 && (
                  <ChevronRight
                    className="w-3 h-3"
                    style={{ color: "#3f3f46" }}
                  />
                )}
                <div
                  className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full"
                  style={{
                    background: active
                      ? "rgba(6,182,212,0.1)"
                      : past
                        ? "rgba(34,197,94,0.1)"
                        : "rgba(255,255,255,0.04)",
                    border: active
                      ? "1px solid rgba(6,182,212,0.3)"
                      : past
                        ? "1px solid rgba(34,197,94,0.2)"
                        : "1px solid rgba(255,255,255,0.08)",
                    color: active ? "#22d3ee" : past ? "#22c55e" : "#52525b",
                  }}
                >
                  {past ? (
                    <CheckCircle2 className="w-3 h-3" />
                  ) : active ? (
                    <Circle className="w-3 h-3" />
                  ) : (
                    <Circle className="w-3 h-3" />
                  )}
                  {s === "connect" ? "Connect" : s === "train" ? "Record" : "Done"}
                </div>
              </div>
            );
          })}
        </div>

        {/* Step 1: Connect */}
        {step === "connect" && (
          <div
            className="rounded-2xl p-8 flex flex-col gap-4 max-w-lg"
            style={{
              background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.08)",
            }}
          >
            <div className="flex items-center gap-3">
              <div
                className="w-10 h-10 rounded-xl flex items-center justify-center"
                style={{ background: "rgba(6,182,212,0.1)" }}
              >
                <Radio className="w-5 h-5" style={{ color: "#06b6d4" }} />
              </div>
              <div>
                <p className="text-sm font-medium text-white">
                  Connect to Echo server
                </p>
                <p className="text-xs mt-0.5" style={{ color: "#52525b" }}>
                  Start the server then connect
                </p>
              </div>
            </div>

            <div
              className="rounded-xl p-3 text-xs font-mono"
              style={{
                background: "rgba(0,0,0,0.4)",
                border: "1px solid rgba(255,255,255,0.07)",
                color: "#22c55e",
              }}
            >
              <span style={{ color: "#52525b" }}>$ </span>
              python scripts/live_translate.py --user alice --ws-port 8765
            </div>

            <div className="flex gap-2">
              <input
                type="text"
                value={wsUrl}
                onChange={(e) => setWsUrl(e.target.value)}
                className="flex-1 px-3 py-2 rounded-xl text-sm outline-none"
                style={{
                  backgroundColor: "rgba(255,255,255,0.06)",
                  border: "1px solid rgba(255,255,255,0.1)",
                  color: "#e4e4e7",
                }}
              />
              <button
                onClick={() => connect(wsUrl)}
                className="px-4 py-2 rounded-xl text-sm font-medium text-white cursor-pointer"
                style={{ background: "rgba(6,182,212,0.7)" }}
              >
                Connect
              </button>
            </div>

            {status === "connecting" && (
              <div
                className="flex items-center gap-2 text-xs"
                style={{ color: "#eab308" }}
              >
                <Loader2 className="w-3 h-3 animate-spin" />
                Connecting...
              </div>
            )}
          </div>
        )}

        {/* Step 2: Record */}
        {step === "train" && (
          <div className="flex flex-col gap-6">
            {/* Progress summary */}
            <div
              className="rounded-2xl p-5 flex items-center justify-between flex-wrap gap-4"
              style={{
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.08)",
              }}
            >
              <div>
                <p className="text-2xl font-bold text-white">
                  {totalDone}
                  <span className="text-base font-normal" style={{ color: "#52525b" }}>
                    {" "}/ 26 letters
                  </span>
                </p>
                <p className="text-xs mt-1" style={{ color: "#52525b" }}>
                  {allDone
                    ? "All letters recorded — ready to train!"
                    : `${26 - totalDone} letters remaining`}
                </p>
              </div>
              <div className="flex items-center gap-3">
                <div
                  className="h-2 w-48 rounded-full overflow-hidden"
                  style={{ background: "rgba(255,255,255,0.08)" }}
                >
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${(totalDone / 26) * 100}%`,
                      background: allDone
                        ? "#22c55e"
                        : "linear-gradient(90deg, #06b6d4, #0ea5e9)",
                    }}
                  />
                </div>
                <button
                  onClick={disconnect}
                  className="text-xs px-3 py-1.5 rounded-lg cursor-pointer"
                  style={{
                    color: "#71717a",
                    background: "rgba(255,255,255,0.04)",
                    border: "1px solid rgba(255,255,255,0.08)",
                  }}
                >
                  Disconnect
                </button>
              </div>
            </div>

            {/* Letter grid */}
            <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 xl:grid-cols-7 gap-3">
              {ASL_LETTERS.map((letter) => (
                <LetterCard
                  key={letter}
                  letter={letter}
                  count={trainStatus[letter.toLowerCase()] ?? 0}
                  isActive={activeLetter === letter}
                  isRecording={isRecording}
                  onRecord={() => handleRecord(letter)}
                  disabled={!isConnected || (isRecording && activeLetter !== letter)}
                />
              ))}
            </div>

            {/* Train model button */}
            {allDone && (
              <div
                className="rounded-2xl p-6 flex flex-col gap-4"
                style={{
                  background: "rgba(34,197,94,0.06)",
                  border: "1px solid rgba(34,197,94,0.2)",
                }}
              >
                <div className="flex items-center gap-3">
                  <CheckCircle2 className="w-5 h-5" style={{ color: "#22c55e" }} />
                  <div>
                    <p className="text-sm font-medium" style={{ color: "#22c55e" }}>
                      All 26 letters recorded
                    </p>
                    <p className="text-xs mt-0.5" style={{ color: "#52525b" }}>
                      Click below to train your personal DyFAV model (~10 seconds)
                    </p>
                  </div>
                </div>
                <button
                  onClick={handleTrainModel}
                  disabled={isTraining}
                  className="flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-medium transition-all duration-200 cursor-pointer"
                  style={{
                    background: isTraining
                      ? "rgba(255,255,255,0.06)"
                      : "rgba(34,197,94,0.2)",
                    border: "1px solid rgba(34,197,94,0.3)",
                    color: isTraining ? "#52525b" : "#22c55e",
                  }}
                >
                  {isTraining ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Training model...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4" />
                      Train My Model
                    </>
                  )}
                </button>
              </div>
            )}
          </div>
        )}

        {/* Step 3: Done */}
        {step === "done" && (
          <div
            className="rounded-2xl p-10 flex flex-col items-center gap-6 max-w-lg mx-auto text-center"
            style={{
              background: "rgba(34,197,94,0.06)",
              border: "1px solid rgba(34,197,94,0.2)",
            }}
          >
            <div
              className="w-16 h-16 rounded-2xl flex items-center justify-center"
              style={{ background: "rgba(34,197,94,0.15)" }}
            >
              <CheckCircle2 className="w-8 h-8" style={{ color: "#22c55e" }} />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white mb-2">
                Model trained!
              </h2>
              <p className="text-sm" style={{ color: "#71717a" }}>
                Your personal DyFAV model has been saved. Head to the
                Translate page to start signing.
              </p>
            </div>
            <a
              href="/translate"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-medium text-white"
              style={{ background: "rgba(6,182,212,0.7)" }}
            >
              <Radio className="w-4 h-4" />
              Start Translating
            </a>

            <div
              className="w-full rounded-xl p-4 text-left"
              style={{
                background: "rgba(0,0,0,0.3)",
                border: "1px solid rgba(255,255,255,0.06)",
              }}
            >
              <p className="text-xs mb-2" style={{ color: "#52525b" }}>
                About DyFAV
              </p>
              <p className="text-xs leading-relaxed" style={{ color: "#71717a" }}>
                Your model was trained using the DyFAV algorithm (Dynamic
                Feature Selection and Voting). Each of the 26 letter agents
                learned which of your 510 EMG+IMU features are most
                discriminative for that specific letter — tuned to your
                personal signing style.
              </p>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
