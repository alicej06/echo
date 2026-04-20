"use client";
import { useState, useCallback, useEffect, useRef } from "react";
import {
  Mic,
  CheckCircle2,
  Circle,
  Radio,
  Play,
  Loader2,
  ChevronRight,
} from "lucide-react";
import { useMyoWs } from "@/hooks/use-myo-ws";

const PHRASES = [
  "hello",
  "my",
  "name",
  "echo",
  "nice to meet you",
  "how are you",
  "thank you",
  "great",
  "what's your name",
] as const;

const REPS_NEEDED = 5;
const REPS_TO_TRAIN = 3;
const NULL_KEY = "_null_";
const NULL_REPS_NEEDED = 35;
const NULL_REPS_TO_TRAIN = 30;

const PHRASE_HINTS: Record<string, string> = {
  hello: "Wave hand side to side",
  my: "Flat hand on chest",
  name: "Tap index + middle fingers together",
  echo: "Fingerspell E-C-H-O",
  "nice to meet you": "Flat hand slides off other palm",
  "how are you": "Bent fingers roll forward, then point",
  "thank you": "Flat hand from chin forward",
  great: "Thumbs up or fist push forward",
  "what's your name": "WH sign → point at person → name sign",
};

function PhraseCard({
  phrase,
  count,
  isListening,
  onRecord,
  disabled,
}: {
  phrase: string;
  count: number;
  isListening: boolean;
  onRecord: () => void;
  disabled: boolean;
}) {
  const done = count >= REPS_NEEDED;
  const hasEnough = count >= REPS_TO_TRAIN;
  const pct = Math.min(count / REPS_NEEDED, 1);

  return (
    <div
      className="rounded-2xl p-4 flex flex-col gap-3 transition-all duration-200"
      style={{
        background: isListening
          ? "rgba(6,182,212,0.08)"
          : done
            ? "rgba(34,197,94,0.06)"
            : "rgba(255,255,255,0.04)",
        border: isListening
          ? "1px solid rgba(6,182,212,0.4)"
          : done
            ? "1px solid rgba(34,197,94,0.2)"
            : "1px solid rgba(255,255,255,0.08)",
      }}
    >
      {/* Phrase text + rep dots */}
      <div className="flex items-start justify-between gap-2">
        <span
          className="text-base font-semibold leading-tight"
          style={{
            color: done ? "#22c55e" : isListening ? "#22d3ee" : "#e4e4e7",
          }}
        >
          {phrase}
        </span>
        <div className="flex items-center gap-1 mt-0.5 flex-shrink-0">
          {Array.from({ length: REPS_NEEDED }).map((_, i) => (
            <div
              key={i}
              className="w-2 h-2 rounded-full transition-all duration-300"
              style={{
                backgroundColor:
                  i < count
                    ? done
                      ? "#22c55e"
                      : hasEnough
                        ? "#a78bfa"
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
              : hasEnough
                ? "linear-gradient(90deg, #7c3aed, #a78bfa)"
                : "linear-gradient(90deg, #06b6d4, #0ea5e9)",
          }}
        />
      </div>

      {/* Hint */}
      <p className="text-xs leading-snug" style={{ color: "rgba(255,255,255,0.6)" }}>
        {PHRASE_HINTS[phrase]}
      </p>

      {/* Record / Complete button */}
      {done ? (
        <div className="flex items-center gap-2">
          <div
            className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-xl text-xs font-medium"
            style={{ background: "rgba(34,197,94,0.1)", color: "#22c55e" }}
          >
            <CheckCircle2 className="w-3 h-3" />
            Complete
          </div>
          <button
            onClick={onRecord}
            disabled={disabled || isListening}
            className="flex items-center justify-center gap-1.5 px-3 py-2 rounded-xl text-xs font-medium transition-all duration-200"
            style={{
              background: "rgba(255,255,255,0.06)",
              border: "1px solid rgba(255,255,255,0.1)",
              color: "#71717a",
              cursor: disabled ? "not-allowed" : "pointer",
              opacity: disabled ? 0.5 : 1,
            }}
          >
            {isListening ? (
              <Loader2 className="w-3 h-3 animate-spin" />
            ) : (
              <Mic className="w-3 h-3" />
            )}
            Add rep
          </button>
        </div>
      ) : (
        <button
          onClick={onRecord}
          disabled={disabled || isListening}
          className="flex items-center justify-center gap-1.5 py-2 rounded-xl text-xs font-medium transition-all duration-200"
          style={{
            background: isListening
              ? "rgba(6,182,212,0.2)"
              : "rgba(255,255,255,0.06)",
            border: isListening
              ? "1px solid rgba(6,182,212,0.4)"
              : "1px solid rgba(255,255,255,0.1)",
            color: isListening ? "#22d3ee" : "#a1a1aa",
            cursor: disabled ? "not-allowed" : "pointer",
            opacity: disabled && !isListening ? 0.5 : 1,
          }}
        >
          {isListening ? (
            <>
              <Loader2 className="w-3 h-3 animate-spin" />
              Listening...
            </>
          ) : (
            <>
              <Mic className="w-3 h-3" />
              {count === 0
                ? "Record"
                : `Record (${count}/${REPS_NEEDED})`}
            </>
          )}
        </button>
      )}
    </div>
  );
}

export default function TrainPage() {
  const {
    status,
    phraseTrainStatus,
    dtwModelReady,
    dtwCvAccuracy,
    connect,
    disconnect,
    trainPhrase,
    trainNull,
    trainPhrasesModel,
  } = useMyoWs();

  const [wsUrl, setWsUrl] = useState("ws://localhost:8765");
  const [listeningPhrase, setListeningPhrase] = useState<string | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [step, setStep] = useState<"connect" | "record" | "done">("connect");
  const prevTrainRef = useRef<typeof phraseTrainStatus>({});
  const listenTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const isConnected = status === "connected";

  const nullCount = phraseTrainStatus[NULL_KEY] ?? 0;

  // Derived counts
  const phrasesWithEnoughReps = PHRASES.filter(
    (p) => (phraseTrainStatus[p] ?? 0) >= REPS_TO_TRAIN,
  ).length;
  const totalComplete = PHRASES.filter(
    (p) => (phraseTrainStatus[p] ?? 0) >= REPS_NEEDED,
  ).length;
  const canTrainModel =
    PHRASES.every((p) => (phraseTrainStatus[p] ?? 0) >= REPS_TO_TRAIN) &&
    nullCount >= NULL_REPS_TO_TRAIN;

  // Detect recording completion (phrase count incremented)
  useEffect(() => {
    if (!listeningPhrase) return;
    const prev = prevTrainRef.current[listeningPhrase] ?? 0;
    const curr = phraseTrainStatus[listeningPhrase] ?? 0;
    if (curr > prev) {
      setListeningPhrase(null);
      if (listenTimeoutRef.current) {
        clearTimeout(listenTimeoutRef.current);
        listenTimeoutRef.current = null;
      }
    }
    prevTrainRef.current = { ...phraseTrainStatus };
  }, [phraseTrainStatus, listeningPhrase]);

  // Move to record step on connect
  useEffect(() => {
    if (isConnected && step === "connect") setStep("record");
    if (status === "disconnected" && step === "record") setStep("connect");
  }, [isConnected, status, step]);

  // Move to done when dtw model is ready
  useEffect(() => {
    if (dtwModelReady) {
      setIsTraining(false);
      setStep("done");
    }
  }, [dtwModelReady]);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (listenTimeoutRef.current) clearTimeout(listenTimeoutRef.current);
    };
  }, []);

  const handleRecord = useCallback(
    (phrase: string) => {
      if (!isConnected || listeningPhrase !== null) return;
      prevTrainRef.current = { ...phraseTrainStatus };
      setListeningPhrase(phrase);
      if (phrase === NULL_KEY) {
        trainNull();
      } else {
        trainPhrase(phrase);
      }
      // 12-second timeout fallback
      if (listenTimeoutRef.current) clearTimeout(listenTimeoutRef.current);
      listenTimeoutRef.current = setTimeout(() => {
        setListeningPhrase(null);
        listenTimeoutRef.current = null;
      }, 12000);
    },
    [isConnected, listeningPhrase, phraseTrainStatus, trainPhrase, trainNull],
  );

  const handleTrainModel = useCallback(() => {
    if (!canTrainModel || isTraining) return;
    setIsTraining(true);
    trainPhrasesModel();
  }, [canTrainModel, isTraining, trainPhrasesModel]);

  return (
    <main
      className="min-h-screen pt-16 pb-20 px-4"
      style={{ background: "linear-gradient(180deg, #9147C8 0%, #A066D8 30%, #C49AEE 65%, #DDD0F8 85%, #EDE8FF 100%)" }}
    >
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="py-8">
          <h1 className="text-2xl font-bold text-white tracking-tight">
            Training
          </h1>
          <p className="text-sm mt-1" style={{ color: "rgba(255,255,255,0.7)" }}>
            Record 5 samples of each phrase to build your personal model
          </p>
        </div>

        {/* Step indicator */}
        <div className="flex items-center gap-2 mb-8">
          {(["connect", "record", "done"] as const).map((s, i) => {
            const active = step === s;
            const past =
              (s === "connect" && step !== "connect") ||
              (s === "record" && step === "done");
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
                  ) : (
                    <Circle className="w-3 h-3" />
                  )}
                  {s === "connect"
                    ? "Connect"
                    : s === "record"
                      ? "Record"
                      : "Done"}
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
                <p className="text-xs mt-0.5" style={{ color: "rgba(255,255,255,0.6)" }}>
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
              <span style={{ color: "rgba(255,255,255,0.6)" }}>$ </span>
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
        {step === "record" && (
          <div className="flex flex-col gap-6">
            {/* Progress summary card */}
            <div
              className="rounded-2xl p-5 flex items-center justify-between flex-wrap gap-4"
              style={{
                background: "rgba(255,255,255,0.15)",
                border: "1px solid rgba(255,255,255,0.3)",
              }}
            >
              <div>
                <p className="text-2xl font-bold text-white">
                  {totalComplete}
                  <span
                    className="text-base font-normal"
                    style={{ color: "rgba(255,255,255,0.6)" }}
                  >
                    {" "}/ {PHRASES.length} phrases complete
                  </span>
                </p>
                <p className="text-xs mt-1" style={{ color: "rgba(255,255,255,0.6)" }}>
                  {canTrainModel
                    ? "All reps recorded — ready to train!"
                    : nullCount < NULL_REPS_TO_TRAIN
                      ? `Need ${NULL_REPS_TO_TRAIN - nullCount} more null movements · ${phrasesWithEnoughReps}/${PHRASES.length} phrases ready`
                      : `${phrasesWithEnoughReps}/${PHRASES.length} phrases have ≥${REPS_TO_TRAIN} reps`}
                </p>
              </div>
              <div className="flex items-center gap-3">
                <div
                  className="h-2 w-36 rounded-full overflow-hidden"
                  style={{ background: "rgba(255,255,255,0.08)" }}
                >
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${(phrasesWithEnoughReps / PHRASES.length) * 100}%`,
                      background: canTrainModel
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

            {/* Phrase cards grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {PHRASES.map((phrase) => (
                <PhraseCard
                  key={phrase}
                  phrase={phrase}
                  count={phraseTrainStatus[phrase] ?? 0}
                  isListening={listeningPhrase === phrase}
                  onRecord={() => handleRecord(phrase)}
                  disabled={
                    !isConnected ||
                    (listeningPhrase !== null && listeningPhrase !== phrase)
                  }
                />
              ))}
            </div>

            {/* Null / background class */}
            <div
              className="rounded-2xl p-4 flex flex-col gap-3"
              style={{
                background: listeningPhrase === NULL_KEY
                  ? "rgba(251,146,60,0.08)"
                  : nullCount >= NULL_REPS_TO_TRAIN
                    ? "rgba(34,197,94,0.06)"
                    : "rgba(255,255,255,0.04)",
                border: listeningPhrase === NULL_KEY
                  ? "1px solid rgba(251,146,60,0.4)"
                  : nullCount >= NULL_REPS_TO_TRAIN
                    ? "1px solid rgba(34,197,94,0.2)"
                    : "1px solid rgba(255,255,255,0.08)",
              }}
            >
              <div className="flex items-start justify-between gap-2">
                <div>
                  <span
                    className="text-base font-semibold leading-tight block"
                    style={{
                      color: nullCount >= NULL_REPS_TO_TRAIN
                        ? "#22c55e"
                        : listeningPhrase === NULL_KEY
                          ? "#fb923c"
                          : "#e4e4e7",
                    }}
                  >
                    Null / background
                  </span>
                  <span className="text-xs mt-0.5 block" style={{ color: "rgba(255,255,255,0.6)" }}>
                    Required — prevents false positives
                  </span>
                </div>
                <div className="flex items-center gap-1 mt-0.5 flex-shrink-0">
                  {Array.from({ length: NULL_REPS_NEEDED }).map((_, i) => (
                    <div
                      key={i}
                      className="w-2 h-2 rounded-full transition-all duration-300"
                      style={{
                        backgroundColor:
                          i < nullCount
                            ? nullCount >= NULL_REPS_TO_TRAIN
                              ? "#22c55e"
                              : "#fb923c"
                            : "rgba(255,255,255,0.12)",
                      }}
                    />
                  ))}
                </div>
              </div>

              <div
                className="h-1 rounded-full overflow-hidden"
                style={{ background: "rgba(255,255,255,0.08)" }}
              >
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${Math.min(nullCount / NULL_REPS_NEEDED, 1) * 100}%`,
                    background: nullCount >= NULL_REPS_TO_TRAIN ? "#22c55e" : "#fb923c",
                  }}
                />
              </div>

              <p className="text-xs leading-snug" style={{ color: "rgba(255,255,255,0.6)" }}>
                Vary each rep: arm resting at side, reaching for something, casual hand wave, pointing, transitions between signs, natural talking gestures. The more variety the better.
              </p>

              <button
                onClick={() => handleRecord(NULL_KEY)}
                disabled={
                  !isConnected ||
                  (listeningPhrase !== null && listeningPhrase !== NULL_KEY)
                }
                className="flex items-center justify-center gap-1.5 py-2 rounded-xl text-xs font-medium transition-all duration-200"
                style={{
                  background: listeningPhrase === NULL_KEY
                    ? "rgba(251,146,60,0.2)"
                    : "rgba(255,255,255,0.06)",
                  border: listeningPhrase === NULL_KEY
                    ? "1px solid rgba(251,146,60,0.4)"
                    : "1px solid rgba(255,255,255,0.1)",
                  color: listeningPhrase === NULL_KEY ? "#fb923c" : "#a1a1aa",
                  cursor: (!isConnected || (listeningPhrase !== null && listeningPhrase !== NULL_KEY))
                    ? "not-allowed"
                    : "pointer",
                  opacity: (!isConnected || (listeningPhrase !== null && listeningPhrase !== NULL_KEY)) ? 0.5 : 1,
                }}
              >
                {listeningPhrase === NULL_KEY ? (
                  <>
                    <Loader2 className="w-3 h-3 animate-spin" />
                    Listening...
                  </>
                ) : (
                  <>
                    <Mic className="w-3 h-3" />
                    {nullCount === 0
                      ? "Record movement"
                      : `Record movement (${nullCount}/${NULL_REPS_NEEDED})`}
                  </>
                )}
              </button>
            </div>

            {/* Train model button — shown when all phrases have ≥ REPS_TO_TRAIN reps */}
            {canTrainModel && (
              <div
                className="rounded-2xl p-6 flex flex-col gap-4"
                style={{
                  background: "rgba(34,197,94,0.06)",
                  border: "1px solid rgba(34,197,94,0.2)",
                }}
              >
                <div className="flex items-center gap-3">
                  <CheckCircle2
                    className="w-5 h-5"
                    style={{ color: "#22c55e" }}
                  />
                  <div>
                    <p
                      className="text-sm font-medium"
                      style={{ color: "#22c55e" }}
                    >
                      All phrases + null class ready
                    </p>
                    <p className="text-xs mt-0.5" style={{ color: "rgba(255,255,255,0.6)" }}>
                      Click below to train your personal model
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
                    cursor: isTraining ? "not-allowed" : "pointer",
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
              {dtwCvAccuracy != null && (
                <p
                  className="text-sm font-medium mb-2"
                  style={{ color: "#22c55e" }}
                >
                  Cross-validation accuracy:{" "}
                  {(dtwCvAccuracy * 100).toFixed(1)}%
                </p>
              )}
              <p className="text-sm" style={{ color: "#71717a" }}>
                Your personal phrase model has been saved. Head to the Translate
                page to start signing.
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
          </div>
        )}
      </div>
    </main>
  );
}
