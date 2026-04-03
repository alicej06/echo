"use client";
import { useState, useEffect } from "react";
import { Bluetooth, Cpu, Volume2, SlidersHorizontal, Save } from "lucide-react";

interface Prefs {
  voiceRate: number;
  confidenceThreshold: number;
  debounceMsWindow: number;
  modelFile: string;
}

const DEFAULT_PREFS: Prefs = {
  voiceRate: 1.0,
  confidenceThreshold: 0.65,
  debounceMsWindow: 300,
  modelFile: "models/lstm_asl.pt",
};

export default function ProfilePage() {
  const [prefs, setPrefs] = useState<Prefs>(DEFAULT_PREFS);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    try {
      const raw = localStorage.getItem("maia_prefs");
      if (raw)
        setPrefs({ ...DEFAULT_PREFS, ...(JSON.parse(raw) as Partial<Prefs>) });
    } catch {
      // ignore
    }
  }, []);

  const save = () => {
    localStorage.setItem("maia_prefs", JSON.stringify(prefs));
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const update = <K extends keyof Prefs>(key: K, value: Prefs[K]) => {
    setPrefs((p) => ({ ...p, [key]: value }));
    setSaved(false);
  };

  const VOICE_OPTIONS: { label: string; value: number }[] = [
    { label: "Default (1x)", value: 1.0 },
    { label: "Slower (0.8x)", value: 0.8 },
    { label: "Faster (1.2x)", value: 1.2 },
  ];

  return (
    <main
      className="min-h-screen pt-16 pb-20 px-4"
      style={{ backgroundColor: "#0a0a0a" }}
    >
      <div className="max-w-2xl mx-auto">
        <div className="py-8">
          <h1 className="text-2xl font-bold text-white tracking-tight">
            Profile
          </h1>
          <p className="text-sm mt-1" style={{ color: "#52525b" }}>
            Device and recognition preferences
          </p>
        </div>

        <div className="flex flex-col gap-4">
          {/* Device */}
          <div
            className="rounded-2xl p-6"
            style={{
              background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.08)",
            }}
          >
            <div className="flex items-center gap-2 mb-5">
              <div
                className="w-8 h-8 rounded-lg flex items-center justify-center"
                style={{ background: "rgba(6,182,212,0.1)" }}
              >
                <Bluetooth className="w-4 h-4" style={{ color: "#22d3ee" }} />
              </div>
              <h2 className="text-sm font-semibold text-white">Device</h2>
            </div>

            <div className="flex flex-col gap-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-white">Myo Armband</p>
                  <p className="text-xs mt-0.5" style={{ color: "#52525b" }}>
                    BLE via dl-myo, no dongle required
                  </p>
                </div>
                <span
                  className="text-xs px-2.5 py-1 rounded-full font-medium"
                  style={{
                    background: "rgba(255,255,255,0.06)",
                    color: "#71717a",
                    border: "1px solid rgba(255,255,255,0.08)",
                  }}
                >
                  Not paired
                </span>
              </div>

              <div>
                <label
                  className="block text-xs font-medium mb-1.5"
                  style={{ color: "#52525b" }}
                >
                  Model file path
                </label>
                <input
                  type="text"
                  value={prefs.modelFile}
                  onChange={(e) => update("modelFile", e.target.value)}
                  className="w-full px-3 py-2 rounded-xl text-sm font-mono outline-none transition-all duration-200"
                  style={{
                    backgroundColor: "rgba(255,255,255,0.05)",
                    border: "1px solid rgba(255,255,255,0.1)",
                    color: "#a1a1aa",
                  }}
                />
              </div>

              <div
                className="rounded-xl p-3 text-xs font-mono"
                style={{
                  background: "rgba(0,0,0,0.4)",
                  border: "1px solid rgba(255,255,255,0.06)",
                  color: "#22c55e",
                }}
              >
                <span style={{ color: "#52525b" }}>$ </span>
                python calibrate_quick.py
              </div>
            </div>
          </div>

          {/* Voice */}
          <div
            className="rounded-2xl p-6"
            style={{
              background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.08)",
            }}
          >
            <div className="flex items-center gap-2 mb-5">
              <div
                className="w-8 h-8 rounded-lg flex items-center justify-center"
                style={{ background: "rgba(6,182,212,0.1)" }}
              >
                <Volume2 className="w-4 h-4" style={{ color: "#22d3ee" }} />
              </div>
              <h2 className="text-sm font-semibold text-white">Voice output</h2>
            </div>

            <div className="flex gap-2 flex-wrap">
              {VOICE_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => update("voiceRate", opt.value)}
                  className="px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 cursor-pointer"
                  style={
                    prefs.voiceRate === opt.value
                      ? {
                          background: "rgba(6,182,212,0.15)",
                          color: "#22d3ee",
                          border: "1px solid rgba(6,182,212,0.25)",
                        }
                      : {
                          background: "rgba(255,255,255,0.04)",
                          color: "#71717a",
                          border: "1px solid rgba(255,255,255,0.08)",
                        }
                  }
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          {/* Recognition */}
          <div
            className="rounded-2xl p-6"
            style={{
              background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.08)",
            }}
          >
            <div className="flex items-center gap-2 mb-5">
              <div
                className="w-8 h-8 rounded-lg flex items-center justify-center"
                style={{ background: "rgba(6,182,212,0.1)" }}
              >
                <SlidersHorizontal
                  className="w-4 h-4"
                  style={{ color: "#22d3ee" }}
                />
              </div>
              <h2 className="text-sm font-semibold text-white">
                Recognition tuning
              </h2>
            </div>

            <div className="flex flex-col gap-6">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-white">
                    Confidence threshold
                  </label>
                  <span
                    className="text-sm font-mono"
                    style={{ color: "#22d3ee" }}
                  >
                    {(prefs.confidenceThreshold * 100).toFixed(0)}%
                  </span>
                </div>
                <input
                  type="range"
                  min={50}
                  max={95}
                  step={5}
                  value={prefs.confidenceThreshold * 100}
                  onChange={(e) =>
                    update("confidenceThreshold", Number(e.target.value) / 100)
                  }
                  className="w-full cursor-pointer"
                  style={{ accentColor: "#06b6d4" }}
                  aria-label="Confidence threshold"
                />
                <div className="flex justify-between mt-1">
                  <span className="text-xs" style={{ color: "#3f3f46" }}>
                    50%
                  </span>
                  <span className="text-xs" style={{ color: "#3f3f46" }}>
                    95%
                  </span>
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-white">
                    Debounce window
                  </label>
                  <span
                    className="text-sm font-mono"
                    style={{ color: "#22d3ee" }}
                  >
                    {prefs.debounceMsWindow}ms
                  </span>
                </div>
                <input
                  type="range"
                  min={100}
                  max={600}
                  step={50}
                  value={prefs.debounceMsWindow}
                  onChange={(e) =>
                    update("debounceMsWindow", Number(e.target.value))
                  }
                  className="w-full cursor-pointer"
                  style={{ accentColor: "#06b6d4" }}
                  aria-label="Debounce window"
                />
                <div className="flex justify-between mt-1">
                  <span className="text-xs" style={{ color: "#3f3f46" }}>
                    100ms
                  </span>
                  <span className="text-xs" style={{ color: "#3f3f46" }}>
                    600ms
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Model info */}
          <div
            className="rounded-2xl p-6"
            style={{
              background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.08)",
            }}
          >
            <div className="flex items-center gap-2 mb-5">
              <div
                className="w-8 h-8 rounded-lg flex items-center justify-center"
                style={{ background: "rgba(6,182,212,0.1)" }}
              >
                <Cpu className="w-4 h-4" style={{ color: "#22d3ee" }} />
              </div>
              <h2 className="text-sm font-semibold text-white">Classifier</h2>
            </div>
            <div className="grid grid-cols-2 gap-3">
              {[
                { label: "Architecture", value: "LSTM" },
                { label: "Classes", value: "26 (A-Z)" },
                { label: "Input shape", value: "(40, 8)" },
                { label: "Sample rate", value: "200 Hz" },
                { label: "Filter", value: "Butterworth 3rd order" },
                { label: "Passband", value: "20-450 Hz" },
              ].map((stat) => (
                <div
                  key={stat.label}
                  className="px-3 py-2.5 rounded-xl"
                  style={{
                    background: "rgba(0,0,0,0.3)",
                    border: "1px solid rgba(255,255,255,0.06)",
                  }}
                >
                  <p className="text-xs mb-0.5" style={{ color: "#52525b" }}>
                    {stat.label}
                  </p>
                  <p className="text-sm font-medium text-white">{stat.value}</p>
                </div>
              ))}
            </div>
          </div>

          <button
            onClick={save}
            className="w-full flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-semibold text-sm transition-all duration-200 cursor-pointer"
            style={
              saved
                ? {
                    background: "rgba(34,197,94,0.15)",
                    color: "#22c55e",
                    border: "1px solid rgba(34,197,94,0.2)",
                  }
                : { background: "rgba(6,182,212,0.7)", color: "#fff" }
            }
          >
            <Save className="w-4 h-4" />
            {saved ? "Saved" : "Save preferences"}
          </button>
        </div>
      </div>
    </main>
  );
}
