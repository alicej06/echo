"use client";
import { useState, useEffect } from "react";
import { User, Watch, SlidersHorizontal, ChevronRight, Shield, HelpCircle, Info, Vibrate, Plus, Sparkles } from "lucide-react";
import { useRouter } from "next/navigation";

const BG = "linear-gradient(180deg, #9147C8 0%, #A066D8 30%, #C49AEE 65%, #DDD0F8 85%, #EDE8FF 100%)";
const CARD = "rgba(255,255,255,0.82)";
const PURPLE = "#7C6FE0";
const PURPLE_LIGHT = "rgba(124,111,224,0.12)";
const GREEN = "#34C759";
const TEXT = "#1C1C1E";
const TEXT2 = "#6C6C70";
const TEXT3 = "#8E8E93";
const SHADOW = "0 2px 12px rgba(80,0,150,0.1)";

interface Prefs {
  voiceRate: number;
  volume: number;
  vibration: boolean;
  vibrationIntensity: number;
  confidenceThreshold: number;
  debounceMsWindow: number;
  modelFile: string;
  selectedVoice: string;
  selectedVoiceId: string;
}

const DEFAULT_PREFS: Prefs = {
  voiceRate: 1.0,
  volume: 80,
  vibration: true,
  vibrationIntensity: 70,
  confidenceThreshold: 0.65,
  debounceMsWindow: 300,
  modelFile: "models/lstm_asl.pt",
  selectedVoice: "Lauren",
  selectedVoiceId: "l4Coq6695JDX9xtLqXDE",
};

const KNOWN_PHRASES = [
  "hello", "my", "name", "echo",
  "nice to meet you", "how are you", "thank you",
  "great", "what's your name",
];

const VOICES = [
  { name: "Lauren", desc: "Warm, conversational", voiceId: "l4Coq6695JDX9xtLqXDE" },
  { name: "Hale",   desc: "Clear, expressive",    voiceId: "wWWn96OtTHu1sn8SRGEr" },
  { name: "Posh Josh", desc: "Confident, polished", voiceId: "NXaTw4ifg0LAguvKuIwZ" },
];

function SectionLabel({ label }: { label: string }) {
  return (
    <p className="text-xs font-semibold uppercase tracking-wider mb-3 mt-6" style={{ color: "rgba(255,255,255,0.65)" }}>
      {label}
    </p>
  );
}

function SliderRow({
  label, value, min, max, unit, onChange,
}: {
  label: string; value: number; min: number; max: number; unit: string;
  onChange: (v: number) => void;
}) {
  return (
    <div className="mb-4">
      <div className="flex justify-between mb-2">
        <span className="text-sm" style={{ color: TEXT2 }}>{label}</span>
        <span className="text-sm font-medium" style={{ color: TEXT }}>{value}{unit}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full"
      />
    </div>
  );
}

export default function ProfilePage() {
  const router = useRouter();
  const [prefs, setPrefs] = useState<Prefs>(DEFAULT_PREFS);

  useEffect(() => {
    try {
      const raw = localStorage.getItem("maia_prefs");
      if (raw) setPrefs({ ...DEFAULT_PREFS, ...(JSON.parse(raw) as Partial<Prefs>) });
    } catch { /* ignore */ }
  }, []);

  const update = <K extends keyof Prefs>(key: K, value: Prefs[K]) => {
    setPrefs((p) => {
      const next = { ...p, [key]: value };
      localStorage.setItem("maia_prefs", JSON.stringify(next));
      return next;
    });
  };

  return (
    <main className="min-h-screen pb-24 px-4" style={{ background: BG }}>
      <div className="max-w-sm mx-auto pt-12">
        {/* Header */}
        <div className="flex items-center gap-3 mb-2">
          <div className="w-12 h-12 rounded-full flex items-center justify-center" style={{ backgroundColor: "rgba(255,255,255,0.2)", border: "1px solid rgba(255,255,255,0.35)" }}>
            <User size={22} style={{ color: "#fff" }} />
          </div>
          <div>
            <h1 className="text-2xl font-bold" style={{ color: "#fff" }}>Settings</h1>
            <p className="text-sm" style={{ color: "rgba(255,255,255,0.8)" }}>Customize your experience</p>
          </div>
        </div>

        {/* DEVICE */}
        <SectionLabel label="Device" />
        <div className="flex flex-col gap-3">
          {/* Wristband card */}
          <div className="rounded-2xl p-4" style={{ backgroundColor: CARD, boxShadow: SHADOW }}>
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 rounded-full flex items-center justify-center" style={{ backgroundColor: PURPLE_LIGHT }}>
                <Watch size={18} style={{ color: PURPLE }} />
              </div>
              <div className="flex-1">
                <p className="text-sm font-semibold" style={{ color: TEXT }}>echo Wristband</p>
                <div className="flex items-center gap-1 mt-0.5">
                  <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: GREEN }} />
                  <span className="text-xs" style={{ color: TEXT3 }}>Connected</span>
                </div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <p className="text-xs mb-0.5" style={{ color: TEXT3 }}>Battery</p>
                <p className="text-sm font-semibold" style={{ color: TEXT }}>85%</p>
              </div>
              <div>
                <p className="text-xs mb-0.5" style={{ color: TEXT3 }}>Firmware</p>
                <p className="text-sm font-semibold" style={{ color: TEXT }}>v2.1.3</p>
              </div>
            </div>
          </div>

          {/* Calibration */}
          <div
            className="rounded-2xl p-4 flex items-center gap-3"
            style={{ backgroundColor: CARD, boxShadow: SHADOW }}
          >
            <div className="w-10 h-10 rounded-full flex items-center justify-center" style={{ backgroundColor: PURPLE_LIGHT }}>
              <SlidersHorizontal size={18} style={{ color: PURPLE }} />
            </div>
            <div className="flex-1">
              <p className="text-sm font-semibold" style={{ color: TEXT }}>Calibration</p>
              <p className="text-xs mt-0.5" style={{ color: TEXT3 }}>Last calibrated 2 days ago</p>
            </div>
            <ChevronRight size={16} style={{ color: TEXT3 }} />
          </div>
        </div>

        {/* VOICE */}
        <SectionLabel label="Voice" />
        <div className="rounded-2xl p-4" style={{ backgroundColor: CARD, boxShadow: SHADOW }}>
          <p className="text-sm font-medium mb-3" style={{ color: TEXT }}>Select Voice</p>
          <div className="flex flex-col gap-2 mb-4">
            {VOICES.map(({ name, desc, voiceId }) => (
              <button
                key={name}
                onClick={() => {
                  update("selectedVoice", name);
                  update("selectedVoiceId", voiceId);
                }}
                className="flex items-center justify-between p-3 rounded-xl cursor-pointer text-left"
                style={{
                  border: prefs.selectedVoice === name
                    ? `2px solid ${PURPLE}`
                    : "1px solid rgba(0,0,0,0.08)",
                  backgroundColor: prefs.selectedVoice === name ? PURPLE_LIGHT : "transparent",
                }}
              >
                <div>
                  <p className="text-sm font-medium" style={{ color: TEXT }}>{name}</p>
                  <p className="text-xs mt-0.5" style={{ color: TEXT3 }}>{desc}</p>
                </div>
                {prefs.selectedVoice === name && (
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: PURPLE }} />
                )}
              </button>
            ))}
          </div>

          {/* Personalize voice entry point */}
          <button
            onClick={() => router.push("/voice-settings")}
            className="w-full flex items-center justify-between p-3 rounded-xl cursor-pointer mb-4"
            style={{ backgroundColor: PURPLE_LIGHT, border: `1px solid ${PURPLE}22` }}
          >
            <div className="flex items-center gap-2">
              <Sparkles size={15} style={{ color: PURPLE }} />
              <div className="text-left">
                <p className="text-sm font-semibold" style={{ color: PURPLE }}>Personalize your voice</p>
                <p className="text-xs mt-0.5" style={{ color: TEXT3 }}>Clone your voice or design one with AI</p>
              </div>
            </div>
            <ChevronRight size={15} style={{ color: PURPLE }} />
          </button>

          <SliderRow
            label="Speed"
            value={Math.round(prefs.voiceRate * 100)}
            min={60}
            max={150}
            unit="x"
            onChange={(v) => update("voiceRate", v / 100)}
          />
          <SliderRow
            label="Volume"
            value={prefs.volume}
            min={0}
            max={100}
            unit="%"
            onChange={(v) => update("volume", v)}
          />
        </div>

        {/* HAPTICS */}
        <SectionLabel label="Haptics" />
        <div className="rounded-2xl p-4" style={{ backgroundColor: CARD, boxShadow: SHADOW }}>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-full flex items-center justify-center" style={{ backgroundColor: PURPLE_LIGHT }}>
              <Vibrate size={18} style={{ color: PURPLE }} />
            </div>
            <div className="flex-1">
              <p className="text-sm font-semibold" style={{ color: TEXT }}>Vibration Feedback</p>
              <p className="text-xs mt-0.5" style={{ color: TEXT3 }}>Feel confirmation haptics</p>
            </div>
            {/* Toggle */}
            <button
              onClick={() => update("vibration", !prefs.vibration)}
              className="relative cursor-pointer flex-shrink-0"
              style={{
                width: 51,
                height: 31,
                borderRadius: 15.5,
                backgroundColor: prefs.vibration ? PURPLE : "#E5E5EA",
                transition: "background-color 0.2s",
                border: "none",
                padding: 0,
              }}
            >
              <div
                style={{
                  position: "absolute",
                  top: 2,
                  left: prefs.vibration ? 22 : 2,
                  width: 27,
                  height: 27,
                  borderRadius: "50%",
                  backgroundColor: "#fff",
                  boxShadow: "0 1px 3px rgba(0,0,0,0.3)",
                  transition: "left 0.2s",
                }}
              />
            </button>
          </div>

          {prefs.vibration && (
            <SliderRow
              label="Intensity"
              value={prefs.vibrationIntensity}
              min={0}
              max={100}
              unit="%"
              onChange={(v) => update("vibrationIntensity", v)}
            />
          )}
        </div>

        {/* PERSONALIZE */}
        <SectionLabel label="Personalize Gestures" />
        <div className="rounded-2xl p-4 mb-1" style={{ backgroundColor: CARD, boxShadow: SHADOW }}>
          <p className="text-xs mb-4" style={{ color: TEXT2 }}>
            Add more reps for any word to improve recognition accuracy for your signing style.
          </p>
          <div className="flex flex-col gap-2">
            {KNOWN_PHRASES.map((phrase) => (
              <div
                key={phrase}
                className="flex items-center justify-between py-2 px-1"
                style={{ borderBottom: "1px solid rgba(0,0,0,0.05)" }}
              >
                <span className="text-sm font-medium capitalize" style={{ color: TEXT }}>{phrase}</span>
                <button
                  onClick={() => router.push(`/teach?word=${encodeURIComponent(phrase)}&calibrate=true`)}
                  className="flex items-center gap-1 px-3 py-1.5 rounded-xl cursor-pointer text-xs font-semibold"
                  style={{ backgroundColor: PURPLE_LIGHT, color: PURPLE }}
                >
                  <Plus size={12} />
                  Add reps
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* MORE */}
        <SectionLabel label="More" />
        <div className="rounded-2xl overflow-hidden" style={{ backgroundColor: CARD, boxShadow: SHADOW }}>
          {[
            { icon: Shield, label: "Privacy", sub: "Manage your data" },
            { icon: HelpCircle, label: "Help & Support", sub: "FAQs and contact" },
            { icon: Info, label: "About", sub: "Version 1.0.0" },
          ].map(({ icon: Icon, label, sub }, i, arr) => (
            <div
              key={label}
              className="flex items-center gap-3 px-4 py-4 cursor-pointer"
              style={{
                borderBottom: i < arr.length - 1 ? "1px solid rgba(0,0,0,0.06)" : "none",
              }}
            >
              <Icon size={18} style={{ color: TEXT3 }} />
              <div className="flex-1">
                <p className="text-sm font-medium" style={{ color: TEXT }}>{label}</p>
                <p className="text-xs mt-0.5" style={{ color: TEXT3 }}>{sub}</p>
              </div>
              <ChevronRight size={16} style={{ color: TEXT3 }} />
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}
