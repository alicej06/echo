import Link from "next/link";
import {
  Radio,
  MessageSquare,
  History,
  ArrowRight,
  Cpu,
  Bluetooth,
  Zap,
  Brain,
} from "lucide-react";

export default function HomePage() {
  return (
    <main className="min-h-screen" style={{ backgroundColor: "#0a0a0a" }}>
      {/* Hero */}
      <section
        className="relative min-h-screen flex flex-col items-center justify-center text-center px-4 pt-20 pb-32 overflow-hidden"
        style={{
          background:
            "radial-gradient(ellipse 80% 50% at 50% -10%, rgba(6,182,212,0.18) 0%, transparent 70%), #0a0a0a",
        }}
      >
        {/* Dot grid */}
        <div
          className="absolute inset-0 opacity-30"
          style={{
            backgroundImage:
              "radial-gradient(circle, rgba(255,255,255,0.055) 1px, transparent 1px)",
            backgroundSize: "32px 32px",
          }}
        />

        <div className="relative z-10 max-w-4xl mx-auto">
          <div
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium mb-8 border"
            style={{
              background: "rgba(6,182,212,0.1)",
              borderColor: "rgba(6,182,212,0.2)",
              color: "#22d3ee",
            }}
          >
            <span
              className="w-1.5 h-1.5 rounded-full blink"
              style={{ backgroundColor: "#22d3ee" }}
            />
            Myo armband + LSTM + Claude Haiku
          </div>

          <h1
            className="text-5xl md:text-7xl font-bold tracking-tight leading-none mb-6"
            style={{ fontFamily: "Inter, sans-serif" }}
          >
            <span
              style={{
                backgroundImage: "linear-gradient(135deg, #ffffff, #a1a1aa)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
              }}
            >
              Sign language,
            </span>
            <br />
            <span
              style={{
                backgroundImage: "linear-gradient(135deg, #06b6d4, #0ea5e9)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
              }}
            >
              in real time.
            </span>
          </h1>

          <p
            className="text-lg md:text-xl max-w-xl mx-auto mb-10 leading-relaxed"
            style={{ color: "#a1a1aa" }}
          >
            MAIA reads 8-channel EMG from your Myo armband, classifies ASL
            letters with a trained LSTM, and reconstructs full sentences with
            Claude Haiku. No camera. No lag.
          </p>

          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <Link
              href="/translate"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-xl font-semibold text-white transition-all duration-200 cursor-pointer"
              style={{
                background: "linear-gradient(135deg, #06b6d4, #0891b2)",
                boxShadow: "0 0 24px rgba(6,182,212,0.3)",
              }}
            >
              <Radio className="w-4 h-4" />
              Start translating
              <ArrowRight className="w-4 h-4" />
            </Link>
            <Link
              href="/conversation"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-xl font-semibold transition-all duration-200 cursor-pointer"
              style={{
                backgroundColor: "rgba(255,255,255,0.06)",
                border: "1px solid rgba(255,255,255,0.1)",
                color: "#e4e4e7",
              }}
            >
              <MessageSquare className="w-4 h-4" />
              Conversation mode
            </Link>
          </div>
        </div>

        {/* Pipeline diagram */}
        <div className="relative z-10 mt-20 max-w-3xl mx-auto w-full">
          <p
            className="text-xs font-medium uppercase tracking-widest mb-6"
            style={{ color: "#52525b" }}
          >
            Signal pipeline
          </p>
          <div className="grid grid-cols-5 items-center gap-2">
            {[
              { icon: Bluetooth, label: "Myo BLE", sub: "8ch sEMG" },
              null,
              { icon: Cpu, label: "LSTM", sub: "26-class" },
              null,
              { icon: Brain, label: "Claude Haiku", sub: "Sentence" },
            ].map((item, i) => {
              if (!item) {
                return (
                  <div key={i} className="flex items-center justify-center">
                    <div
                      className="w-full h-px"
                      style={{
                        background:
                          "linear-gradient(90deg, transparent, rgba(6,182,212,0.4), transparent)",
                      }}
                    />
                  </div>
                );
              }
              const Icon = item.icon;
              return (
                <div
                  key={i}
                  className="flex flex-col items-center gap-2 p-3 rounded-xl"
                  style={{
                    background: "rgba(255,255,255,0.04)",
                    border: "1px solid rgba(255,255,255,0.07)",
                  }}
                >
                  <div
                    className="w-8 h-8 rounded-lg flex items-center justify-center"
                    style={{ background: "rgba(6,182,212,0.12)" }}
                  >
                    <Icon className="w-4 h-4" style={{ color: "#22d3ee" }} />
                  </div>
                  <div className="text-center">
                    <p className="text-xs font-semibold text-white">
                      {item.label}
                    </p>
                    <p className="text-xs" style={{ color: "#52525b" }}>
                      {item.sub}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="py-24 px-4 max-w-5xl mx-auto">
        <div className="text-center mb-14">
          <h2 className="text-3xl md:text-4xl font-bold tracking-tight text-white mb-3">
            No compromises
          </h2>
          <p
            className="text-base max-w-md mx-auto"
            style={{ color: "#71717a" }}
          >
            Every component chosen for accuracy, latency, and real-world
            wearability.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[
            {
              icon: Zap,
              title: "Sub-100ms letter latency",
              body: "200Hz EMG sampling. 40-frame windows. LSTM inference runs locally. Letters appear before the gesture ends.",
            },
            {
              icon: Brain,
              title: "AI sentence reconstruction",
              body: "Claude Haiku takes the raw letter stream and corrects spelling, adds spaces, and returns natural language. Falls back to Ollama when offline.",
            },
            {
              icon: Bluetooth,
              title: "Wireless, no dongle",
              body: "dl-myo talks directly to the Myo over BLE via Bleak. No MyoConnect daemon. No USB dongle. Pairs in seconds.",
            },
            {
              icon: Cpu,
              title: "26-class LSTM classifier",
              body: "Trained on 8-channel sEMG for all ASL letters. 3rd-order Butterworth bandpass filter at 20-450Hz before inference.",
            },
          ].map((f) => {
            const Icon = f.icon;
            return (
              <div
                key={f.title}
                className="p-6 rounded-2xl transition-all duration-200"
                style={{
                  background: "rgba(255,255,255,0.04)",
                  border: "1px solid rgba(255,255,255,0.08)",
                }}
              >
                <div
                  className="w-10 h-10 rounded-xl flex items-center justify-center mb-4"
                  style={{ background: "rgba(6,182,212,0.1)" }}
                >
                  <Icon className="w-5 h-5" style={{ color: "#22d3ee" }} />
                </div>
                <h3 className="text-base font-semibold text-white mb-2">
                  {f.title}
                </h3>
                <p
                  className="text-sm leading-relaxed"
                  style={{ color: "#71717a" }}
                >
                  {f.body}
                </p>
              </div>
            );
          })}
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 px-4">
        <div
          className="max-w-2xl mx-auto text-center p-12 rounded-3xl"
          style={{
            background:
              "radial-gradient(ellipse at center, rgba(6,182,212,0.1) 0%, rgba(255,255,255,0.03) 80%)",
            border: "1px solid rgba(6,182,212,0.15)",
          }}
        >
          <h2 className="text-3xl font-bold text-white mb-3">Ready to sign?</h2>
          <p className="text-sm mb-8" style={{ color: "#71717a" }}>
            Connect your Myo, run the pipeline, and start translating. Or try
            demo mode without hardware.
          </p>
          <div className="flex gap-3 justify-center flex-wrap">
            <Link
              href="/translate"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-sm text-white transition-all duration-200 cursor-pointer"
              style={{
                background: "rgba(6,182,212,0.8)",
                boxShadow: "0 0 20px rgba(6,182,212,0.25)",
              }}
            >
              <Radio className="w-4 h-4" />
              Open live translation
            </Link>
            <Link
              href="/history"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium text-sm transition-all duration-200 cursor-pointer"
              style={{
                background: "rgba(255,255,255,0.05)",
                border: "1px solid rgba(255,255,255,0.1)",
                color: "#a1a1aa",
              }}
            >
              <History className="w-4 h-4" />
              View history
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer
        className="py-10 px-4 text-center border-t"
        style={{ borderColor: "rgba(255,255,255,0.06)" }}
      >
        <p className="text-xs" style={{ color: "#3f3f46" }}>
          MAIA ASL Communication System. Built for Echo.
        </p>
      </footer>
    </main>
  );
}
