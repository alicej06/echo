"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Watch, Hand, MessageSquare, ChevronRight, Clock, Lightbulb, Sparkles, LogOut } from "lucide-react";

const BG = "linear-gradient(180deg, #9147C8 0%, #A066D8 30%, #C49AEE 65%, #DDD0F8 85%, #EDE8FF 100%)";
const CARD = "rgba(255,255,255,0.82)";
const PURPLE = "#7C6FE0";
const PURPLE_LIGHT = "rgba(124,111,224,0.12)";
const TEXT = "#1C1C1E";
const TEXT2 = "#6C6C70";
const TEXT3 = "#8E8E93";
const SHADOW = "0 2px 12px rgba(80,0,150,0.1)";

interface SessionRecord {
  id: string;
  date: string;
  phrases?: string[];
  sentences?: string[];
  duration: number;
}

function timeAgo(dateStr: string) {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs} hr ago`;
  return `${Math.floor(hrs / 24)} days ago`;
}

export default function HomePage() {
  const router = useRouter();
  const [recent, setRecent] = useState<SessionRecord[]>([]);
  const [userEmail, setUserEmail] = useState("");

  useEffect(() => {
    try {
      const raw = localStorage.getItem("maia_sessions");
      if (raw) {
        const all = JSON.parse(raw) as SessionRecord[];
        setRecent([...all].reverse().slice(0, 3));
      }
      const email = localStorage.getItem("userEmail");
      setUserEmail(email || "");
    } catch { /* ignore */ }
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("userToken");
    localStorage.removeItem("userEmail");
    router.push("/login");
  };

  return (
    <main className="min-h-screen pb-24 px-4" style={{ background: BG }}>
      <div className="max-w-sm mx-auto pt-12">
        {/* Header */}
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold" style={{ color: "#fff" }}>echo</h1>
            <p className="text-sm mt-1" style={{ color: "rgba(255,255,255,0.8)" }}>Welcome back! Ready to communicate?</p>
          </div>
          <button
            onClick={handleLogout}
            style={{
              padding: "8px 12px",
              borderRadius: 8,
              background: "rgba(255,255,255,0.15)",
              border: "1px solid rgba(255,255,255,0.3)",
              color: "#fff",
              fontSize: 12,
              fontWeight: 600,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <LogOut size={14} />
            Logout
          </button>
        </div>

        {/* Device card */}
        <div
          className="rounded-2xl p-4 mb-6 flex items-center justify-between"
          style={{ backgroundColor: CARD, boxShadow: SHADOW }}
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full flex items-center justify-center" style={{ backgroundColor: PURPLE_LIGHT }}>
              <Watch size={20} style={{ color: PURPLE }} />
            </div>
            <div>
              <p className="text-sm font-semibold" style={{ color: TEXT }}>echo Wristband</p>
              <div className="flex items-center gap-1 mt-0.5">
                <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: "#34C759" }} />
                <p className="text-xs" style={{ color: TEXT3 }}>Connected</p>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-1.5" style={{ color: TEXT3 }}>
            <span style={{ fontSize: 12 }}>🔋</span>
            <span className="text-sm font-medium" style={{ color: TEXT2 }}>85%</span>
          </div>
        </div>

        {/* Teach Echo — hero CTA */}
        <Link
          href="/teach"
          className="rounded-2xl p-5 flex items-center gap-4 mb-6"
          style={{
            background: `linear-gradient(135deg, ${PURPLE} 0%, #9b8cf2 100%)`,
            textDecoration: "none",
            boxShadow: "0 4px 14px rgba(124,111,224,0.35)",
            border: "1.5px solid rgba(255,255,255,0.45)",
          }}
        >
          <div className="w-12 h-12 rounded-full flex items-center justify-center flex-shrink-0"
               style={{ backgroundColor: "rgba(255,255,255,0.2)" }}>
            <Sparkles size={22} color="#fff" />
          </div>
          <div className="flex-1">
            <p className="text-base font-bold text-white">Teach Echo a new gesture</p>
            <p className="text-xs mt-0.5" style={{ color: "rgba(255,255,255,0.75)" }}>
              Record any word or phrase in 60 seconds
            </p>
          </div>
          <ChevronRight size={18} color="rgba(255,255,255,0.7)" />
        </Link>

        {/* Quick Actions */}
        <p className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: "rgba(255,255,255,0.65)" }}>
          Quick Actions
        </p>
        <div className="flex flex-col gap-3 mb-6">
          <Link
            href="/translate"
            className="rounded-2xl p-4 flex items-center gap-3"
            style={{ backgroundColor: CARD, boxShadow: SHADOW, textDecoration: "none" }}
          >
            <div className="w-10 h-10 rounded-full flex items-center justify-center" style={{ backgroundColor: PURPLE_LIGHT }}>
              <Hand size={20} style={{ color: PURPLE }} />
            </div>
            <div className="flex-1">
              <p className="text-sm font-semibold" style={{ color: TEXT }}>Live Translation</p>
              <p className="text-xs mt-0.5" style={{ color: TEXT3 }}>ASL gestures to speech</p>
            </div>
            <ChevronRight size={16} style={{ color: TEXT3 }} />
          </Link>

          <Link
            href="/conversation"
            className="rounded-2xl p-4 flex items-center gap-3"
            style={{ backgroundColor: CARD, boxShadow: SHADOW, textDecoration: "none" }}
          >
            <div className="w-10 h-10 rounded-full flex items-center justify-center" style={{ backgroundColor: PURPLE_LIGHT }}>
              <MessageSquare size={20} style={{ color: PURPLE }} />
            </div>
            <div className="flex-1">
              <p className="text-sm font-semibold" style={{ color: TEXT }}>Conversation Mode</p>
              <p className="text-xs mt-0.5" style={{ color: TEXT3 }}>Two-way communication</p>
            </div>
            <ChevronRight size={16} style={{ color: TEXT3 }} />
          </Link>
        </div>

        {/* Recent */}
        <div className="flex items-center justify-between mb-3">
          <p className="text-xs font-semibold uppercase tracking-wider" style={{ color: "rgba(255,255,255,0.65)" }}>Recent</p>
          <Link href="/history" className="flex items-center gap-1 text-xs" style={{ color: "rgba(255,255,255,0.8)", textDecoration: "none" }}>
            View All <Clock size={12} />
          </Link>
        </div>

        <div className="flex flex-col gap-3 mb-6">
          {recent.length === 0 ? (
            <div className="rounded-2xl p-6 text-center" style={{ backgroundColor: CARD, boxShadow: SHADOW }}>
              <p className="text-sm" style={{ color: TEXT3 }}>No sessions yet. Start translating!</p>
            </div>
          ) : (
            recent.map((s) => (
              <div key={s.id} className="rounded-2xl p-4" style={{ backgroundColor: CARD, boxShadow: SHADOW }}>
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium" style={{ color: TEXT }}>
                    {(s.phrases ?? s.sentences ?? [])[0] || "Session"}
                  </p>
                  <p className="text-xs" style={{ color: TEXT3 }}>{timeAgo(s.date)}</p>
                </div>
              </div>
            ))
          )}
        </div>

        {/* Pro Tip */}
        <div className="rounded-2xl p-4 flex gap-3" style={{ backgroundColor: CARD, border: "1px solid rgba(255,255,255,0.4)" }}>
          <Lightbulb size={20} style={{ color: "#F59E0B", flexShrink: 0 }} />
          <div>
            <p className="text-sm font-semibold mb-1" style={{ color: TEXT }}>Pro Tip</p>
            <p className="text-xs leading-relaxed" style={{ color: TEXT2 }}>
              Keep your wrist steady during gestures for better recognition accuracy. Calibrate regularly for optimal results.
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}
