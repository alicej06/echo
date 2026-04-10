"use client";
import { useState, useEffect } from "react";
import { Search, Hand, MessageSquare, ChevronRight, Star } from "lucide-react";

const BG = "#F0EFF8";
const CARD = "#FFFFFF";
const PURPLE = "#7C6FE0";
const TEXT = "#1C1C1E";
const TEXT2 = "#6C6C70";
const TEXT3 = "#8E8E93";
const SHADOW = "0 1px 4px rgba(0,0,0,0.07)";

type FilterTab = "all" | "favorites" | "translations" | "chats";

interface SessionRecord {
  id: string;
  date: string;
  // legacy fields
  letters?: string[];
  sentences?: string[];
  // current field (saved by translate page)
  phrases?: string[];
  duration: number;
  type?: "translation" | "conversation";
  favorited?: boolean;
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

export default function HistoryPage() {
  const [sessions, setSessions] = useState<SessionRecord[]>([]);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<FilterTab>("all");
  const [favorites, setFavorites] = useState<Set<string>>(new Set());

  useEffect(() => {
    try {
      const raw = localStorage.getItem("maia_sessions");
      if (raw) setSessions(JSON.parse(raw) as SessionRecord[]);
      const favRaw = localStorage.getItem("maia_favorites");
      if (favRaw) setFavorites(new Set(JSON.parse(favRaw) as string[]));
    } catch { /* ignore */ }
  }, []);

  const toggleFavorite = (id: string) => {
    setFavorites((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      localStorage.setItem("maia_favorites", JSON.stringify([...next]));
      return next;
    });
  };

  const filtered = [...sessions]
    .reverse()
    .filter((s) => {
      const words = [...(s.phrases ?? s.sentences ?? []), ...(s.letters ?? [])];
      const text = words.join(" ").toLowerCase();
      if (search && !text.includes(search.toLowerCase())) return false;
      if (filter === "favorites" && !favorites.has(s.id)) return false;
      if (filter === "translations" && s.type === "conversation") return false;
      if (filter === "chats" && s.type !== "conversation") return false;
      return true;
    });

  const TABS: { key: FilterTab; label: string }[] = [
    { key: "all", label: "All" },
    { key: "favorites", label: "Favorites" },
    { key: "translations", label: "Translations" },
    { key: "chats", label: "Chats" },
  ];

  return (
    <main className="min-h-screen pb-24 px-4" style={{ backgroundColor: BG }}>
      <div className="max-w-sm mx-auto pt-12">
        {/* Header */}
        <div className="mb-5">
          <h1 className="text-2xl font-bold" style={{ color: TEXT }}>History</h1>
          <p className="text-sm mt-1" style={{ color: TEXT2 }}>Your past translations and conversations</p>
        </div>

        {/* Search */}
        <div
          className="flex items-center gap-2 rounded-2xl px-4 py-3 mb-4"
          style={{ backgroundColor: CARD, boxShadow: SHADOW }}
        >
          <Search size={16} style={{ color: TEXT3, flexShrink: 0 }} />
          <input
            type="text"
            placeholder="Search history..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="flex-1 bg-transparent outline-none text-sm"
            style={{ color: TEXT }}
          />
        </div>

        {/* Filter tabs */}
        <div className="flex gap-2 mb-5 overflow-x-auto pb-1" style={{ scrollbarWidth: "none" }}>
          {TABS.map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setFilter(key)}
              className="flex-shrink-0 px-4 py-2 rounded-full text-sm font-medium cursor-pointer"
              style={
                filter === key
                  ? { backgroundColor: PURPLE, color: "#fff" }
                  : { backgroundColor: CARD, color: TEXT3, border: "1px solid rgba(0,0,0,0.08)" }
              }
            >
              {label}
            </button>
          ))}
        </div>

        {/* History list */}
        {filtered.length === 0 ? (
          <div
            className="rounded-2xl p-12 flex flex-col items-center gap-3 text-center"
            style={{ backgroundColor: CARD, boxShadow: SHADOW }}
          >
            <p className="font-medium" style={{ color: TEXT }}>No sessions found</p>
            <p className="text-sm" style={{ color: TEXT3 }}>
              {sessions.length === 0
                ? "Start a live translation session and it will appear here."
                : "Try adjusting your search or filter."}
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            {filtered.map((s) => {
              const isConversation = s.type === "conversation";
              const isFav = favorites.has(s.id);
              const allWords = s.phrases ?? s.sentences ?? [];
              const preview = allWords[0] || (s.letters ?? []).slice(0, 20).join("") || "Session";
              const subtext = allWords[1] ?? null;

              return (
                <div
                  key={s.id}
                  className="rounded-2xl p-4"
                  style={{ backgroundColor: CARD, boxShadow: SHADOW }}
                >
                  <div className="flex items-start gap-2">
                    <div className="flex-1 min-w-0">
                      {/* Type + time */}
                      <div className="flex items-center gap-2 mb-1">
                        {isConversation ? (
                          <MessageSquare size={11} style={{ color: TEXT3 }} />
                        ) : (
                          <Hand size={11} style={{ color: TEXT3 }} />
                        )}
                        <span className="text-xs font-semibold uppercase tracking-wide" style={{ color: TEXT3 }}>
                          {isConversation ? "Conversation" : "Translation"}
                        </span>
                        <span className="text-xs" style={{ color: TEXT3 }}>• {timeAgo(s.date)}</span>
                      </div>

                      {/* Main text */}
                      <p className="text-sm font-medium truncate" style={{ color: TEXT }}>{preview}</p>
                      {subtext && (
                        <p className="text-xs mt-0.5 truncate" style={{ color: TEXT2 }}>{subtext}</p>
                      )}
                    </div>

                    <div className="flex items-center gap-2 flex-shrink-0 ml-2">
                      <button
                        onClick={() => toggleFavorite(s.id)}
                        className="cursor-pointer"
                        aria-label="Toggle favorite"
                      >
                        <Star
                          size={16}
                          fill={isFav ? "#F59E0B" : "none"}
                          style={{ color: isFav ? "#F59E0B" : TEXT3 }}
                        />
                      </button>
                      <ChevronRight size={16} style={{ color: TEXT3 }} />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </main>
  );
}
