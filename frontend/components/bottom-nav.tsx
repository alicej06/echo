"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Home, Hand, MessageSquare, Clock, User } from "lucide-react";

const TABS = [
  { href: "/home", label: "Home", icon: Home },
  { href: "/translate", label: "Translate", icon: Hand },
  { href: "/conversation", label: "Chat", icon: MessageSquare },
  { href: "/history", label: "History", icon: Clock },
  { href: "/profile", label: "Profile", icon: User },
];

const PURPLE = "#7C6FE0";
const GRAY = "#8E8E93";

export function BottomNav() {
  const pathname = usePathname();
  if (pathname === "/") return null;

  return (
    <nav
      className="fixed bottom-0 left-0 right-0 z-50 flex"
      style={{
        backgroundColor: "#FFFFFF",
        borderTop: "1px solid rgba(0,0,0,0.08)",
        paddingBottom: "env(safe-area-inset-bottom)",
      }}
    >
      {TABS.map(({ href, label, icon: Icon }) => {
        const active = pathname === href;
        return (
          <Link
            key={href}
            href={href}
            className="flex-1 flex flex-col items-center justify-center py-2 gap-0.5"
            style={{ color: active ? PURPLE : GRAY, textDecoration: "none" }}
          >
            <Icon size={22} strokeWidth={active ? 2.5 : 1.5} />
            <span style={{ fontSize: 10, fontWeight: active ? 600 : 400 }}>{label}</span>
          </Link>
        );
      })}
    </nav>
  );
}
