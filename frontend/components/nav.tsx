"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Radio, MessageSquare, History, User, Zap } from "lucide-react";
import { useScrollNav } from "@/hooks/use-scroll-nav";

const links = [
  { href: "/", label: "Home" },
  { href: "/translate", label: "Translate", icon: Radio },
  { href: "/conversation", label: "Conversation", icon: MessageSquare },
  { href: "/history", label: "History", icon: History },
  { href: "/profile", label: "Profile", icon: User },
];

export function Nav() {
  const visible = useScrollNav();
  const pathname = usePathname();

  return (
    <nav
      className="fixed top-0 left-0 right-0 z-50 backdrop-blur-md border-b"
      style={{
        backgroundColor: "rgba(10, 10, 10, 0.85)",
        borderColor: "rgba(255,255,255,0.07)",
        transform: visible ? "translateY(0)" : "translateY(-100%)",
        opacity: visible ? 1 : 0,
        transition: "transform 0.3s ease, opacity 0.3s ease",
      }}
    >
      <div className="max-w-6xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2 group">
          <div
            className="w-7 h-7 rounded-lg flex items-center justify-center"
            style={{ background: "linear-gradient(135deg, #06b6d4, #0891b2)" }}
          >
            <Zap className="w-4 h-4 text-white" />
          </div>
          <span className="text-sm font-bold tracking-tight text-white">
            MAIA
          </span>
        </Link>

        <div className="flex items-center gap-1">
          {links.map((link) => {
            const active = pathname === link.href;
            const Icon = link.icon;
            return (
              <Link
                key={link.href}
                href={link.href}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-200"
                style={{
                  color: active ? "#22d3ee" : "#a1a1aa",
                  backgroundColor: active
                    ? "rgba(6,182,212,0.12)"
                    : "transparent",
                  border: active
                    ? "1px solid rgba(6,182,212,0.2)"
                    : "1px solid transparent",
                }}
              >
                {Icon && <Icon className="w-3.5 h-3.5" />}
                {link.label}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
