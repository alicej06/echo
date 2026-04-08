import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { BottomNav } from "@/components/bottom-nav";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "echo — Bridging Communication",
  description: "AI-powered ASL translation wristband — real-time sign language detection and conversion",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="antialiased" style={{ backgroundColor: "#F0EFF8", minHeight: "100vh" }}>
        {children}
        <BottomNav />
      </body>
    </html>
  );
}
