import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Nav } from "@/components/nav";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "MAIA — ASL Communication",
  description: "Real-time ASL translation powered by EMG and AI",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`dark ${inter.variable}`}>
      <body
        className="antialiased min-h-screen"
        style={{ backgroundColor: "#0a0a0a" }}
      >
        <Nav />
        {children}
      </body>
    </html>
  );
}
