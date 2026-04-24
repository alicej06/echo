"use client";
import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (!email || !password) {
      setError("Please fill in all fields");
      return;
    }

    setLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 500));

      // Store mock token in localStorage
      localStorage.setItem("userToken", "demo-token-" + Date.now());
      localStorage.setItem("userEmail", email);

      router.push("/home");
    } catch (err) {
      setError("Login failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main
      className="min-h-screen flex items-center justify-center"
      style={{
        background:
          "linear-gradient(180deg, #9147C8 0%, #A066D8 30%, #C49AEE 65%, #DDD0F8 85%, #EDE8FF 100%)",
      }}
    >
      <div className="flex flex-col items-center w-full px-10" style={{ maxWidth: 340 }}>
        <h1
          className="text-center mb-8"
          style={{
            fontSize: 40,
            fontWeight: 300,
            color: "rgba(255,255,255,0.88)",
            letterSpacing: "-1px",
            fontFamily: "Inter, sans-serif",
          }}
        >
          Sign In
        </h1>

        <form onSubmit={handleLogin} className="w-full">
          {error && (
            <div
              className="mb-4 p-3 rounded-lg text-sm text-center"
              style={{
                backgroundColor: "rgba(239, 68, 68, 0.15)",
                color: "rgba(255, 255, 255, 0.9)",
              }}
            >
              {error}
            </div>
          )}

          <div className="mb-4">
            <input
              type="email"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              style={{
                width: "100%",
                padding: "12px 16px",
                borderRadius: 12,
                border: "1px solid rgba(255,255,255,0.2)",
                background: "rgba(255,255,255,0.1)",
                backdropFilter: "blur(12px)",
                color: "rgba(255,255,255,0.92)",
                fontSize: 14,
                fontFamily: "Inter, sans-serif",
              }}
              disabled={loading}
            />
          </div>

          <div className="mb-6">
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              style={{
                width: "100%",
                padding: "12px 16px",
                borderRadius: 12,
                border: "1px solid rgba(255,255,255,0.2)",
                background: "rgba(255,255,255,0.1)",
                backdropFilter: "blur(12px)",
                color: "rgba(255,255,255,0.92)",
                fontSize: 14,
                fontFamily: "Inter, sans-serif",
              }}
              disabled={loading}
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            style={{
              width: "100%",
              padding: "14px",
              borderRadius: 12,
              background: "rgba(255,255,255,0.22)",
              backdropFilter: "blur(12px)",
              border: "1px solid rgba(255,255,255,0.35)",
              color: "rgba(255,255,255,0.92)",
              fontSize: 14,
              fontWeight: 600,
              cursor: loading ? "not-allowed" : "pointer",
              opacity: loading ? 0.6 : 1,
              fontFamily: "Inter, sans-serif",
            }}
          >
            {loading ? "Signing In..." : "Sign In"}
          </button>
        </form>

        <div className="mt-6 text-center">
          <p style={{ color: "rgba(255,255,255,0.7)", fontSize: 14 }}>
            Don't have an account?{" "}
            <Link
              href="/signup"
              style={{
                color: "rgba(255,255,255,0.92)",
                textDecoration: "underline",
                fontWeight: 600,
              }}
            >
              Sign Up
            </Link>
          </p>
        </div>
      </div>
    </main>
  );
}
