import Link from "next/link";

export default function WelcomePage() {
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
          className="text-center"
          style={{
            fontSize: 56,
            fontWeight: 300,
            color: "rgba(255,255,255,0.88)",
            letterSpacing: "-1px",
            marginBottom: 88,
            fontFamily: "Inter, sans-serif",
          }}
        >
          echo
        </h1>

        <div className="flex flex-col gap-3 w-full">
          <Link
            href="/home"
            className="block text-center"
            style={{
              padding: "15px",
              borderRadius: 100,
              background: "rgba(255,255,255,0.22)",
              backdropFilter: "blur(12px)",
              border: "1px solid rgba(255,255,255,0.35)",
              color: "rgba(255,255,255,0.92)",
              textDecoration: "none",
              fontSize: 12,
              fontWeight: 600,
              letterSpacing: "0.12em",
            }}
          >
            GET STARTED
          </Link>
          <Link
            href="/home"
            className="block text-center"
            style={{
              padding: "15px",
              borderRadius: 100,
              background: "rgba(255,255,255,0.12)",
              backdropFilter: "blur(12px)",
              border: "1px solid rgba(255,255,255,0.2)",
              color: "rgba(255,255,255,0.65)",
              textDecoration: "none",
              fontSize: 12,
              fontWeight: 600,
              letterSpacing: "0.12em",
            }}
          >
            SIGN IN
          </Link>
        </div>
      </div>
    </main>
  );
}
