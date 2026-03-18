import Link from "next/link";

export default function Home() {
  return (
    <div className="flex min-h-screen items-center justify-center" style={{ background: "var(--background)" }}>
      <div className="text-center space-y-8">
        <div>
          <h1 className="text-4xl font-bold tracking-tight mb-2" style={{ color: "var(--foreground)" }}>
            AI Proctor
          </h1>
          <p className="text-lg" style={{ color: "var(--muted)" }}>
            Real-time exam proctoring powered by computer vision
          </p>
        </div>

        <div className="flex gap-4 justify-center">
          <Link
            href="/candidate"
            className="px-8 py-4 rounded-lg text-white font-semibold text-lg transition-all hover:opacity-90 hover:scale-105"
            style={{ background: "#2563eb" }}
          >
            Join Exam
          </Link>
          <Link
            href="/admin"
            className="px-8 py-4 rounded-lg font-semibold text-lg transition-all hover:opacity-90 hover:scale-105"
            style={{ background: "var(--surface2)", color: "var(--foreground)", border: "1px solid var(--border)" }}
          >
            Admin Dashboard
          </Link>
        </div>
      </div>
    </div>
  );
}
