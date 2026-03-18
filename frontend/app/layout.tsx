import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Proctor",
  description: "AI-powered exam proctoring system",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased min-h-screen" style={{ background: "var(--background)", color: "var(--foreground)" }}>
        {children}
      </body>
    </html>
  );
}
