import type { Metadata } from "next";
import localFont from "next/font/local";
import { ForensicProvider } from "@/context/ForensicContext";
import "./globals.css";


const satoshi = localFont({
  src: "../fonts/Satoshi-Variable.woff2",
  variable: "--font-satoshi",
  display: "swap",
  weight: "300 900",
});

const dotoFont = localFont({
  src: "../fonts/Doto-Variable.woff2",
  variable: "--font-doto",
  display: "swap",
  weight: "100 900",
  declarations: [
    { prop: "unicode-range", value: "U+0030-0039" },
  ],
});

export const metadata: Metadata = {
  title: "HEARSAY — Autonomous Audio Forensics & Intelligence Workstation",
  description: "NSA Audio Authentication Challenge at HackGT 13: 8-modality forensic verification, real-time voice bar analysis, and acoustic intelligence copilot.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${satoshi.variable} ${dotoFont.variable}`}
    >
      <body className="antialiased font-sans min-h-screen selection:bg-sky-500 selection:text-white">
        <ForensicProvider>
          {children}
        </ForensicProvider>
      </body>
    </html>
  );
}
