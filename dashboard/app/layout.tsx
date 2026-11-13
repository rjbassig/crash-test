import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = { title: "CRASH TEST", description: "Spatiotemporal traffic collision risk forecasting in San Diego." };
export default function Layout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en"><body>{children}</body></html>; }
