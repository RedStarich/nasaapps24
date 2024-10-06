import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Analytics } from "@vercel/analytics/react";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Unimori AI",
  description: "Your university life made easier",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <meta charSet="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <meta name="description" content="Your university life made easier" />
        <meta property="og:title" content="Unimori AI" />
        <meta property="og:description" content="Your university life made easier" />
        <meta property="og:image" content="/images/logo.png" />
        <meta property="og:url" content="https://unimori1.vercel.app/" />
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content="Unimori AI" />
        <meta name="twitter:description" content="Your university life made easier" />
        <meta name="twitter:image" content="/images/logo.png" />
        <link rel="icon" href="/images/favicon.ico" />
        <title>Unimori AI</title>
      </head>
      <body className={inter.className}>
        {children}
        <Analytics />
      </body>
    </html>
  );
}
