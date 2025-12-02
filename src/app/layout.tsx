import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'A.L.I.C.E. - Asistente IA',
  description: 'Tu asistente IA personal',
  icons: {
    icon: '/icon.ico',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="es">
      <body className="antialiased">{children}</body>
    </html>
  )
}
