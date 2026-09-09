import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'A.L.I.C.E.',
  description: 'A.L.I.C.E. — asistente LLM',
  icons: {
    icon: [
      { url: '/icon.ico?v=2', sizes: 'any' },
      { url: '/alice-eye.webp?v=2', type: 'image/webp' },
      { url: '/favicon-32.png?v=2', sizes: '32x32', type: 'image/png' },
    ],
    apple: '/icon-192.png?v=2',
  },
}

/** Evita flash dark→light: aplica tema de localStorage antes de React. */
const themeBootScript = `
(function(){
  try {
    var t = localStorage.getItem('alice_theme');
    var dark = t !== 'light';
    var bg = dark ? '#0a0a0b' : '#f4f5f8';
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
    document.documentElement.style.backgroundColor = bg;
    document.documentElement.style.colorScheme = dark ? 'dark' : 'light';
  } catch (e) {}
})();
`

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="es" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeBootScript }} />
      </head>
      <body className="antialiased" suppressHydrationWarning>
        {children}
      </body>
    </html>
  )
}
