'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import AliceWordmark from './AliceWordmark'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000'
const THEME_KEY = 'alice_theme'

type Props = {
  onDone: () => void
  dark?: boolean
}

function readDarkPreference(fallback = true): boolean {
  if (typeof window === 'undefined') return fallback
  try {
    const ds = document.documentElement.dataset.theme
    if (ds === 'light') return false
    if (ds === 'dark') return true
    const v = window.localStorage.getItem(THEME_KEY)
    if (v === 'light') return false
    if (v === 'dark') return true
  } catch {
    /* ignore */
  }
  return fallback
}

/** Splash respeta tema guardado; un solo status + "..." */
export default function LoadingScreen({ onDone, dark: darkProp }: Props) {
  const [dark, setDark] = useState<boolean | null>(null)
  const [progress, setProgress] = useState(0.04)
  const [leaving, setLeaving] = useState(false)
  const [status, setStatus] = useState('Iniciando A.L.I.C.E.')
  const [dots, setDots] = useState(1)
  const doneRef = useRef(false)

  useEffect(() => {
    setDark(typeof darkProp === 'boolean' ? darkProp : readDarkPreference(true))
  }, [darkProp])

  const ink = dark ? '#f2f2f3' : '#121214'
  const muted = dark ? '#8a8a90' : '#5c5c64'
  const bg = dark ? '#0a0a0b' : '#f4f5f8'

  const finish = useCallback(() => {
    if (doneRef.current) return
    doneRef.current = true
    setProgress(1)
    setStatus('Listo')
    setLeaving(true)
    window.setTimeout(() => onDone(), 520)
  }, [onDone])

  useEffect(() => {
    if (leaving || dark === null) return
    const t = window.setInterval(() => setDots((d) => (d % 3) + 1), 420)
    return () => window.clearInterval(t)
  }, [leaving, dark])

  useEffect(() => {
    if (dark === null) return
    let cancelled = false
    let ticks = 0
    const maxTicks = 90

    const bump = (target: number) => {
      setProgress((p) => Math.max(p, Math.min(0.97, target)))
    }

    const poll = async () => {
      ticks += 1
      bump(0.08 + Math.min(0.45, ticks * 0.012))

      if (ticks === 1) setStatus('Conectando con el servidor')
      else if (ticks === 3) setStatus('Esperando respuesta del backend')
      else if (ticks === 8) setStatus('El servidor puede estar despertando')
      else if (ticks === 20) setStatus('Todavía sin respuesta, reintentando')

      try {
        const health = await fetch(`${API_URL}/api/health`, {
          method: 'GET',
          cache: 'no-store',
        })
        if (cancelled) return
        if (health.ok) {
          bump(0.62)
          let brain = ''
          try {
            const j = await health.json()
            brain = j?.brain || j?.model || ''
          } catch {
            /* ignore */
          }
          setStatus(brain ? `Backend activo (${brain})` : 'Backend activo')
          await new Promise((r) => setTimeout(r, 280))
          if (cancelled) return
          setStatus('Calentando el modelo')
          try {
            const warm = await fetch(`${API_URL}/api/warmup`, {
              method: 'POST',
              cache: 'no-store',
            })
            if (cancelled) return
            if (warm.ok) {
              bump(0.92)
              setStatus('Modelo listo')
            } else {
              bump(0.78)
              setStatus('Warmup parcial, entrando')
            }
          } catch {
            bump(0.75)
            setStatus('Warmup omitido, entrando')
          }
          if (!cancelled) finish()
          return
        }
        if (ticks === 1 || ticks % 10 === 0) {
          setStatus(`Servidor respondió ${health.status}, reintentando`)
        }
      } catch {
        if (ticks === 1) setStatus('Sin conexión al backend aún')
      }

      if (ticks >= maxTicks) {
        setStatus('Tiempo agotado, abriendo interfaz')
        if (!cancelled) finish()
        return
      }
      if (!cancelled) window.setTimeout(poll, 500)
    }

    void poll()
    return () => {
      cancelled = true
    }
  }, [finish, dark])

  const ellipsis = '.'.repeat(dots)

  // Mientras no hay tema: fondo del html (script), sin forzar oscuro
  if (dark === null) {
    return (
      <div
        className="fixed inset-0 z-[100]"
        style={{ background: 'inherit' }}
        aria-hidden
      />
    )
  }

  return (
    <div
      className={`fixed inset-0 z-[100] flex flex-col items-center justify-center px-6 transition-opacity duration-500 ${
        leaving ? 'pointer-events-none opacity-0' : 'opacity-100'
      }`}
      style={{ background: bg }}
      role="status"
      aria-live="polite"
      aria-label="A.L.I.C.E."
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(progress * 100)}
    >
      <AliceWordmark size="hero" darkBg={dark} metallic lineProgress={progress} />

      <div
        className="mt-12 flex min-h-[1.75rem] max-w-lg items-baseline justify-center gap-0.5 text-center font-mono text-[12px] md:text-[13px]"
        style={{ color: muted }}
      >
        <AnimatePresence mode="wait">
          <motion.span
            key={status}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.25 }}
            className="inline-block"
          >
            {status}
          </motion.span>
        </AnimatePresence>
        {!leaving && status !== 'Listo' && (
          <span
            className="inline-block w-7 text-left tracking-[0.2em]"
            style={{ color: ink, opacity: 0.55 }}
          >
            {ellipsis}
          </span>
        )}
      </div>
    </div>
  )
}
