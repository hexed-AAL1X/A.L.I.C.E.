'use client'

import { motion } from 'framer-motion'

type AliceWordmarkProps = {
  className?: string
  darkBg?: boolean
  metallic?: boolean
  color?: string
  lineProgress?: number | null
  animateLine?: boolean
  duration?: number
  size?: 'sm' | 'md' | 'lg' | 'hero'
}

const SIZE: Record<NonNullable<AliceWordmarkProps['size']>, string> = {
  sm: 'text-lg tracking-[0.18em]',
  md: 'text-2xl tracking-[0.2em]',
  lg: 'text-4xl tracking-[0.22em] md:text-5xl',
  hero: 'text-5xl tracking-[0.24em] sm:text-6xl md:text-7xl lg:text-8xl',
}

const LINE_H: Record<NonNullable<AliceWordmarkProps['size']>, number> = {
  sm: 2,
  md: 2,
  lg: 2.5,
  hero: 3,
}

/** Oscuro: plata → blanco (sin azul) */
const GRAD_ON_DARK =
  'linear-gradient(135deg, #9a9aa0 0%, #d8d8dc 45%, #f5f5f6 100%)'

/** Claro: gris carbón → negro (sin azul) */
const GRAD_ON_LIGHT =
  'linear-gradient(135deg, #3a3a40 0%, #1a1a1d 50%, #0a0a0b 100%)'

export default function AliceWordmark({
  className = '',
  darkBg = false,
  metallic = true,
  color,
  lineProgress = null,
  animateLine = false,
  duration = 1.35,
  size = 'md',
}: AliceWordmarkProps) {
  const controlled = typeof lineProgress === 'number'
  const widthPct = controlled
    ? `${Math.max(0, Math.min(100, lineProgress * 100))}%`
    : animateLine
      ? undefined
      : '100%'

  const grad = darkBg ? GRAD_ON_DARK : GRAD_ON_LIGHT
  const solid = color ?? (darkBg ? '#f2f2f3' : '#121214')

  const textStyle = metallic
    ? {
        backgroundImage: grad,
        WebkitBackgroundClip: 'text' as const,
        backgroundClip: 'text' as const,
        color: 'transparent',
        WebkitTextFillColor: 'transparent',
      }
    : { color: solid }

  return (
    <div
      className={`relative inline-block select-none font-sans font-medium uppercase leading-none ${SIZE[size]} ${className}`}
      aria-label="A.L.I.C.E."
    >
      <span className="relative z-0 block pb-[0.22em]" style={textStyle}>
        A.L.I.C.E.
      </span>
      <motion.span
        aria-hidden
        className="pointer-events-none absolute left-0 block"
        style={{
          bottom: 0,
          height: LINE_H[size],
          transformOrigin: 'left center',
          width: controlled ? widthPct : undefined,
          ...(metallic ? { backgroundImage: grad } : { background: solid }),
        }}
        initial={controlled ? false : animateLine ? { width: '0%' } : { width: '100%' }}
        animate={controlled ? { width: widthPct } : { width: '100%' }}
        transition={
          controlled
            ? { duration: 0.35, ease: 'easeOut' }
            : animateLine
              ? { duration, ease: [0.22, 1, 0.36, 1] }
              : { duration: 0 }
        }
      />
    </div>
  )
}
