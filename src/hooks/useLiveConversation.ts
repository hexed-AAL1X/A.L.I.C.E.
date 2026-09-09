"use client"

import { useCallback, useEffect, useRef, useState } from "react"

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000"

type LocationPayload = { lat: number; lon: number; accuracy?: number } | null

type SpeechRec = {
  lang: string
  continuous: boolean
  interimResults: boolean
  maxAlternatives?: number
  start: () => void
  stop: () => void
  abort?: () => void
  onstart: ((ev: Event) => void) | null
  onend: ((ev: Event) => void) | null
  onerror: ((ev: Event) => void) | null
  onresult: ((ev: any) => void) | null
  onspeechstart?: ((ev: Event) => void) | null
  onspeechend?: ((ev: Event) => void) | null
}

const BACKCHANNELS = ["Mm-hmm.", "Ya.", "Dale.", "Ok.", "Te escucho.", "A ver…"]

/** Fragmenta como habla una persona: oraciones y cláusulas cortas. */
function splitSpeakable(buffer: string): { ready: string[]; rest: string } {
  // Prioriza fin de frase; si el buffer crece mucho, corta en coma/y
  const sentence = buffer.split(/(?<=[.!?…:;])\s+/)
  if (sentence.length > 1) {
    const rest = sentence.pop() || ""
    return { ready: sentence.filter((s) => s.trim().length > 0), rest }
  }
  if (buffer.length > 72) {
    const clause = buffer.split(/(?<=,)\s+|(?<=\sy\s)/)
    if (clause.length > 1) {
      const rest = clause.pop() || ""
      const ready = clause.filter((s) => s.trim().length > 8)
      if (ready.length) return { ready, rest: clause.slice(ready.length).join(" ") + rest }
    }
  }
  return { ready: [], rest: buffer }
}

function pickBackchannel(userText: string): string {
  const t = userText.toLowerCase()
  if (/hola|hey|buenas/.test(t)) return "Hey."
  if (/gracias/.test(t)) return "De nada."
  if (/\?|qué|que|cómo|como|dónde|donde|cuándo|cuando/.test(t)) return "A ver…"
  return BACKCHANNELS[Math.floor(Math.random() * BACKCHANNELS.length)]
}

export function useLiveConversation(opts: {
  location: LocationPayload
  voiceEnabled: boolean
  onUserUtterance: (text: string) => void
  onAssistantDelta: (id: string, text: string) => void
  onAssistantFinal: (id: string, text: string) => void
  onInterim?: (text: string) => void
  createAssistantId: () => string
}) {
  const [liveOn, setLiveOn] = useState(false)
  const [listening, setListening] = useState(false)
  const [speaking, setSpeaking] = useState(false)
  const [status, setStatus] = useState("")
  const [interim, setInterim] = useState("")

  const liveOnRef = useRef(false)
  const speakingRef = useRef(false)
  const generationIdRef = useRef<string | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)
  const recognitionRef = useRef<SpeechRec | null>(null)
  const speakQueueRef = useRef<string[]>([])
  const speakBufRef = useRef("")
  const processingRef = useRef(false)
  const pendingUtteranceRef = useRef<string | null>(null)
  const lastFinalAtRef = useRef(0)
  const ignoreEchoUntilRef = useRef(0)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const bargeAnalyserRef = useRef<AnalyserNode | null>(null)
  const bargeRafRef = useRef<number | null>(null)
  const optsRef = useRef(opts)
  optsRef.current = opts

  const setSpeakingBoth = (v: boolean) => {
    speakingRef.current = v
    setSpeaking(v)
  }

  const stopSpeaking = useCallback(() => {
    if (typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.cancel()
    }
    speakQueueRef.current = []
    speakBufRef.current = ""
    setSpeakingBoth(false)
  }, [])

  const abortGeneration = useCallback(async () => {
    abortControllerRef.current?.abort()
    abortControllerRef.current = null
    const gid = generationIdRef.current
    if (gid) {
      try {
        await fetch(`${API_URL}/api/chat/abort`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ generation_id: gid }),
        })
      } catch {
        /* ignore */
      }
      generationIdRef.current = null
    }
    stopSpeaking()
  }, [stopSpeaking])

  const flushSpeakQueue = useCallback(() => {
    if (!optsRef.current.voiceEnabled) return
    if (typeof window === "undefined" || !window.speechSynthesis) return
    if (window.speechSynthesis.speaking || window.speechSynthesis.pending) return
    const next = speakQueueRef.current.shift()
    if (!next) {
      setSpeakingBoth(false)
      // ventana anti-eco tras hablar
      ignoreEchoUntilRef.current = Date.now() + 350
      return
    }
    setSpeakingBoth(true)
    ignoreEchoUntilRef.current = Date.now() + 10_000
    const u = new SpeechSynthesisUtterance(next)
    u.lang = "es-MX"
    // Ritmo más humano / un poco imperfecto
    u.rate = 0.98 + Math.random() * 0.12
    u.pitch = 0.95 + Math.random() * 0.15
    u.onend = () => flushSpeakQueue()
    u.onerror = () => flushSpeakQueue()
    window.speechSynthesis.speak(u)
  }, [])

  const enqueueSpeech = useCallback(
    (chunk: string, force = false) => {
      if (!optsRef.current.voiceEnabled) return
      speakBufRef.current += chunk
      if (force && speakBufRef.current.trim()) {
        speakQueueRef.current.push(speakBufRef.current.trim())
        speakBufRef.current = ""
        flushSpeakQueue()
        return
      }
      const { ready, rest } = splitSpeakable(speakBufRef.current)
      speakBufRef.current = rest
      for (const s of ready) speakQueueRef.current.push(s.trim())
      flushSpeakQueue()
    },
    [flushSpeakQueue],
  )

  const speakLocal = useCallback(
    (text: string) => {
      if (!optsRef.current.voiceEnabled) return
      speakQueueRef.current.unshift(text)
      flushSpeakQueue()
    },
    [flushSpeakQueue],
  )

  const streamChat = useCallback(
    async (message: string, liveMode: boolean) => {
      await abortGeneration()
      const o = optsRef.current
      const assistantId = o.createAssistantId()
      o.onAssistantDelta(assistantId, "")

      // Acuse inmediato: se siente humana antes del modelo
      if (liveMode && o.voiceEnabled) {
        speakLocal(pickBackchannel(message))
      }

      const controller = new AbortController()
      abortControllerRef.current = controller
      const gid =
        typeof crypto !== "undefined" && "randomUUID" in crypto
          ? crypto.randomUUID().replace(/-/g, "")
          : `${Date.now()}`
      generationIdRef.current = gid

      let full = ""
      setStatus("pensando…")

      try {
        const res = await fetch(`${API_URL}/api/chat/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify({
            message,
            location: o.location,
            generation_id: gid,
            live: liveMode,
          }),
        })
        if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`)

        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ""
        let firstToken = true

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          const blocks = buffer.split("\n\n")
          buffer = blocks.pop() || ""
          for (const block of blocks) {
            const lines = block.split("\n")
            let event = "message"
            let dataLine = ""
            for (const line of lines) {
              if (line.startsWith("event:")) event = line.slice(6).trim()
              if (line.startsWith("data:")) dataLine += line.slice(5).trim()
            }
            if (!dataLine) continue
            let data: Record<string, unknown> = {}
            try {
              data = JSON.parse(dataLine)
            } catch {
              continue
            }
            if (event === "ack" && typeof data.text === "string") {
              // backend también puede mandar acuse
              continue
            }
            if (event === "token" && typeof data.text === "string") {
              if (firstToken) {
                firstToken = false
                setStatus("hablando…")
              }
              full += data.text
              o.onAssistantDelta(assistantId, full)
              enqueueSpeech(data.text)
            }
            if (event === "aborted") {
              enqueueSpeech("", true)
              o.onAssistantFinal(assistantId, full || "…")
              setStatus(liveOnRef.current ? "escuchando…" : "")
              return
            }
            if (event === "error") {
              o.onAssistantFinal(assistantId, String(data.message || "Error"))
              setStatus(liveOnRef.current ? "escuchando…" : "")
              return
            }
            if (event === "done") {
              enqueueSpeech("", true)
              o.onAssistantFinal(assistantId, full)
              setStatus(liveOnRef.current ? "escuchando…" : "")
              return
            }
          }
        }
        o.onAssistantFinal(assistantId, full)
      } catch (e) {
        if ((e as Error).name === "AbortError") {
          o.onAssistantFinal(assistantId, full || "…")
        } else {
          o.onAssistantFinal(assistantId, (e as Error).message)
        }
        setStatus(liveOnRef.current ? "escuchando…" : "")
      } finally {
        processingRef.current = false
        // Si el usuario habló mientras respondíamos, procesar lo pendiente
        const pending = pendingUtteranceRef.current
        pendingUtteranceRef.current = null
        if (pending && liveOnRef.current) {
          processingRef.current = true
          optsRef.current.onUserUtterance(pending)
          void streamChat(pending, true)
        }
      }
    },
    [abortGeneration, enqueueSpeech, speakLocal],
  )

  const handleFinalUtterance = useCallback(
    async (text: string) => {
      const cleaned = text.trim()
      if (!cleaned) return
      // Evitar dobles finales del mismo reconocimiento
      const now = Date.now()
      if (now - lastFinalAtRef.current < 400) return
      lastFinalAtRef.current = now

      // Anti-eco: ignorar finales justo después/durante TTS
      if (Date.now() < ignoreEchoUntilRef.current && speakingRef.current) {
        // barge-in real: si hay bastante texto, sí cortar
        if (cleaned.split(/\s+/).length < 3) return
      }

      if (processingRef.current || speakingRef.current) {
        // Cola la siguiente intervención (más humano que ignorar)
        pendingUtteranceRef.current = cleaned
        await abortGeneration()
        return
      }

      processingRef.current = true
      setInterim("")
      optsRef.current.onInterim?.("")
      optsRef.current.onUserUtterance(cleaned)
      await streamChat(cleaned, true)
    },
    [abortGeneration, streamChat],
  )

  const stopBargeMonitor = useCallback(() => {
    if (bargeRafRef.current) cancelAnimationFrame(bargeRafRef.current)
    bargeRafRef.current = null
  }, [])

  const startBargeMonitor = useCallback(
    async (stream: MediaStream) => {
      try {
        const ctx = new AudioContext()
        audioCtxRef.current = ctx
        const source = ctx.createMediaStreamSource(stream)
        const analyser = ctx.createAnalyser()
        analyser.fftSize = 512
        source.connect(analyser)
        bargeAnalyserRef.current = analyser
        const data = new Uint8Array(analyser.frequencyBinCount)

        const tick = () => {
          analyser.getByteTimeDomainData(data)
          let sum = 0
          for (let i = 0; i < data.length; i++) {
            const v = (data[i] - 128) / 128
            sum += v * v
          }
          const rms = Math.sqrt(sum / data.length)
          // Usuario habla fuerte mientras ella habla → cortar
          if (speakingRef.current && rms > 0.08) {
            void abortGeneration()
            setStatus("te escucho…")
          }
          bargeRafRef.current = requestAnimationFrame(tick)
        }
        tick()
      } catch {
        /* sin monitor de barge-in */
      }
    },
    [abortGeneration],
  )

  const stopLive = useCallback(() => {
    liveOnRef.current = false
    setLiveOn(false)
    setListening(false)
    setStatus("")
    setInterim("")
    recognitionRef.current?.stop()
    recognitionRef.current = null
    stopBargeMonitor()
    audioCtxRef.current?.close().catch(() => {})
    audioCtxRef.current = null
    void abortGeneration()
  }, [abortGeneration, stopBargeMonitor])

  const startLive = useCallback(async () => {
    const w = window as unknown as {
      SpeechRecognition?: new () => SpeechRec
      webkitSpeechRecognition?: new () => SpeechRec
    }
    const SR = w.SpeechRecognition || w.webkitSpeechRecognition
    if (!SR) {
      setStatus("Usa Chrome para modo en vivo")
      return
    }

    try {
      const mic = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      })
      void startBargeMonitor(mic)
    } catch {
      setStatus("Permiso de micrófono necesario")
      return
    }

    const recognition = new SR()
    recognition.lang = "es-MX"
    recognition.continuous = true
    recognition.interimResults = true
    recognition.maxAlternatives = 1
    recognition.onstart = () => {
      setListening(true)
      setStatus("escuchando…")
    }
    recognition.onend = () => {
      setListening(false)
      if (liveOnRef.current) {
        try {
          recognition.start()
        } catch {
          /* ignore */
        }
      }
    }
    recognition.onerror = () => setStatus("mic — reintentando")
    recognition.onspeechstart = () => {
      if (speakingRef.current) void abortGeneration()
    }
    recognition.onresult = (event: any) => {
      let interimText = ""
      let finalText = ""
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const res = event.results[i]
        const transcript = res[0].transcript
        if (res.isFinal) finalText += transcript
        else interimText += transcript
      }
      if (interimText) {
        setInterim(interimText)
        optsRef.current.onInterim?.(interimText)
        // Barge-in temprano: ya empezaste a hablar
        if (speakingRef.current && interimText.trim().split(/\s+/).length >= 2) {
          void abortGeneration()
        }
      }
      if (finalText.trim()) void handleFinalUtterance(finalText)
    }
    recognitionRef.current = recognition
    liveOnRef.current = true
    setLiveOn(true)
    try {
      recognition.start()
    } catch {
      setStatus("No se pudo iniciar el mic")
    }
  }, [abortGeneration, handleFinalUtterance, startBargeMonitor])

  const toggleLive = useCallback(() => {
    if (liveOnRef.current) stopLive()
    else void startLive()
  }, [startLive, stopLive])

  const sendStreamText = useCallback(
    async (text: string) => {
      optsRef.current.onUserUtterance(text)
      await streamChat(text, liveOnRef.current)
    },
    [streamChat],
  )

  useEffect(() => {
    return () => {
      recognitionRef.current?.stop()
      stopBargeMonitor()
      void abortGeneration()
    }
  }, [abortGeneration, stopBargeMonitor])

  return {
    liveOn,
    listening,
    speaking,
    status,
    interim,
    toggleLive,
    stopLive,
    abortGeneration,
    sendStreamText,
    stopSpeaking,
  }
}
