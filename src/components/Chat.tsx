'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import axios from 'axios'
import {
  MessageSquare,
  Plus,
  ChevronLeft,
  ChevronRight,
  BookOpen,
  Trash2,
  Settings2,
  Volume2,
  VolumeX,
  Moon,
  Sun,
  Ban,
  X,
} from 'lucide-react'
import { motion } from 'framer-motion'
import Message from './Message'
import InputArea from './InputArea'
import CameraModal from './CameraModal'
import AliceWordmark from './AliceWordmark'
import { useLiveConversation } from '../hooks/useLiveConversation'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000'

interface MessageType {
  id: string
  role: 'user' | 'assistant'
  content: string
  image?: string
}

interface ChatSession {
  id: string
  title: string
  messages: MessageType[]
  createdAt: number
  updatedAt: number
}

const createEmptySession = (): ChatSession => {
  const id =
    typeof crypto !== 'undefined' && 'randomUUID' in crypto
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`
  const now = Date.now()
  return {
    id,
    title: 'Nuevo chat',
    messages: [],
    createdAt: now,
    updatedAt: now,
  }
}

function sessionLabel(session: ChatSession | undefined | null): string {
  if (!session) return 'A.L.I.C.E.'
  const raw = (session.title || '').trim()
  if (!raw || raw === 'Nuevo chat') {
    return session.messages.length > 0 ? 'Chat' : 'A.L.I.C.E.'
  }
  return raw
}

export default function Chat() {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [isLoading, setIsLoading] = useState(false)
  const [showCamera, setShowCamera] = useState(false)
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [voiceEnabled, setVoiceEnabled] = useState(true)
  // null = aún no leído (evita SSR que pisa el tema guardado)
  const [isDarkTheme, setIsDarkTheme] = useState<boolean | null>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [location, setLocation] = useState<{
    lat: number
    lon: number
    accuracy?: number
  } | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const sessionIdRef = useRef<string | null>(null)
  const sessionsReadyRef = useRef(false)
  const settingsRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    sessionIdRef.current = currentSessionId
  }, [currentSessionId])

  // Leer tema una sola vez en el cliente
  useEffect(() => {
    let dark = true
    try {
      const ds = document.documentElement.dataset.theme
      const v = window.localStorage.getItem('alice_theme')
      if (ds === 'light' || v === 'light') dark = false
      else if (ds === 'dark' || v === 'dark') dark = true
    } catch {
      /* ignore */
    }
    setIsDarkTheme(dark)
  }, [])

  // Guardar solo después de haber leído
  useEffect(() => {
    if (isDarkTheme === null) return
    try {
      const next = isDarkTheme ? 'dark' : 'light'
      window.localStorage.setItem('alice_theme', next)
      document.documentElement.dataset.theme = next
      document.documentElement.style.backgroundColor = isDarkTheme ? '#0a0a0b' : '#f4f5f8'
      document.documentElement.style.colorScheme = isDarkTheme ? 'dark' : 'light'
      document.body.style.backgroundColor = isDarkTheme ? '#0a0a0b' : '#f4f5f8'
    } catch {
      /* ignore */
    }
  }, [isDarkTheme])

  useEffect(() => {
    if (typeof window === 'undefined' || !navigator.geolocation) return
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const payload = {
          lat: pos.coords.latitude,
          lon: pos.coords.longitude,
          accuracy: pos.coords.accuracy,
        }
        setLocation(payload)
        axios.post(`${API_URL}/api/location`, payload).catch(() => {})
      },
      () => {},
      { enableHighAccuracy: true, maximumAge: 60_000, timeout: 10_000 },
    )
  }, [])

  useEffect(() => {
    if (typeof window === 'undefined') return
    try {
      const stored = window.localStorage.getItem('alice_sessions')
      if (stored) {
        const parsed: ChatSession[] = JSON.parse(stored)
        if (parsed.length > 0) {
          setSessions(parsed)
          setCurrentSessionId(parsed[0].id)
          sessionsReadyRef.current = true
          return
        }
      }
    } catch {
      /* ignore */
    }
    const first = createEmptySession()
    setSessions([first])
    setCurrentSessionId(first.id)
    sessionsReadyRef.current = true
  }, [])

  useEffect(() => {
    if (typeof window === 'undefined') return
    if (!sessionsReadyRef.current || sessions.length === 0) return
    try {
      window.localStorage.setItem('alice_sessions', JSON.stringify(sessions))
    } catch {
      /* ignore */
    }
  }, [sessions])

  const currentSession = sessions.find((s) => s.id === currentSessionId)
  const messages = currentSession?.messages ?? []
  const headerContext = sessionLabel(currentSession)
  const hasConversationContext =
    !!currentSession &&
    currentSession.messages.length > 0 &&
    currentSession.title.trim() !== '' &&
    currentSession.title.trim() !== 'Nuevo chat'

  useEffect(() => {
    if (typeof document === 'undefined') return
    document.title = hasConversationContext ? headerContext : 'A.L.I.C.E.'
  }, [hasConversationContext, headerContext])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length, currentSessionId, messages[messages.length - 1]?.content])

  useEffect(() => {
    if (!settingsOpen) return
    const onDown = (e: MouseEvent) => {
      if (settingsRef.current && !settingsRef.current.contains(e.target as Node)) {
        setSettingsOpen(false)
      }
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [settingsOpen])

  const ensureSessionId = useCallback((): string => {
    if (sessionIdRef.current) return sessionIdRef.current
    const session = createEmptySession()
    setSessions([session])
    setCurrentSessionId(session.id)
    sessionIdRef.current = session.id
    return session.id
  }, [])

  const appendMessagesToSession = useCallback((sessionId: string, newMessages: MessageType[]) => {
    setSessions((prev) =>
      prev.map((session) => {
        if (session.id !== sessionId) return session
        const combined = [...session.messages, ...newMessages]
        let title = session.title
        const firstUserInNew = newMessages.find((m) => m.role === 'user')
        if (title === 'Nuevo chat' && firstUserInNew && firstUserInNew.content.trim()) {
          const base = firstUserInNew.content.trim()
          title = base.length > 40 ? `${base.slice(0, 37)}…` : base
        }
        return {
          ...session,
          messages: combined,
          title,
          updatedAt: Date.now(),
        }
      }),
    )
  }, [])

  const upsertAssistantMessage = useCallback((id: string, content: string) => {
    const sid = sessionIdRef.current
    if (!sid) return
    setSessions((prev) =>
      prev.map((session) => {
        if (session.id !== sid) return session
        const exists = session.messages.some((m) => m.id === id)
        const nextMessages = exists
          ? session.messages.map((m) => (m.id === id ? { ...m, content } : m))
          : [...session.messages, { id, role: 'assistant' as const, content }]
        return { ...session, messages: nextMessages, updatedAt: Date.now() }
      }),
    )
  }, [])

  const handleNewChat = () => {
    const newSession = createEmptySession()
    setSessions((prev) => [newSession, ...prev])
    setCurrentSessionId(newSession.id)
    sessionIdRef.current = newSession.id
  }

  const deleteSession = (id: string) => {
    setSessions((prev) => {
      const filtered = prev.filter((s) => s.id !== id)
      if (filtered.length === 0) {
        const first = createEmptySession()
        setCurrentSessionId(first.id)
        sessionIdRef.current = first.id
        return [first]
      }
      if (id === sessionIdRef.current) {
        setCurrentSessionId(filtered[0].id)
        sessionIdRef.current = filtered[0].id
      }
      return filtered
    })
  }

  const speakText = (text: string) => {
    if (typeof window === 'undefined' || !window.speechSynthesis || !voiceEnabled) return
    window.speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.lang = 'es-MX'
    utterance.rate = 1.05
    window.speechSynthesis.speak(utterance)
  }

  const live = useLiveConversation({
    location,
    voiceEnabled,
    createAssistantId: () => `${Date.now()}-assistant-live`,
    onUserUtterance: (text) => {
      const sid = ensureSessionId()
      appendMessagesToSession(sid, [
        { id: `${Date.now()}-user-live`, role: 'user', content: text },
      ])
    },
    onAssistantDelta: (id, text) => upsertAssistantMessage(id, text),
    onAssistantFinal: (id, text) => upsertAssistantMessage(id, text),
  })

  useEffect(() => {
    if (live.liveOn) live.stopLive()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const sendMessage = async (text: string, image?: string) => {
    if (!text.trim() && !image) return
    const sid = ensureSessionId()
    setSelectedImage(null)

    if (image) {
      appendMessagesToSession(sid, [
        { id: `${Date.now()}-user`, role: 'user', content: text, image },
      ])
      setIsLoading(true)
      try {
        const response = await axios.post(`${API_URL}/api/image`, {
          message: text || 'Analiza coherencia/autenticidad de esta imagen',
          image,
          location,
        })
        appendMessagesToSession(sid, [
          { id: `${Date.now()}-assistant`, role: 'assistant', content: response.data.response },
        ])
        if (voiceEnabled && response.data.response) speakText(response.data.response)
      } catch (error) {
        console.error('Error sending message:', error)
        appendMessagesToSession(sid, [
          {
            id: `${Date.now()}-error`,
            role: 'assistant',
            content: 'Lo siento, hubo un error al procesar tu mensaje.',
          },
        ])
      } finally {
        setIsLoading(false)
      }
      return
    }

    setIsLoading(true)
    try {
      await live.sendStreamText(text)
    } finally {
      setIsLoading(false)
    }
  }

  const shell = isDarkTheme !== false // null o true → oscuro hasta resolver; false → claro
  const themeResolved = isDarkTheme !== null
  const sidebarW = sidebarOpen ? 'w-[260px]' : 'w-[56px]'
  const mainMl = sidebarOpen ? 'ml-[260px]' : 'ml-[56px]'

  // Tema oscuro = tinta del logo; claro = papel
  const bgMain = shell ? 'bg-[#0a0a0b] text-[#f2f2f3]' : 'bg-[#f4f5f8] text-[#121214]'
  const bgSide = shell ? 'bg-[#111113] text-[#d4d4d8]' : 'bg-[#ececee] text-[#121214]'
  const hover = shell ? 'hover:bg-[#1a1a1d]' : 'hover:bg-black/[0.06]'
  const activeItem = shell ? 'bg-[#1a1a1d] text-white' : 'bg-white text-[#121214] shadow-sm'
  const border = shell ? 'border-[#2c2c30]' : 'border-[#cfd3e0]'

  return (
    <div className={`relative min-h-screen overflow-x-hidden ${bgMain}`}>
      <aside className={`fixed inset-y-0 left-0 z-30 flex flex-col transition-[width] duration-200 ${sidebarW} ${bgSide}`}>
        <div className={`flex items-center gap-1 p-2 ${sidebarOpen ? 'justify-between' : 'flex-col gap-1'}`}>
          <button
            type="button"
            onClick={() => setSidebarOpen((p) => !p)}
            className={`flex h-9 w-9 items-center justify-center rounded-md transition-colors ${hover}`}
            title={sidebarOpen ? 'Ocultar panel' : 'Mostrar panel'}
          >
            {sidebarOpen ? <ChevronLeft className="h-[18px] w-[18px]" /> : <ChevronRight className="h-[18px] w-[18px]" />}
          </button>
          <button
            type="button"
            onClick={handleNewChat}
            className={`flex h-9 w-9 items-center justify-center rounded-md transition-colors ${hover}`}
            title="Nuevo chat"
          >
            <Plus className="h-[18px] w-[18px]" />
          </button>
        </div>

        {sidebarOpen && (
          <div className="px-2 pb-2">
            <div className="mb-3 flex justify-center px-2 pt-1">
              <AliceWordmark size="sm" metallic darkBg={shell} />
            </div>
            <button
              type="button"
              onClick={handleNewChat}
              className={`flex w-full items-center gap-2 rounded-md px-3 py-2.5 text-sm font-medium transition-colors ${hover}`}
            >
              <Plus className="h-4 w-4 opacity-80" />
              Nuevo chat
            </button>
          </div>
        )}

        {!sidebarOpen && (
          <div className="flex justify-center py-3">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={shell ? '/alice-eye-clear.webp?v=2' : '/alice-eye-dark.webp?v=2'}
              alt="A.L.I.C.E."
              width={28}
              height={28}
              className="h-7 w-7 object-contain"
            />
          </div>
        )}

        <div className="flex-1 overflow-y-auto chat-scrollbar px-2 pb-2">
          {sidebarOpen && (
            <p className="flex items-center gap-1.5 px-3 pb-2 pt-1 text-[11px] font-medium uppercase tracking-wider text-[#8b93a7]">
              <BookOpen className="h-3 w-3" />
              Conversaciones
            </p>
          )}
          <div className="space-y-0.5">
            {sessions.map((session) => {
              const active = session.id === currentSessionId
              const label =
                session.title && session.title.trim() ? session.title.trim() : 'Nuevo chat'
              return (
                <div key={session.id} className="group relative flex items-center">
                  <button
                    type="button"
                    onClick={() => {
                      setCurrentSessionId(session.id)
                      sessionIdRef.current = session.id
                    }}
                    className={`flex flex-1 items-center gap-2.5 rounded-md px-2.5 py-2 text-left text-[13px] transition-colors ${
                      active ? activeItem : hover
                    } ${!sidebarOpen ? 'justify-center' : ''}`}
                    title={label}
                  >
                    <MessageSquare className="h-4 w-4 flex-shrink-0 opacity-70" />
                    {sidebarOpen && <span className="truncate pr-6">{label}</span>}
                  </button>
                  {sidebarOpen && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation()
                        if (window.confirm('¿Eliminar este chat?')) deleteSession(session.id)
                      }}
                      className={`absolute right-1 rounded p-1.5 opacity-0 transition-opacity group-hover:opacity-100 ${hover} text-[#8b93a7]`}
                      aria-label="Eliminar"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>
              )
            })}
          </div>
        </div>

        <div className={`relative border-t p-2 ${border}`} ref={settingsRef}>
          <button
            type="button"
            onClick={() => setSettingsOpen((p) => !p)}
            className={`flex h-10 w-full items-center gap-2.5 rounded-md px-2.5 transition-colors ${hover} ${
              !sidebarOpen ? 'justify-center' : ''
            }`}
            title="Ajustes"
          >
            <Settings2 className="h-[18px] w-[18px] flex-shrink-0" />
            {sidebarOpen && <span className="text-sm">Ajustes</span>}
          </button>

          {settingsOpen && (
            <div
              className={`absolute bottom-14 z-40 min-w-[200px] rounded-lg border p-1.5 shadow-xl ${
                sidebarOpen ? 'left-2 right-2' : 'left-14'
              } ${shell ? 'border-[#2c2c30] bg-[#1a1a1d] text-[#f2f2f3]' : 'border-black/10 bg-white text-[#121214]'}`}
            >
              <button
                type="button"
                onClick={() => {
                  void live.abortGeneration()
                  setSettingsOpen(false)
                }}
                className={`flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm ${hover}`}
              >
                <Ban className="h-4 w-4" />
                Cortar respuesta
              </button>
              <button
                type="button"
                onClick={() => setVoiceEnabled((p) => !p)}
                className={`flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm ${hover}`}
              >
                {voiceEnabled ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
                TTS {voiceEnabled ? 'activado' : 'desactivado'}
              </button>
              <button
                type="button"
                onClick={() => {
                  if (!themeResolved) return
                  setIsDarkTheme((p) => !(p ?? true))
                }}
                className={`flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm ${hover}`}
              >
                {shell ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
                {shell ? 'Cambiar a tema claro' : 'Cambiar a tema oscuro'}
              </button>
              <button
                type="button"
                onClick={() => setSettingsOpen(false)}
                className={`flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm text-[#8b93a7] ${hover}`}
              >
                <X className="h-4 w-4" />
                Cerrar
              </button>
            </div>
          )}
        </div>
      </aside>

      <div className={`relative z-10 flex min-h-screen flex-col transition-[margin] duration-200 ${mainMl}`}>
        <header
          className={`sticky top-0 z-20 flex h-14 items-center justify-center border-b px-4 backdrop-blur ${border} ${
            shell ? 'bg-[#0a0a0b]/92' : 'bg-[#f4f5f8]/92'
          }`}
        >
          <h1
            className="max-w-[min(90%,42rem)] truncate text-center text-[15px] font-medium tracking-tight"
            title={headerContext}
          >
            {headerContext}
          </h1>
        </header>

        <div className="relative z-10 mx-auto flex w-full max-w-3xl flex-1 flex-col px-3 md:px-4">
          <div className="flex-1 pb-36 pt-6 chat-scrollbar">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-16 md:py-24">
                <motion.p
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.45 }}
                  className={`text-center text-2xl font-light tracking-tight md:text-3xl ${
                    shell ? 'text-[#f2f2f3]' : 'text-[#121214]'
                  }`}
                >
                  ¿En qué te ayudo hoy?
                </motion.p>
              </div>
            ) : (
              <>
                {messages.map((message) => (
                  <Message key={message.id} message={message} dark={shell} />
                ))}
                {isLoading && (
                  <div className="flex items-center gap-2 py-6 pl-1">
                    <div className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#8b93a7]" />
                    <div
                      className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#8b93a7]"
                      style={{ animationDelay: '0.2s' }}
                    />
                    <div
                      className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#8b93a7]"
                      style={{ animationDelay: '0.4s' }}
                    />
                  </div>
                )}
              </>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className={`sticky bottom-0 left-0 right-0 pb-4 pt-2 ${shell ? 'bg-[#0a0a0b]' : 'bg-[#f4f5f8]'}`}>
            <InputArea
              onSendMessage={sendMessage}
              onOpenCamera={() => setShowCamera(true)}
              selectedImage={selectedImage}
              onImageSelect={setSelectedImage}
              onImageRemove={() => setSelectedImage(null)}
              disabled={isLoading}
              dark={shell}
              liveMode={false}
            />
          </div>

          {showCamera && (
            <CameraModal
              onClose={() => setShowCamera(false)}
              onCapture={(imageData) => {
                setSelectedImage(imageData)
                setShowCamera(false)
              }}
            />
          )}
        </div>
      </div>
    </div>
  )
}
