"use client"

import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import { MessageSquare, Plus, ChevronLeft, ChevronRight, Trash2 } from 'lucide-react'
import { motion } from 'framer-motion'
import AOS from 'aos'
import 'aos/dist/aos.css'
import Lenis from 'lenis'
import Message from './Message'
import InputArea from './InputArea'
import CameraModal from './CameraModal'

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

export default function Chat() {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [showCamera, setShowCamera] = useState(false)
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [voiceEnabled, setVoiceEnabled] = useState(true)
  const [isDarkTheme, setIsDarkTheme] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (typeof window === 'undefined') return

    const lenis = new Lenis({
      smoothWheel: true,
      smoothTouch: false,
    })

    const raf = (time: number) => {
      lenis.raf(time)
      requestAnimationFrame(raf)
    }

    requestAnimationFrame(raf)

    AOS.init({
      duration: 700,
      easing: 'ease-out-quart',
      once: true,
    })

    return () => {
      lenis.destroy()
    }
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
          return
        }
      }
    } catch {
      // ignore
    }
    const first = createEmptySession()
    setSessions([first])
    setCurrentSessionId(first.id)
  }, [])

  useEffect(() => {
    if (typeof window === 'undefined') return
    if (sessions.length === 0) return
    try {
      window.localStorage.setItem('alice_sessions', JSON.stringify(sessions))
    } catch {
      // ignore
    }
  }, [sessions])

  const currentSession = sessions.find((s) => s.id === currentSessionId)
  const messages = currentSession?.messages ?? []

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length, currentSessionId])

  const ensureSession = (): ChatSession => {
    if (currentSession) return currentSession
    const session = createEmptySession()
    setSessions([session])
    setCurrentSessionId(session.id)
    return session
  }

  const appendMessagesToCurrent = (newMessages: MessageType[]) => {
    setSessions((prev) =>
      prev.map((session) => {
        if (session.id !== currentSessionId) return session
        const combined = [...session.messages, ...newMessages]
        let title = session.title
        const firstUserInNew = newMessages.find((m) => m.role === 'user')
        if (title === 'Nuevo chat' && firstUserInNew && firstUserInNew.content.trim()) {
          const base = firstUserInNew.content.trim()
          // Limitar títulos en el sidebar para que siempre se vean limpios
          title = base.length > 20 ? `${base.slice(0, 17)}...` : base
        }
        return {
          ...session,
          messages: combined,
          title,
          updatedAt: Date.now(),
        }
      }),
    )
  }

  const handleNewChat = () => {
    const newSession = createEmptySession()
    setSessions((prev) => [newSession, ...prev])
    setCurrentSessionId(newSession.id)
  }

  const deleteSession = (id: string) => {
    setSessions((prev) => {
      const filtered = prev.filter((s) => s.id !== id)

      if (filtered.length === 0) {
        const first = createEmptySession()
        setCurrentSessionId(first.id)
        return [first]
      }

      if (id === currentSessionId) {
        setCurrentSessionId(filtered[0].id)
      }

      return filtered
    })
  }

  const sendMessage = async (text: string, image?: string) => {
    if (!text.trim() && !image) return
    ensureSession()

    const userMessage: MessageType = {
      id: `${Date.now()}-user`,
      role: 'user',
      content: text,
      image,
    }
    appendMessagesToCurrent([userMessage])
    setSelectedImage(null)
    setIsLoading(true)

    try {
      let response
      if (image) {
        response = await axios.post(`${API_URL}/api/image`, {
          message: text || 'Describe esta imagen',
          image,
        })
      } else {
        response = await axios.post(`${API_URL}/api/chat`, {
          message: text,
        })
      }

      const assistantMessage: MessageType = {
        id: `${Date.now()}-assistant`,
        role: 'assistant',
        content: response.data.response,
      }
      appendMessagesToCurrent([assistantMessage])

      // Reproducir respuesta con voz si está habilitado
      if (voiceEnabled && response.data.response) {
        speakText(response.data.response)
      }
    } catch (error) {
      console.error('Error sending message:', error)
      const errorMessage: MessageType = {
        id: `${Date.now()}-error`,
        role: 'assistant',
        content: 'Lo siento, hubo un error al procesar tu mensaje.',
      }
      appendMessagesToCurrent([errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const sendAudioMessage = async (audioBlob: Blob) => {
    setIsLoading(true)
    ensureSession()

    try {
      const formData = new FormData()
      formData.append('audio', audioBlob, 'audio.wav')

      const response = await axios.post(`${API_URL}/api/audio`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      const transcription: string = response.data.transcription || ''

      const userMessage: MessageType = {
        id: `${Date.now()}-user-audio`,
        role: 'user',
        content: transcription || '[Audio sin transcripción]',
      }

      const assistantMessage: MessageType = {
        id: `${Date.now()}-assistant-audio`,
        role: 'assistant',
        content: response.data.response,
      }

      appendMessagesToCurrent([userMessage, assistantMessage])

      // Reproducir respuesta con voz si está habilitado
      if (voiceEnabled && response.data.response) {
        speakText(response.data.response)
      }
    } catch (error) {
      console.error('Error sending audio:', error)
      const errorMessage: MessageType = {
        id: `${Date.now()}-audio-error`,
        role: 'assistant',
        content: 'Lo siento, no pude procesar el audio.',
      }
      appendMessagesToCurrent([errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const speakText = (text: string) => {
    if (typeof window === 'undefined') return
    if (!window.speechSynthesis) return

    // Cancelar cualquier reproducción anterior
    window.speechSynthesis.cancel()

    const utterance = new SpeechSynthesisUtterance(text)
    utterance.lang = 'es-ES'
    utterance.rate = 1.0
    utterance.pitch = 1.0
    utterance.volume = 1.0

    window.speechSynthesis.speak(utterance)
  }

  const contentMarginClass = isSidebarCollapsed ? 'ml-20' : 'ml-64'
  const containerClass = isDarkTheme
    ? 'bg-gradient-to-b from-[#020617] via-[#020617] to-[#020617]'
    : 'bg-gradient-to-b from-white via-[#f4f6ff] to-[#e0e7ff]'
  const mainPanelClass = isDarkTheme ? 'bg-slate-800/90' : 'bg-white/80'
  const headerTitleClass = isDarkTheme ? 'text-white' : 'text-text-dark'
  const headerSubTitleClass = isDarkTheme ? 'text-slate-300' : 'text-text-gray'
  const sidebarClass = isDarkTheme
    ? 'bg-slate-800 border-r border-slate-700'
    : 'bg-white border-r border-gray-100'
  const sidebarHeaderClass = isDarkTheme ? 'border-b border-slate-700' : 'border-b border-gray-100'
  const sidebarTextClass = isDarkTheme ? 'text-slate-300' : 'text-text-gray'
  const sidebarHoverClass = isDarkTheme ? 'hover:bg-slate-700' : 'hover:bg-gray-50'

  return (
    <div className={`relative min-h-screen ${containerClass} overflow-x-hidden`}>
      {/* Sidebar limpio y colapsable */}
      <aside
        data-aos="fade-right"
        className={`fixed inset-y-0 left-0 z-20 flex flex-col transition-all duration-200 ${sidebarClass} shadow-[8px_0_30px_rgba(15,23,42,0.08)] ${
          isSidebarCollapsed
            ? 'w-20 min-w-[5rem] max-w-[5rem]'
            : 'w-64'
        }`}
      >
        <div
          className={`flex items-center justify-between ${sidebarHeaderClass} ${
            isSidebarCollapsed ? 'px-2 pt-4 pb-3' : 'px-4 pt-4 pb-3'
          }`}
        >
          <span className={`text-xs font-medium uppercase tracking-brand ${sidebarTextClass}`}>
            Chats
          </span>
          <button
            onClick={() => setIsSidebarCollapsed((prev) => !prev)}
            className={`w-8 h-8 flex items-center justify-center rounded-lg ${sidebarHoverClass} ${sidebarTextClass} transition-colors`}
            aria-label="Alternar sidebar"
          >
            {isSidebarCollapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <ChevronLeft className="h-4 w-4" />
            )}
          </button>
        </div>

        <div
          className={`${
            isSidebarCollapsed ? 'px-2 pt-2 pb-3' : 'px-4 py-4'
          } ${sidebarHeaderClass}`}
        >
          <button
            onClick={handleNewChat}
            className={`flex items-center justify-center gap-2 w-full text-sm font-medium rounded-lg transition-colors px-4 py-2.5 ${
              isDarkTheme
                ? 'bg-slate-700 text-white hover:bg-slate-600'
                : 'bg-text-dark text-white hover:bg-deep-blue'
            }`}
          >
            <Plus className="h-4 w-4" />
            {!isSidebarCollapsed && <span>Nuevo chat</span>}
          </button>
        </div>

        <div
          className={`flex-1 overflow-y-auto chat-scrollbar ${
            isSidebarCollapsed ? 'px-2 py-3' : 'px-4 py-4'
          }`}
        >
          {sessions.length === 0 ? (
            <p className={`text-sm px-2 ${sidebarTextClass}`}>
              Sin chats. Crea uno nuevo.
            </p>
          ) : (
            <div className="space-y-2">
              {sessions.map((session) => (
                <div
                  key={session.id}
                  className="flex items-center gap-1 group"
                  onClick={() => setCurrentSessionId(session.id)}
                >
                  <button
                    className={`flex-1 text-left px-3 py-3 rounded-lg text-sm transition-colors flex items-center gap-3 ${
                      session.id === currentSessionId
                        ? isDarkTheme
                          ? 'bg-slate-700 text-blue-400 font-medium'
                          : 'bg-blue-gray text-primary-blue font-medium'
                        : isDarkTheme
                        ? 'text-slate-300 hover:bg-slate-700/50'
                        : 'text-text-dark hover:bg-gray-50'
                    }`}
                  >
                    <MessageSquare className="h-4 w-4 flex-shrink-0" />
                    {!isSidebarCollapsed && (
                      <div className="flex-1 min-w-0">
                        <p className="truncate">
                          {(() => {
                            const raw = session.title && session.title.trim().length > 0 ? session.title.trim() : 'Nuevo chat'
                            return raw.length > 20 ? `${raw.slice(0, 17)}...` : raw
                          })()}
                        </p>
                      </div>
                    )}
                  </button>

                  {!isSidebarCollapsed && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        if (window.confirm('¿Eliminar este chat?')) {
                          deleteSession(session.id)
                        }
                      }}
                      className={`p-2 rounded-lg transition-colors opacity-0 group-hover:opacity-100 ${
                        isDarkTheme
                          ? 'text-slate-400 hover:text-red-400 hover:bg-red-900/20'
                          : 'text-text-gray/60 hover:text-red-500 hover:bg-red-50'
                      }`}
                      aria-label="Eliminar chat"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Configuración fija en el sidebar */}
        {!isSidebarCollapsed && (
          <div className={`${sidebarHeaderClass} px-4 py-3 space-y-2`}>
            <p className={`text-[11px] font-medium uppercase tracking-brand ${sidebarTextClass}`}>
              Configuración
            </p>
            <button
              onClick={() => setVoiceEnabled((prev) => !prev)}
              className={`w-full px-3 py-1.5 rounded-lg text-xs font-medium border text-left transition-colors ${
                voiceEnabled
                  ? isDarkTheme
                    ? 'bg-blue-500/20 text-blue-400 border-blue-500/50'
                    : 'bg-light-blue/10 text-primary-blue border-light-blue'
                  : isDarkTheme
                  ? 'bg-slate-700 text-slate-300 border-slate-600 hover:bg-slate-600'
                  : 'bg-white text-text-gray border-gray-200 hover:bg-gray-50'
              }`}
            >
              Voz {voiceEnabled ? 'activada' : 'desactivada'}
            </button>

            <button
              onClick={() => setIsDarkTheme((prev) => !prev)}
              className={`w-full px-3 py-1.5 rounded-lg text-xs font-medium border text-left transition-colors ${
                isDarkTheme
                  ? 'bg-slate-700 text-slate-300 border-slate-600 hover:bg-slate-600'
                  : 'bg-white text-text-gray border-gray-200 hover:bg-gray-50'
              }`}
            >
              Tema {isDarkTheme ? 'oscuro' : 'claro'}
            </button>
          </div>
        )}
      </aside>

      {/* Área principal ultra-limpia */}
      <div
        className={`relative z-10 flex min-h-screen flex-col ${mainPanelClass} backdrop-blur-sm ${contentMarginClass}`}
      >
        <header className="relative z-10 border-b border-white/60 px-10 py-6">
          <h1 className={`text-2xl font-display font-semibold tracking-brand ${headerTitleClass}`}>
            A.L.I.C.E.
          </h1>
          <p className={`text-sm mt-0.5 ${headerSubTitleClass}`}>Tu asistente IA personal</p>
        </header>

        <div className="relative z-10 flex-1 flex flex-col max-w-4xl mx-auto w-full">
          <div className="flex-1 px-8 pt-8 pb-32 md:pt-10 md:pb-40 chat-scrollbar">
            {messages.length === 0 ? (
              <div className="relative z-10 text-center py-14 md:py-18" data-aos="fade-up">
                <motion.h2
                  initial={{ opacity: 0, y: 18 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.7, ease: 'easeOut' }}
                  className={`text-4xl md:text-6xl lg:text-7xl font-display font-light mb-4 tracking-tight ${
                    isDarkTheme ? 'text-white' : 'text-text-dark'
                  }`}
                >
                  Hola, soy A.L.I.C.E.
                </motion.h2>
                <motion.p
                  initial={{ opacity: 0, y: 14 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.7, delay: 0.1, ease: 'easeOut' }}
                  className={`text-base md:text-lg max-w-xl mx-auto ${
                    isDarkTheme ? 'text-slate-300' : 'text-text-gray/80'
                  }`}
                >
                  Escribe, habla o comparte una imagen para comenzar.
                </motion.p>
              </div>
            ) : (
              <>
                {messages.map((message) => (
                  <Message key={message.id} message={message} />
                ))}
                {isLoading && (
                  <div className="flex items-center gap-2 py-6">
                    <div className="w-2 h-2 bg-light-blue rounded-full animate-pulse" />
                    <div
                      className="w-2 h-2 bg-light-blue rounded-full animate-pulse"
                      style={{ animationDelay: '0.2s' }}
                    />
                    <div
                      className="w-2 h-2 bg-light-blue rounded-full animate-pulse"
                      style={{ animationDelay: '0.4s' }}
                    />
                  </div>
                )}
              </>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div
            data-aos="fade-up"
            className="sticky bottom-0 left-0 right-0 px-8 pb-6 bg-white/90 backdrop-blur-sm border-t border-gray-100"
          >
            <InputArea
              onSendMessage={sendMessage}
              onSendAudio={sendAudioMessage}
              onOpenCamera={() => setShowCamera(true)}
              selectedImage={selectedImage}
              onImageSelect={setSelectedImage}
              onImageRemove={() => setSelectedImage(null)}
              disabled={isLoading}
              voiceEnabled={voiceEnabled}
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
