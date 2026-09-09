'use client'

import { useState, useRef, KeyboardEvent, ChangeEvent, useEffect } from 'react'
import { Paperclip, Camera, Mic, Send, X } from 'lucide-react'

interface InputAreaProps {
  onSendMessage: (text: string, image?: string) => void
  onOpenCamera: () => void
  selectedImage: string | null
  onImageSelect: (image: string) => void
  onImageRemove: () => void
  disabled: boolean
  dark?: boolean
  /** Si true, el mic del input se desactiva (usa Modo en vivo) */
  liveMode?: boolean
}

type SpeechRec = {
  lang: string
  continuous: boolean
  interimResults: boolean
  start: () => void
  stop: () => void
  abort?: () => void
  onstart: ((ev: Event) => void) | null
  onend: ((ev: Event) => void) | null
  onerror: ((ev: any) => void) | null
  onresult: ((ev: any) => void) | null
}

export default function InputArea({
  onSendMessage,
  onOpenCamera,
  selectedImage,
  onImageSelect,
  onImageRemove,
  disabled,
  dark = false,
  liveMode = false,
}: InputAreaProps) {
  const [input, setInput] = useState('')
  const [isRecording, setIsRecording] = useState(false)
  const [interim, setInterim] = useState('')
  const [micError, setMicError] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const recognitionRef = useRef<SpeechRec | null>(null)
  const onSendRef = useRef(onSendMessage)
  onSendRef.current = onSendMessage

  useEffect(() => {
    return () => {
      try {
        recognitionRef.current?.abort?.()
        recognitionRef.current?.stop()
      } catch {
        /* ignore */
      }
    }
  }, [])

  const handleSend = () => {
    if (input.trim() || selectedImage) {
      onSendMessage(input, selectedImage || undefined)
      setInput('')
      if (textareaRef.current) textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleFileUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onloadend = () => onImageSelect(reader.result as string)
    reader.readAsDataURL(file)
  }

  const stopListening = () => {
    try {
      recognitionRef.current?.stop()
    } catch {
      /* ignore */
    }
    recognitionRef.current = null
    setIsRecording(false)
    setInterim('')
  }

  const startListening = async () => {
    setMicError('')
    if (liveMode) {
      setMicError('')
      return
    }

    const w = window as unknown as {
      SpeechRecognition?: new () => SpeechRec
      webkitSpeechRecognition?: new () => SpeechRec
    }
    const SR = w.SpeechRecognition || w.webkitSpeechRecognition
    if (!SR) {
      setMicError('Dictado no disponible. Usa Chrome o Modo en vivo.')
      return
    }

    try {
      await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch {
      setMicError('Permiso de micrófono denegado.')
      return
    }

    const recognition = new SR()
    recognition.lang = 'es-MX'
    recognition.continuous = false
    recognition.interimResults = true

    recognition.onstart = () => {
      setIsRecording(true)
      setInterim('')
      setMicError('')
    }
    recognition.onend = () => {
      setIsRecording(false)
      recognitionRef.current = null
    }
    recognition.onerror = (ev: any) => {
      setIsRecording(false)
      const err = String(ev?.error || '')
      if (err === 'aborted' || err === 'no-speech') return
      setMicError(err === 'not-allowed' ? 'Permiso de mic denegado.' : `Mic: ${err || 'error'}`)
    }
    recognition.onresult = (event: any) => {
      let interimText = ''
      let finalText = ''
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const res = event.results[i]
        if (res.isFinal) finalText += res[0].transcript
        else interimText += res[0].transcript
      }
      if (interimText) setInterim(interimText)
      if (finalText.trim()) {
        setInterim('')
        setIsRecording(false)
        onSendRef.current(finalText.trim())
        try {
          recognition.stop()
        } catch {
          /* ignore */
        }
      }
    }

    recognitionRef.current = recognition
    try {
      recognition.start()
    } catch {
      setMicError('No se pudo iniciar el micrófono.')
      setIsRecording(false)
    }
  }

  const shell = dark
    ? 'border-[#2c2c30] bg-[#111113]/95'
    : 'border-[#cfd3e0] bg-white shadow-[0_8px_30px_rgba(10,10,11,0.06)]'
  const icon = dark
    ? 'text-[#8a8a90] hover:text-[#f2f2f3]'
    : 'text-[#5c5c64] hover:text-[#121214]'
  const text = dark
    ? 'text-[#f2f2f3] placeholder:text-[#8a8a90]'
    : 'text-[#121214] placeholder:text-[#8a8a90]'

  return (
    <div className="px-1 pb-1">
      {selectedImage && (
        <div className="mb-3 relative inline-block">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={selectedImage}
            alt="Vista previa"
            className="max-h-28 rounded-xl border border-black/5"
          />
          <button
            onClick={onImageRemove}
            className="absolute -top-2 -right-2 w-6 h-6 bg-text-dark text-white rounded-full flex items-center justify-center"
            type="button"
          >
            <X className="h-3 w-3" />
          </button>
        </div>
      )}

      {(isRecording || interim || micError) && (
        <p className={`mb-2 text-xs ${micError ? 'text-red-500' : dark ? 'text-blue-300' : 'text-primary-blue'}`}>
          {micError
            ? micError
            : isRecording
              ? `Escuchando… ${interim ? `“${interim}”` : 'habla ahora'}`
              : null}
        </p>
      )}

      <div className={`flex items-end gap-2 rounded-2xl border px-3 py-2.5 shadow-[0_12px_40px_rgba(15,23,42,0.06)] backdrop-blur ${shell}`}>
        <input
          type="file"
          ref={fileInputRef}
          accept="image/*"
          onChange={handleFileUpload}
          className="hidden"
        />

        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
          className={`p-2 rounded-xl transition-colors disabled:opacity-40 ${icon}`}
          title="Subir imagen"
        >
          <Paperclip className="w-5 h-5" />
        </button>

        <button
          type="button"
          onClick={onOpenCamera}
          disabled={disabled}
          className={`p-2 rounded-xl transition-colors disabled:opacity-40 ${icon}`}
          title="Cámara"
        >
          <Camera className="w-5 h-5" />
        </button>

        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => {
            setInput(e.target.value)
            if (textareaRef.current) {
              textareaRef.current.style.height = 'auto'
              textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 140)}px`
            }
          }}
          onKeyDown={handleKeyDown}
          placeholder="Escribe o toca el mic…"
          disabled={disabled}
          className={`flex-1 bg-transparent resize-none outline-none max-h-32 text-[15px] py-2 disabled:opacity-50 ${text}`}
          rows={1}
        />

        <button
          type="button"
          onClick={() => (isRecording ? stopListening() : void startListening())}
          disabled={disabled || liveMode}
          className={`p-2 rounded-xl transition-colors disabled:opacity-40 ${
            isRecording
              ? 'text-red-500 bg-red-50'
              : icon
          }`}
          title={liveMode ? 'Usa Modo en vivo' : isRecording ? 'Detener' : 'Hablar'}
        >
          <Mic className="w-5 h-5" />
        </button>

        <button
          type="button"
          onClick={handleSend}
          disabled={disabled || (!input.trim() && !selectedImage)}
          className="p-2.5 rounded-xl bg-[var(--alice-ink)] text-white hover:bg-deep-blue transition-colors disabled:opacity-40"
          title="Enviar"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}
