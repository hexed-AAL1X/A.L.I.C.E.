'use client'

import { useState, useRef, KeyboardEvent, ChangeEvent } from 'react'
import { Paperclip, Camera, Mic, Send, X } from 'lucide-react'

interface InputAreaProps {
  onSendMessage: (text: string, image?: string) => void
  onSendAudio: (audioBlob: Blob) => void
  onOpenCamera: () => void
  selectedImage: string | null
  onImageSelect: (image: string) => void
  onImageRemove: () => void
  disabled: boolean
  voiceEnabled: boolean
}

export default function InputArea({
  onSendMessage,
  onSendAudio,
  onOpenCamera,
  selectedImage,
  onImageSelect,
  onImageRemove,
  disabled,
  voiceEnabled,
}: InputAreaProps) {
  const [input, setInput] = useState('')
  const [isRecording, setIsRecording] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])

  const handleSend = () => {
    if (input.trim() || selectedImage) {
      onSendMessage(input, selectedImage || undefined)
      setInput('')
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto'
      }
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
    if (file) {
      const reader = new FileReader()
      reader.onloadend = () => {
        onImageSelect(reader.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      
      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data)
      }

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' })
        onSendAudio(audioBlob)
        audioChunksRef.current = []
        stream.getTracks().forEach(track => track.stop())
      }

      mediaRecorderRef.current = mediaRecorder
      mediaRecorder.start()
      setIsRecording(true)
    } catch (error) {
      console.error('Error accessing microphone:', error)
      alert('No se pudo acceder al micrófono')
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
    }
  }

  return (
    <div className="border-t border-gray-100 px-8 py-5 bg-white">
      {/* Image Preview */}
      {selectedImage && (
        <div className="mb-4 relative inline-block">
          <img
            src={selectedImage}
            alt="Vista previa"
            className="max-h-32 rounded-lg border border-gray-200"
          />
          <button
            onClick={onImageRemove}
            className="absolute -top-2 -right-2 w-6 h-6 bg-text-dark text-white rounded-full flex items-center justify-center hover:bg-deep-blue transition-colors"
          >
            <X className="h-3 w-3" />
          </button>
        </div>
      )}

      {/* Input */}
      <div className="flex items-center gap-2 px-4 py-3 border border-gray-200 rounded-xl hover:border-light-blue focus-within:border-primary-blue transition-colors bg-white">
          <input
            type="file"
            ref={fileInputRef}
            accept="image/*"
            onChange={handleFileUpload}
            className="hidden"
          />

        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
          className="p-2 text-text-gray hover:text-primary-blue transition-colors disabled:opacity-50"
          title="Subir imagen"
        >
          <Paperclip className="w-5 h-5" />
        </button>

        <button
          onClick={onOpenCamera}
          disabled={disabled}
          className="p-2 text-text-gray hover:text-primary-blue transition-colors disabled:opacity-50"
          title="Tomar foto"
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
              textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`
            }
          }}
          onKeyDown={handleKeyDown}
          placeholder="Escribe tu mensaje..."
          disabled={disabled}
          className="flex-1 bg-transparent resize-none outline-none max-h-32 text-[15px] text-text-dark placeholder:text-text-gray disabled:opacity-50"
          rows={1}
        />

        {voiceEnabled && (
          <button
            onClick={isRecording ? stopRecording : startRecording}
            disabled={disabled}
            className={`p-2 rounded-lg transition-colors disabled:opacity-50 ${
              isRecording
                ? 'text-red-500 bg-red-50'
                : 'text-text-gray hover:text-primary-blue hover:bg-blue-gray'
            }`}
            title={isRecording ? 'Detener' : 'Grabar'}
          >
            <Mic className="w-5 h-5" />
          </button>
        )}

        <button
          onClick={handleSend}
          disabled={disabled || (!input.trim() && !selectedImage)}
          className="p-2 bg-text-dark text-white rounded-lg hover:bg-deep-blue transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          title="Enviar"
        >
          <Send className="w-5 h-5" />
        </button>
      </div>
    </div>
  )
}
