'use client'

import { useRef, useEffect } from 'react'
import { X, Camera } from 'lucide-react'

interface CameraModalProps {
  onClose: () => void
  onCapture: (imageData: string) => void
}

export default function CameraModal({ onClose, onCapture }: CameraModalProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)

  useEffect(() => {
    startCamera()
    return () => stopCamera()
  }, [])

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user' },
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
    } catch (error) {
      console.error('Error accessing camera:', error)
      alert('No se pudo acceder a la cámara')
      onClose()
    }
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
    }
  }

  const handleCapture = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current
      const canvas = canvasRef.current
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      const ctx = canvas.getContext('2d')
      if (ctx) {
        ctx.drawImage(video, 0, 0)
        const imageData = canvas.toDataURL('image/jpeg')
        onCapture(imageData)
        stopCamera()
      }
    }
  }

  return (
    <div
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
      onClick={() => {
        stopCamera()
        onClose()
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="bg-white rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-hidden"
      >
        <div className="flex items-center justify-between p-6 border-b border-gray-100">
          <h3 className="text-xl font-semibold text-text-dark">Capturar foto</h3>
          <button
            onClick={() => {
              stopCamera()
              onClose()
            }}
            className="w-10 h-10 rounded-lg hover:bg-gray-100 transition-colors flex items-center justify-center text-text-gray hover:text-text-dark"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="w-full rounded-xl"
          />
          <canvas ref={canvasRef} style={{ display: 'none' }} />
        </div>

        <div className="flex justify-end gap-3 p-6 border-t border-gray-100">
          <button
            onClick={() => {
              stopCamera()
              onClose()
            }}
            className="px-6 py-2.5 rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors font-medium text-text-dark"
          >
            Cancelar
          </button>
          <button
            onClick={handleCapture}
            className="px-6 py-2.5 rounded-lg bg-text-dark text-white hover:bg-deep-blue transition-colors flex items-center gap-2 font-medium"
          >
            <Camera className="w-4 h-4" />
            Capturar
          </button>
        </div>
      </div>
    </div>
  )
}
