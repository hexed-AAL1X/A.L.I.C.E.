interface MessageProps {
  message: {
    role: 'user' | 'assistant'
    content: string
    image?: string
  }
}

export default function Message({ message }: MessageProps) {
  const isUser = message.role === 'user'

  return (
    <div className={`mb-8 ${isUser ? 'text-right' : 'text-left'}`}>
      {/* Etiqueta */}
      <div className="mb-2">
        <span className="text-xs font-medium text-text-gray uppercase tracking-wide">
          {isUser ? 'Tú' : 'A.L.I.C.E.'}
        </span>
      </div>

      {/* Contenido */}
      <div className={`inline-block max-w-[85%]`}>
        {message.image && (
          <img
            src={message.image}
            alt="Imagen"
            className="rounded-xl mb-3 max-w-md border border-gray-100"
          />
        )}
        <div
          className={`px-6 py-4 rounded-xl ${
            isUser
              ? 'bg-text-dark text-white'
              : 'bg-blue-gray text-text-dark'
          }`}
        >
          <p className="text-[15px] leading-relaxed whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    </div>
  )
}
