interface MessageProps {
  message: {
    role: 'user' | 'assistant'
    content: string
    image?: string
  }
  dark?: boolean
}

export default function Message({ message, dark = false }: MessageProps) {
  const isUser = message.role === 'user'

  return (
    <div className={`mb-7 ${isUser ? 'text-right' : 'text-left'}`}>
      <div className="mb-1.5">
        <span
          className={`text-[11px] font-medium uppercase tracking-[0.14em] ${
            dark ? 'text-slate-400' : 'text-text-gray'
          }`}
        >
          {isUser ? 'Tú' : 'A.L.I.C.E.'}
        </span>
      </div>

      <div className="inline-block max-w-[min(92%,42rem)] text-left">
        {message.image && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={message.image}
            alt="Adjunto"
            className="mb-3 max-h-64 rounded-2xl border border-black/5 object-cover"
          />
        )}
        <div
          className={`px-5 py-3.5 text-[15px] leading-relaxed whitespace-pre-wrap ${
            isUser
              ? 'rounded-2xl rounded-tr-md bg-[var(--alice-ink)] text-white'
              : dark
                ? 'rounded-2xl rounded-tl-md bg-[#1a1a1d] text-[#f2f2f3] ring-1 ring-white/10'
                : 'rounded-2xl rounded-tl-md bg-white/90 text-[#14192c] ring-1 ring-black/[0.04] shadow-[0_10px_30px_rgba(20,25,44,0.04)]'
          }`}
        >
          {message.content || (isUser ? '' : '…')}
        </div>
      </div>
    </div>
  )
}
