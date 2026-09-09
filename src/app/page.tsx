'use client'

import { useCallback, useState } from 'react'
import Chat from '@/components/Chat'
import LoadingScreen from '@/components/LoadingScreen'

export default function Home() {
  const [ready, setReady] = useState(false)
  const onDone = useCallback(() => setReady(true), [])

  return (
    <main className="min-h-screen w-full overflow-x-hidden bg-inherit">
      {!ready && <LoadingScreen onDone={onDone} />}
      <div
        className={
          ready ? 'opacity-100' : 'pointer-events-none absolute inset-0 opacity-0'
        }
      >
        <Chat />
      </div>
    </main>
  )
}
