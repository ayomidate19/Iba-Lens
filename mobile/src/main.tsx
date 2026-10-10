import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

if (new URLSearchParams(location.search).has('parity'))
  import('./parity').then((m) => m.parity().then((r) => ((window as unknown as { __parity: unknown }).__parity = r)))

if (import.meta.env.PROD && 'serviceWorker' in navigator && location.protocol.startsWith('http'))
  navigator.serviceWorker.register(import.meta.env.BASE_URL + 'sw.js').catch(() => {})
