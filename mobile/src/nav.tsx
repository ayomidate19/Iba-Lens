// iOS-style navigation: a push/pop stack per tab, large titles that collapse to an inline title,
// back button labelled with the previous screen, edge swipe-back, and spring-like transitions.
import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react'

interface Entry { id: number; title: string; node: ReactNode; phase: 'in0' | 'idle' | 'out' }
const NavCtx = createContext<{ push: (title: string, node: ReactNode) => void; pop: () => void }>({ push() {}, pop() {} })
const LayerCtx = createContext<{ back?: string }>({})
export const useNav = () => useContext(NavCtx)

const DURATION = 380

export function Stack({ rootTitle, root, active }: { rootTitle: string; root: ReactNode; active: boolean }) {
  const [entries, setEntries] = useState<Entry[]>([{ id: 0, title: rootTitle, node: null, phase: 'idle' }])
  const idRef = useRef(1)
  const ref = useRef<HTMLDivElement>(null)
  const entriesRef = useRef(entries)
  entriesRef.current = entries

  const push = useCallback((title: string, node: ReactNode) => {
    const id = idRef.current++
    setEntries((e) => [...e, { id, title, node, phase: 'in0' }])
    requestAnimationFrame(() => requestAnimationFrame(() => setEntries((e) => e.map((x) => (x.id === id ? { ...x, phase: 'idle' } : x)))))
  }, [])

  const pop = useCallback(() => {
    const live = entriesRef.current.filter((x) => x.phase !== 'out')
    if (live.length < 2) return
    const top = live[live.length - 1].id
    setEntries((e) => e.map((x) => (x.id === top ? { ...x, phase: 'out' } : x)))
    setTimeout(() => setEntries((e) => e.filter((x) => x.phase !== 'out')), DURATION)
  }, [])

  // Edge swipe-back
  useEffect(() => {
    const el = ref.current!
    let startX = -1, startY = 0, dx = 0, t0 = 0, dragging = false
    const layers = () => Array.from(el.querySelectorAll<HTMLElement>(':scope > .layer'))
    const topTwo = () => {
      const l = layers().filter((n) => n.dataset.phase !== 'out')
      return l.length >= 2 ? [l[l.length - 1], l[l.length - 2]] : null
    }
    const start = (e: TouchEvent) => {
      const x = e.touches[0].clientX - el.getBoundingClientRect().left
      if (x < 24 && topTwo()) { startX = e.touches[0].clientX; startY = e.touches[0].clientY; dx = 0; t0 = Date.now(); dragging = false }
      else startX = -1
    }
    const move = (e: TouchEvent) => {
      if (startX < 0) return
      dx = e.touches[0].clientX - startX
      if (!dragging && Math.abs(e.touches[0].clientY - startY) > Math.abs(dx) && Math.abs(dx) < 10) { startX = -1; return }
      if (dx > 6) dragging = true
      if (!dragging) return
      e.preventDefault()
      const t = topTwo(); if (!t) return
      const w = el.clientWidth, d = Math.max(0, dx)
      t[0].style.transition = t[1].style.transition = 'none'
      t[0].style.transform = `translateX(${d}px)`
      t[1].style.transform = `translateX(${-0.28 * w + 0.28 * d}px)`
    }
    const end = () => {
      if (startX < 0) return
      const t = topTwo()
      if (t) { for (const n of t) { n.style.transition = ''; n.style.transform = '' } }
      const v = dx / Math.max(1, Date.now() - t0)
      if (dragging && (dx > el.clientWidth * 0.35 || v > 0.5)) pop()
      startX = -1; dragging = false
    }
    el.addEventListener('touchstart', start, { passive: true })
    el.addEventListener('touchmove', move, { passive: false })
    el.addEventListener('touchend', end)
    el.addEventListener('touchcancel', end)
    return () => { el.removeEventListener('touchstart', start); el.removeEventListener('touchmove', move); el.removeEventListener('touchend', end); el.removeEventListener('touchcancel', end) }
  }, [pop])

  const topIdx = entries.reduce((a, x, i) => (x.phase !== 'out' ? i : a), 0)
  return (
    <NavCtx.Provider value={{ push, pop }}>
      <div ref={ref} className="stack" style={{ display: active ? undefined : 'none' }}>
        {entries.map((e, i) => {
          const prev = i > 0 ? entries[i - 1].title : undefined
          return (
            <div key={e.id} className={`layer${i < topIdx ? ' covered' : ''}${i < topIdx - 1 ? ' deep' : ''}`} data-phase={e.phase}>
              <LayerCtx.Provider value={{ back: prev && (prev.length <= 14 ? prev : 'Back') }}>{e.id === 0 ? root : e.node}</LayerCtx.Provider>
            </div>
          )
        })}
      </div>
    </NavCtx.Provider>
  )
}

export function Screen({ title, trailing, children }: { title: string; trailing?: ReactNode; children: ReactNode }) {
  const { back } = useContext(LayerCtx)
  const { pop } = useNav()
  const [collapsed, setCollapsed] = useState(false)
  return (
    <section className="screen">
      <header className={`navbar${collapsed ? ' collapsed' : ''}`}>
        <div className="nav-side left">
          {back && (
            <button className="back" onClick={pop} aria-label={`Back to ${back}`}>
              <svg width="12" height="20" viewBox="0 0 12 20" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round"><path d="M10 2 2 10l8 8" /></svg>
              {back}
            </button>
          )}
        </div>
        <div className="nav-title">{title}</div>
        <div className="nav-side right">{trailing}</div>
      </header>
      <div className="scroll" onScroll={(e) => { const c = e.currentTarget.scrollTop > 34; if (c !== collapsed) setCollapsed(c) }}>
        <h1 className="large-title">{title}</h1>
        {children}
        <div className="spacer" />
      </div>
    </section>
  )
}

export function Segmented<T extends string>({ value, options, onChange }: { value: T; options: { value: T; label: string }[]; onChange: (v: T) => void }) {
  const i = options.findIndex((o) => o.value === value)
  return (
    <div className="segmented" role="tablist">
      <div className="thumb-sel" style={{ width: `calc((100% - 4px) / ${options.length})`, transform: `translateX(${i * 100}%)` }} />
      {options.map((o) => (
        <button key={o.value} role="tab" aria-selected={o.value === value} className={o.value === value ? 'on' : ''} onClick={() => onChange(o.value)}>{o.label}</button>
      ))}
    </div>
  )
}
