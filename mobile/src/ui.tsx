import { createPortal } from 'react-dom'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { occlusion, type Band } from './lib/model'
import { SIZE } from './lib/segment'
import type { Cell } from './analyse'
import { Segmented } from './nav'

export const bandClass = (b: Band) => (b === 'likely positive' ? 'pos' : b === 'unclear' ? 'unc' : 'neg')
export const bandLabel = (b: Band) => b.replace('likely ', '')
export const Chip = ({ band }: { band: Band }) => <span className={`chip ${bandClass(band)}`}>{bandLabel(band)}</span>
export const pct = (n: number) => (n < 10 ? n.toFixed(1) : n.toFixed(0)) + '%'

export const Chevron = () => <svg className="chev" width="8" height="14" viewBox="0 0 8 14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m1 1 6 6-6 6" /></svg>

export const Group = ({ header, footer, children }: { header?: string; footer?: ReactNode; children: ReactNode }) => (
  <>
    {header && <h2 className="group-header">{header}</h2>}
    <div className="group">{children}</div>
    {footer && <p className="group-footer">{footer}</p>}
  </>
)

export const Stat = ({ value, label, tone }: { value: string; label: string; tone?: 'pos' | 'unc' | 'neg' }) => (
  <div className={`stat${tone ? ' ' + tone : ''}`}><b>{value}</b><span>{label}</span></div>
)

export function CellCanvas({ crop, heat, className }: { crop: Uint8Array; heat?: Float32Array | null; className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const ctx = ref.current!.getContext('2d')!
    const im = ctx.createImageData(SIZE, SIZE)
    for (let i = 0; i < SIZE * SIZE; i++) {
      let r = crop[i * 3], g = crop[i * 3 + 1], b = crop[i * 3 + 2]
      const fg = r + g + b > 30
      if (!fg) r = g = b = 28
      if (heat && fg) {
        const h = heat[i], a = Math.min(0.85, h * 0.9)
        r = r * (1 - a) + 255 * a
        g = g * (1 - a) + (69 + 120 * (1 - h)) * a
        b = b * (1 - a) + 58 * a
      }
      im.data.set([r, g, b, 255], i * 4)
    }
    ctx.putImageData(im, 0, 0)
  }, [crop, heat])
  return <canvas ref={ref} width={SIZE} height={SIZE} className={className} />
}

/** Bottom sheet with grabber and two detents (medium / large). Drag the top area to resize or dismiss. */
export function Sheet({ title, onClose, children }: { title: string; onClose: () => void; children: ReactNode }) {
  const [large, setLarge] = useState(false)
  const [drag, setDrag] = useState(0)
  const [closing, setClosing] = useState(false)
  const y0 = useRef<number | null>(null)
  const close = () => { setClosing(true); setTimeout(onClose, 260) }
  const down = (e: React.PointerEvent) => { y0.current = e.clientY; (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId) }
  const move = (e: React.PointerEvent) => { if (y0.current !== null) setDrag(e.clientY - y0.current) }
  const up = () => {
    if (y0.current === null) return
    if (drag > 110) { if (large) setLarge(false); else close() }
    else if (drag < -60) setLarge(true)
    y0.current = null; setDrag(0)
  }
  return createPortal(
    <div className={`scrim${closing ? ' out' : ''}`} onClick={close}>
      <div className={`sheet${large ? ' large' : ''}`} style={{ transform: drag ? `translateY(${Math.max(drag, -80)}px)` : undefined, transition: drag ? 'none' : undefined }} onClick={(e) => e.stopPropagation()}>
        <div className="sheet-top" onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up}>
          <div className="grab" />
          <div className="sheet-head"><h3>{title}</h3><button className="done" onClick={close}>Done</button></div>
        </div>
        <div className="sheet-body">{children}</div>
      </div>
    </div>,
    document.body,
  )
}

export function CellSheet({ cell, onClose }: { cell: Cell; onClose: () => void }) {
  const [heat, setHeat] = useState<Float32Array | null>(null)
  const [view, setView] = useState<'cell' | 'heat'>('heat')
  useEffect(() => {
    let live = true
    occlusion(cell.crop, cell.prob).then((h) => live && setHeat(h))
    return () => { live = false }
  }, [cell])
  return (
    <Sheet title="Cell evidence" onClose={onClose}>
      <div className="evidence">
        <CellCanvas crop={cell.crop} heat={view === 'heat' ? heat : null} className="big-cell" />
        {view === 'heat' && !heat && <div className="computing"><span className="spinner" /> Computing on device…</div>}
      </div>
      <Segmented value={view} onChange={setView} options={[{ value: 'heat', label: 'Heatmap' }, { value: 'cell', label: 'Original' }]} />
      <div className="stat-row two">
        <Stat value={`${(cell.prob * 100).toFixed(1)}%`} label="Parasite probability" tone={bandClass(cell.band)} />
        <div className="stat"><b><Chip band={cell.band} /></b><span>Triage band</span></div>
      </div>
      <p className="group-footer">Red areas are where hiding pixels lowers the score most. Computed on this phone by occlusion. A lab scientist confirms every result.</p>
    </Sheet>
  )
}
