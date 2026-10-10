import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { analyse, type Analysis, type Cell } from './analyse'
import { occlusion } from './lib/model'
import { SIZE } from './lib/segment'
import type { Band } from './lib/model'
import './App.css'

const base = import.meta.env.BASE_URL
const DEMO_IDS = ['LAB-0412', 'LAB-0413', 'LAB-0414', 'LAB-0415', 'LAB-0416', 'LAB-0417', 'LAB-0418', 'LAB-0419']
type Tab = 'slides' | 'scan' | 'model'

const bandClass = (b: Band) => (b === 'likely positive' ? 'pos' : b === 'unclear' ? 'unc' : 'neg')
const Chip = ({ band }: { band: Band }) => <span className={`chip ${bandClass(band)}`}>{band.replace('likely ', '')}</span>
const pct = (n: number) => (n < 10 ? n.toFixed(1) : n.toFixed(0)) + '%'

function CellCanvas({ crop, heat, className }: { crop: Uint8Array; heat?: Float32Array | null; className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const ctx = ref.current!.getContext('2d')!
    const im = ctx.createImageData(SIZE, SIZE)
    for (let i = 0; i < SIZE * SIZE; i++) {
      let r = crop[i * 3], g = crop[i * 3 + 1], b = crop[i * 3 + 2]
      const fg = r + g + b > 30
      if (!fg) { r = g = b = 28 }
      if (heat && fg) {
        const h = heat[i]
        const a = Math.min(0.85, h * 0.9)
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

function Sheet({ cell, onClose }: { cell: Cell; onClose: () => void }) {
  const [heat, setHeat] = useState<Float32Array | null>(null)
  const [showHeat, setShowHeat] = useState(true)
  useEffect(() => {
    let live = true
    setHeat(null)
    occlusion(cell.crop, cell.prob).then((h) => live && setHeat(h))
    return () => { live = false }
  }, [cell])
  return (
    <div className="scrim" onClick={onClose}>
      <div className="sheet" onClick={(e) => e.stopPropagation()}>
        <div className="grab" />
        <div className="sheet-head">
          <h3>Cell evidence</h3>
          <button className="link" onClick={onClose}>Done</button>
        </div>
        <CellCanvas crop={cell.crop} heat={showHeat ? heat : null} className="big-cell" />
        <div className="center"><Chip band={cell.band} /></div>
        <div className="group">
          <div className="row"><span>Parasite probability</span><b>{(cell.prob * 100).toFixed(1)}%</b></div>
          <div className="row">
            <span>Show attention heatmap</span>
            <label className="switch"><input type="checkbox" checked={showHeat} onChange={(e) => setShowHeat(e.target.checked)} /><i /></label>
          </div>
        </div>
        <p className="foot">{heat ? 'Red areas are where hiding the pixels most lowers the score — computed on this phone by occlusion.' : 'Computing heatmap on device…'}</p>
      </div>
    </div>
  )
}

function Detail({ a, onBack }: { a: Analysis; onBack: () => void }) {
  const [sel, setSel] = useState<Cell | null>(null)
  const flagged = useMemo(() => a.cells.filter((c) => c.band !== 'likely negative').sort((x, y) => y.prob - x.prob), [a])
  const s = a.summary
  return (
    <div className="page">
      <header className="navbar"><button className="back" onClick={onBack}>‹ Slides</button><span className="nav-title">{a.name}</span></header>
      <h1 className="large">{a.name}</h1>
      <div className="hero">
        <Chip band={s.band} />
        <div className="big-num">{pct(s.parasitemia_pct)}</div>
        <div className="muted">estimated parasitemia · {s.cells} cells scanned</div>
      </div>
      <div className="group">
        <div className="row"><span>Likely positive cells</span><b>{s.flagged_positive}</b></div>
        <div className="row"><span>Unclear cells</span><b>{s.flagged_unclear}</b></div>
        <div className="row"><span>Upper estimate</span><b>{pct(s.parasitemia_upper_pct)}</b></div>
      </div>
      <p className="section">Field</p>
      <div className="field">
        <img src={a.url} alt={a.name} />
        <svg viewBox={`0 0 ${a.width} ${a.height}`} preserveAspectRatio="none">
          {a.cells.map((c, i) => c.band !== 'likely negative' && (
            <rect key={i} x={c.bbox[1]} y={c.bbox[0]} width={c.bbox[3] - c.bbox[1]} height={c.bbox[2] - c.bbox[0]} rx={10}
              className={`box ${bandClass(c.band)}`} onClick={() => setSel(c)} />
          ))}
        </svg>
      </div>
      <p className="section">Flagged cells · tap for evidence</p>
      {flagged.length ? (
        <div className="grid">
          {flagged.map((c, i) => (
            <button key={i} className={`tile ${bandClass(c.band)}`} onClick={() => setSel(c)}>
              <CellCanvas crop={c.crop} /><span>{(c.prob * 100).toFixed(0)}%</span>
            </button>
          ))}
        </div>
      ) : <div className="group"><div className="row muted">No cells flagged.</div></div>}
      <p className="foot">Decision support only. A lab scientist confirms every result.</p>
      {sel && <Sheet cell={sel} onClose={() => setSel(null)} />}
    </div>
  )
}

function SlidesTab({ items, open }: { items: Analysis[]; open: (a: Analysis) => void }) {
  const order = { 'likely positive': 0, unclear: 1, 'likely negative': 2 }
  const sorted = [...items].sort((a, b) => order[a.summary.band] - order[b.summary.band] || b.summary.max_prob - a.summary.max_prob)
  const pending = DEMO_IDS.length - items.filter((i) => i.source === 'demo').length
  return (
    <div className="page">
      <h1 className="large">Slides</h1>
      <p className="sub">Triage queue · most urgent first</p>
      <div className="group">
        {sorted.map((a) => (
          <button key={a.id} className="row tap" onClick={() => open(a)}>
            <img className="thumb" src={a.url} alt="" />
            <div className="grow"><b>{a.name}</b><div className="muted">{a.summary.cells} cells · {a.summary.flagged_positive} positive · {a.summary.flagged_unclear} unclear</div></div>
            <Chip band={a.summary.band} /><span className="chev">›</span>
          </button>
        ))}
        {pending > 0 && <div className="row muted"><span className="spinner" />Analysing {pending} slide{pending > 1 ? 's' : ''} on device…</div>}
      </div>
      <p className="foot">Everything runs on this phone. No image leaves the device.</p>
    </div>
  )
}

function ScanTab({ onFile, busy, error }: { onFile: (f: File) => void; busy: boolean; error: string | null }) {
  const cam = useRef<HTMLInputElement>(null)
  const lib = useRef<HTMLInputElement>(null)
  const pick = (e: React.ChangeEvent<HTMLInputElement>) => { const f = e.target.files?.[0]; if (f) onFile(f); e.target.value = '' }
  return (
    <div className="page">
      <h1 className="large">Scan</h1>
      <p className="sub">Photograph a blood smear through the microscope eyepiece.</p>
      <div className="viewfinder"><div className="reticle" /></div>
      <button className="primary" disabled={busy} onClick={() => cam.current?.click()}>{busy ? 'Analysing…' : 'Scan slide'}</button>
      <button className="secondary" disabled={busy} onClick={() => lib.current?.click()}>Choose from Photos</button>
      <input ref={cam} type="file" accept="image/*" capture="environment" hidden onChange={pick} />
      <input ref={lib} type="file" accept="image/*" hidden onChange={pick} />
      {error && <p className="foot err">{error}</p>}
      <p className="foot">Works offline. The model and cell detector run on this device.</p>
    </div>
  )
}

interface Card { metrics: { val_auc: number; test: Record<string, any> }; slides: { by_level: Record<string, any> } }
function ModelTab() {
  const [m, setM] = useState<Card | null>(null)
  useEffect(() => { fetch(base + 'model/model_card.json').then((r) => r.json()).then(setM) }, [])
  const t = m?.metrics.test
  const f = (x: number) => (x * 100).toFixed(1) + '%'
  return (
    <div className="page">
      <h1 className="large">Model</h1>
      <p className="sub">MobileNetV2 · 9 MB · ONNX Runtime on device</p>
      {t && <>
        <p className="section">Held-out patients · {t.n_cells.toLocaleString()} cells · {t.n_patients} patients</p>
        <div className="group">
          <div className="row"><span>AUC</span><b>{t.auc.toFixed(3)}</b></div>
          <div className="row"><span>Rule-out sensitivity</span><b>{f(t.at_t_low.sensitivity)}</b></div>
          <div className="row"><span>Rule-out specificity</span><b>{f(t.at_t_low.specificity)}</b></div>
          <div className="row"><span>Rule-in sensitivity</span><b>{f(t.at_t_high.sensitivity)}</b></div>
          <div className="row"><span>Rule-in specificity</span><b>{f(t.at_t_high.specificity)}</b></div>
        </div>
        <p className="section">Simulated 100-cell fields</p>
        <div className="group">
          {Object.entries(m!.slides.by_level).map(([lvl, v]: [string, any]) => (
            <div className="row" key={lvl}><span>{lvl} true parasitemia</span><b>{v.est_pct_mean.toFixed(1)}% est · {v.likely_positive}/{v.slides} positive</b></div>
          ))}
        </div>
      </>}
      <p className="foot">Trained on the public NIH thin-smear cell dataset. Research prototype, not a medical device. A lab scientist confirms every result.</p>
    </div>
  )
}

const icons: Record<Tab, string> = {
  slides: 'M4 6h16M4 12h16M4 18h16',
  scan: 'M4 8V5h3M17 5h3v3M20 16v3h-3M7 19H4v-3M12 9a3 3 0 100 6 3 3 0 000-6',
  model: 'M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3zM12 12l8-4.5M12 12v9M12 12L4 7.5',
}

export default function App() {
  const [tab, setTab] = useState<Tab>('slides')
  const [items, setItems] = useState<Analysis[]>([])
  const [open, setOpen] = useState<Analysis | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let live = true
    ;(async () => {
      for (const id of DEMO_IDS) {
        if (!live) return
        try {
          const a = await analyse(id, id, `${base}demo/${id}.png`, 'demo')
          setItems((x) => (x.some((y) => y.id === id) ? x : [...x, a]))
        } catch (e) { console.error(e) }
      }
    })()
    return () => { live = false }
  }, [])

  const onFile = useCallback(async (f: File) => {
    setBusy(true); setError(null)
    try {
      const n = items.filter((i) => i.source === 'scan').length + 1
      const a = await analyse(`SCAN-${n}`, `Scan ${n}`, URL.createObjectURL(f), 'scan')
      if (!a.cells.length) throw new Error('No cells found. Move closer and make sure the smear fills the frame.')
      setItems((x) => [a, ...x]); setOpen(a)
    } catch (e) { setError(e instanceof Error ? e.message : String(e)) }
    setBusy(false)
  }, [items])

  return (
    <div className="app">
      <main>
        {open ? <Detail a={open} onBack={() => setOpen(null)} />
          : tab === 'slides' ? <SlidesTab items={items} open={setOpen} />
          : tab === 'scan' ? <ScanTab onFile={onFile} busy={busy} error={error} />
          : <ModelTab />}
      </main>
      {!open && (
        <nav className="tabbar">
          {(['slides', 'scan', 'model'] as Tab[]).map((t) => (
            <button key={t} className={tab === t ? 'on' : ''} onClick={() => setTab(t)}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d={icons[t]} /></svg>
              <span>{t[0].toUpperCase() + t.slice(1)}</span>
            </button>
          ))}
        </nav>
      )}
    </div>
  )
}
