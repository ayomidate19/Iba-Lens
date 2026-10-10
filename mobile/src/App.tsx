import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import { analyse, type Analysis, type Cell } from './analyse'
import type { Band } from './lib/model'
import { Screen, Segmented, Stack, useNav } from './nav'
import { CellCanvas, CellSheet, Chevron, Chip, Group, Stat, bandClass, pct } from './ui'
import './App.css'

const base = import.meta.env.BASE_URL
const DEMO_IDS = ['LAB-0412', 'LAB-0413', 'LAB-0414', 'LAB-0415', 'LAB-0416', 'LAB-0417', 'LAB-0418', 'LAB-0419']
type Tab = 'slides' | 'scan' | 'model'

/* Scientist's sign-off per slide. In-memory; the tool never decides, the human does. */
type Verdict = 'positive' | 'negative' | 'repeat'
const VERDICT_LABEL: Record<Verdict, string> = { positive: 'Confirmed positive', negative: 'Confirmed negative', repeat: 'Repeat smear' }
const reviews = new Map<string, Verdict>()
const subs = new Set<() => void>()
let snap = 0
const setReview = (id: string, v: Verdict | null) => { if (v) reviews.set(id, v); else reviews.delete(id); snap++; subs.forEach((f) => f()) }
const useReviews = () => { useSyncExternalStore((f) => { subs.add(f); return () => { subs.delete(f) } }, () => snap); return reviews }
const MIN_CELLS = 20

/* ───────── Slide flow: queue → detail → field / flagged cells → evidence sheet ───────── */

function FieldScreen({ a }: { a: Analysis }) {
  const [sel, setSel] = useState<Cell | null>(null)
  return (
    <Screen title="Field">
      <div className="field">
        <img src={a.url} alt={`Field image of ${a.name}`} />
        <svg viewBox={`0 0 ${a.width} ${a.height}`} preserveAspectRatio="none">
          {a.cells.map((c, i) => c.band !== 'likely negative' && (
            <rect key={i} x={c.bbox[1]} y={c.bbox[0]} width={c.bbox[3] - c.bbox[1]} height={c.bbox[2] - c.bbox[0]} rx={10} className={`box ${bandClass(c.band)}`} onClick={() => setSel(c)} />
          ))}
        </svg>
      </div>
      <div className="legend"><span><i className="dot pos" />Looks infected</span><span><i className="dot unc" />Unclear</span></div>
      <p className="group-footer">Tap a boxed cell to see the evidence.</p>
      {sel && <CellSheet cell={sel} onClose={() => setSel(null)} />}
    </Screen>
  )
}

function FlaggedScreen({ a }: { a: Analysis }) {
  const [sel, setSel] = useState<Cell | null>(null)
  const pos = useMemo(() => a.cells.filter((c) => c.band === 'likely positive').sort((x, y) => y.prob - x.prob), [a])
  const unc = useMemo(() => a.cells.filter((c) => c.band === 'unclear').sort((x, y) => y.prob - x.prob), [a])
  const [which, setWhich] = useState<'pos' | 'unc'>(pos.length || !unc.length ? 'pos' : 'unc')
  const list = which === 'pos' ? pos : unc
  return (
    <Screen title="Flagged Cells">
      <Segmented value={which} onChange={setWhich} options={[{ value: 'pos', label: `Positive (${pos.length})` }, { value: 'unc', label: `Unclear (${unc.length})` }]} />
      {list.length ? (
        <div className="grid">
          {list.map((c, i) => (
            <button key={i} className={`tile ${bandClass(c.band)}`} onClick={() => setSel(c)} aria-label={`Cell, ${(c.prob * 100).toFixed(0)} percent`}>
              <CellCanvas crop={c.crop} /><span>{(c.prob * 100).toFixed(0)}%</span>
            </button>
          ))}
        </div>
      ) : <Group><div className="row empty">No cells in this group.</div></Group>}
      <p className="group-footer">Sorted by probability. Tap a cell for its heatmap.</p>
      {sel && <CellSheet cell={sel} onClose={() => setSel(null)} />}
    </Screen>
  )
}

function SlideDetail({ a }: { a: Analysis }) {
  const { push } = useNav()
  const s = a.summary
  const rv = useReviews().get(a.id)
  const few = s.cells < MIN_CELLS
  return (
    <Screen title={a.name}>
      <div className="card hero">
        <Chip band={s.band} />
        <div className="big-num">{pct(s.parasitemia_pct)}</div>
        <div className="caption">Red cells that look infected · {s.cells} cells read</div>
      </div>
      {few && <p className="notice" role="alert">Only {s.cells} cells could be read. That is too few to clear this slide. Retake the photo closer, with the smear filling the frame, or read it by eye.</p>}
      <div className="stat-row">
        <Stat value={String(s.flagged_positive)} label="Look infected" tone="pos" />
        <Stat value={String(s.flagged_unclear)} label="Unclear" tone="unc" />
        <Stat value={pct(s.parasitemia_upper_pct)} label="Worst case" />
      </div>
      <Group header="Scientist sign-off" footer={rv ? 'Saved on this phone. Tap again to undo.' : 'Iba Lens never decides. Record your reading.'}>
        <div className="signoff">
          {(['positive', 'negative', 'repeat'] as Verdict[]).map((v) => (
            <button key={v} className={`vbtn ${v}${rv === v ? ' on' : ''}`} onClick={() => setReview(a.id, rv === v ? null : v)}>{VERDICT_LABEL[v]}</button>
          ))}
        </div>
      </Group>
      <Group header="Review">
        <button className="row tap" onClick={() => push('Field', <FieldScreen a={a} />)}>
          <img className="thumb" src={a.url} alt="" />
          <div className="grow"><b>Field image</b><div className="caption">Boxes mark flagged cells</div></div><Chevron />
        </button>
        <button className="row tap" onClick={() => push('Flagged Cells', <FlaggedScreen a={a} />)}>
          <div className="grow"><b>Flagged cells</b><div className="caption">With on-device heatmaps</div></div>
          <span className="value">{s.flagged_positive + s.flagged_unclear}</span><Chevron />
        </button>
      </Group>
      <p className="group-footer">Decision support only. A lab scientist confirms every result.</p>
    </Screen>
  )
}

const order: Record<Band, number> = { 'likely positive': 0, unclear: 1, 'likely negative': 2 }
function SlidesRoot({ items }: { items: Analysis[] }) {
  const { push } = useNav()
  const rv = useReviews()
  const [f, setF] = useState<'all' | Band>('all')
  const demo = items.filter((i) => i.source === 'demo')
  const pending = DEMO_IDS.length - demo.length
  const sorted = [...demo].sort((a, b) => Number(rv.has(a.id)) - Number(rv.has(b.id)) || order[a.summary.band] - order[b.summary.band] || b.summary.max_prob - a.summary.max_prob).filter((a) => f === 'all' || a.summary.band === f)
  return (
    <Screen title="Slides">
      <Segmented value={f} onChange={setF} options={[{ value: 'all', label: 'All' }, { value: 'likely positive', label: 'Positive' }, { value: 'unclear', label: 'Unclear' }, { value: 'likely negative', label: 'Negative' }]} />
      <Group header="Triage queue" footer="Most urgent first; signed-off slides drop to the bottom. Everything runs on this phone. No image leaves the device.">
        {sorted.map((a) => (
          <button key={a.id} className="row tap" onClick={() => push(a.name, <SlideDetail a={a} />)}>
            <img className="thumb" src={a.url} alt="" />
            <div className="grow"><b>{a.name}</b><div className="caption">{a.summary.cells} cells · {a.summary.flagged_positive} look infected{rv.has(a.id) ? ` · ${VERDICT_LABEL[rv.get(a.id)!]}` : ''}</div></div>
            <Chip band={a.summary.band} /><Chevron />
          </button>
        ))}
        {pending > 0 && <div className="row empty"><span className="spinner" />Analysing {pending} slide{pending > 1 ? 's' : ''}…</div>}
        {!pending && !sorted.length && <div className="row empty">No slides in this group.</div>}
      </Group>
    </Screen>
  )
}

/* ───────── Scan flow: capture → result ───────── */

function ScanRoot({ items, onFile, busy, error }: { items: Analysis[]; onFile: (f: File) => void; busy: boolean; error: string | null }) {
  const { push } = useNav()
  const cam = useRef<HTMLInputElement>(null)
  const lib = useRef<HTMLInputElement>(null)
  const pick = (e: React.ChangeEvent<HTMLInputElement>) => { const f = e.target.files?.[0]; if (f) onFile(f); e.target.value = '' }
  const scans = items.filter((i) => i.source === 'scan')
  const steps = ['Place the slide under the microscope and focus on the thin smear.', 'Hold the phone to the eyepiece so cells fill the frame.', 'Take a photo. Analysis runs on this phone.']
  return (
    <Screen title="Scan">
      <Group header="How to scan">
        {steps.map((t, i) => (<div className="row step" key={i}><span className="num">{i + 1}</span><span>{t}</span></div>))}
      </Group>
      <div className="actions">
        <button className="primary" disabled={busy} onClick={() => cam.current?.click()}>{busy ? <><span className="spinner light" /> Analysing…</> : 'Take Photo'}</button>
        <button className="secondary" disabled={busy} onClick={() => lib.current?.click()}>Choose from Library</button>
      </div>
      {error && <p className="group-footer err" role="alert">{error}</p>}
      <input ref={cam} type="file" accept="image/*" capture="environment" hidden onChange={pick} />
      <input ref={lib} type="file" accept="image/*" hidden onChange={pick} />
      {scans.length > 0 && (
        <Group header="Recent scans">
          {scans.map((a) => (
            <button key={a.id} className="row tap" onClick={() => push(a.name, <SlideDetail a={a} />)}>
              <img className="thumb" src={a.url} alt="" />
              <div className="grow"><b>{a.name}</b><div className="caption">{a.summary.cells} cells</div></div><Chip band={a.summary.band} /><Chevron />
            </button>
          ))}
        </Group>
      )}
      <p className="group-footer">Works offline. The model and cell detector run on this device.</p>
    </Screen>
  )
}

/* ───────── Model flow ───────── */

interface Card { metrics: { test: Record<string, any> }; slides: { by_level: Record<string, any>; setup: string } }
const f1 = (x: number) => (x * 100).toFixed(1) + '%'

function AccuracyScreen({ m }: { m: Card }) {
  const t = m.metrics.test
  const [v, setV] = useState<'cells' | 'slides'>('cells')
  return (
    <Screen title="Accuracy">
      <Segmented value={v} onChange={setV} options={[{ value: 'cells', label: 'Cells' }, { value: 'slides', label: 'Simulated slides' }]} />
      {v === 'cells' ? (
        <>
          <p className="plain">Out of 100 infected cells it catches about {Math.round(t['at_0.5'].sensitivity * 100)}. Out of 100 healthy cells it wrongly flags about {Math.round((1 - t['at_0.5'].specificity) * 100)}.</p>
          <div className="stat-row two"><Stat value={t.auc.toFixed(3)} label="AUC (1.0 is perfect)" /><Stat value={t.n_cells.toLocaleString()} label={`Cells · ${t.n_patients} patients`} /></div>
          <div className="stat-row two">
            <Stat value={f1(t.at_t_low.sensitivity)} label="Catches infected (clearing)" tone="neg" /><Stat value={f1(t.at_t_low.specificity)} label="Keeps healthy (clearing)" />
            <Stat value={f1(t.at_t_high.sensitivity)} label="Catches infected (flagging)" /><Stat value={f1(t.at_t_high.specificity)} label="Keeps healthy (flagging)" tone="pos" />
          </div>
          <p className="group-footer">Measured on cells from patients never seen in training.</p>
        </>
      ) : (
        <Group header="True parasitemia → result" footer={m.slides.setup}>
          {Object.entries(m.slides.by_level).map(([lvl, s]: [string, any]) => (
            <div className="row between" key={lvl}><div><b>{lvl}</b><div className="caption">estimated {s.est_pct_mean.toFixed(1)}%</div></div><span className="value">{s.likely_positive}/{s.slides} positive</span></div>
          ))}
        </Group>
      )}
    </Screen>
  )
}

function StepsScreen() {
  const steps: [string, string][] = [
    ['Find cells', 'The photo is thresholded (Otsu), cleaned, and split into single red blood cells.'],
    ['Classify each cell', 'A 9 MB MobileNetV2 scores each 96 px cell for parasites using ONNX Runtime.'],
    ['Set the triage band', 'Calibrated thresholds turn scores into positive, unclear, or negative, for the cell and the slide.'],
    ['Show the evidence', 'Occlusion sensitivity highlights the pixels that drove a cell’s score.'],
  ]
  return (
    <Screen title="How It Works">
      <Group footer="Every step runs on this device. Nothing is uploaded.">
        {steps.map(([h, b], i) => (<div className="row step top" key={h}><span className="num">{i + 1}</span><div><b>{h}</b><div className="caption body">{b}</div></div></div>))}
      </Group>
    </Screen>
  )
}

function LimitsScreen() {
  const items = ['Research prototype, not a medical device.', 'Trained on one public thin-smear dataset; other stains and cameras may perform worse.', 'Estimates the share of infected red cells from the cells found, not a full-slide count. One field cannot clear a slide: read several, as labs already do.', 'Species and parasite stage are not identified.', 'A lab scientist confirms every result.']
  return (
    <Screen title="Limits & Safety">
      <Group>{items.map((t) => <div className="row step top" key={t}><span className="bullet" /><span>{t}</span></div>)}</Group>
    </Screen>
  )
}

function ModelRoot() {
  const { push } = useNav()
  const [m, setM] = useState<Card | null>(null)
  useEffect(() => { fetch(base + 'model/model_card.json').then((r) => r.json()).then(setM) }, [])
  const Icon = ({ c, d }: { c: string; d: string }) => (<span className="icon-sq" style={{ background: c }}><svg viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d={d} /></svg></span>)
  return (
    <Screen title="Model">
      <div className="card hero left"><b>MobileNetV2</b><div className="caption">9 MB · ONNX Runtime · runs on device</div></div>
      <Group>
        <button className="row tap" disabled={!m} onClick={() => m && push('Accuracy', <AccuracyScreen m={m} />)}><Icon c="#34c759" d="M4 19V9M10 19V5M16 19v-7M22 19H2" /><span className="grow">Accuracy</span><Chevron /></button>
        <button className="row tap" onClick={() => push('How It Works', <StepsScreen />)}><Icon c="#007aff" d="M5 12h14M13 6l6 6-6 6" /><span className="grow">How It Works</span><Chevron /></button>
        <button className="row tap" onClick={() => push('Limits & Safety', <LimitsScreen />)}><Icon c="#ff9500" d="M12 8v5M12 17h.01M10.3 4l-8 14a2 2 0 001.7 3h16a2 2 0 001.7-3l-8-14a2 2 0 00-3.4 0z" /><span className="grow">Limits &amp; Safety</span><Chevron /></button>
      </Group>
      <p className="group-footer">Trained on the public NIH thin-smear cell dataset. A lab scientist confirms every result.</p>
    </Screen>
  )
}

/* ───────── Shell ───────── */

const icons: Record<Tab, string> = {
  slides: 'M8 6h12M8 12h12M8 18h12M3.5 6h.01M3.5 12h.01M3.5 18h.01',
  scan: 'M4 8V6a2 2 0 012-2h2M16 4h2a2 2 0 012 2v2M20 16v2a2 2 0 01-2 2h-2M8 20H6a2 2 0 01-2-2v-2M12 9a3 3 0 100 6 3 3 0 000-6',
  model: 'M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3zM12 12l8-4.5M12 12v9M12 12L4 7.5',
}
const TABS: Tab[] = ['slides', 'scan', 'model']

export default function App() {
  const [tab, setTab] = useState<Tab>('slides')
  const [items, setItems] = useState<Analysis[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [pushScan, setPushScan] = useState<{ a: Analysis; n: number } | null>(null)

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
      const a = await analyse(`Scan ${n}`, `Scan ${n}`, URL.createObjectURL(f), 'scan')
      if (!a.cells.length) throw new Error('No cells found. Move closer and make sure the smear fills the frame.')
      setItems((x) => [a, ...x]); setPushScan({ a, n })
    } catch (e) { setError(e instanceof Error ? e.message : String(e)) }
    setBusy(false)
  }, [items])

  return (
    <div className="app">
      <div className="content">
        <Stack rootTitle="Slides" active={tab === 'slides'} root={<SlidesRoot items={items} />} key="s" />
        <ScanStack active={tab === 'scan'} items={items} onFile={onFile} busy={busy} error={error} pushed={pushScan} />
        <Stack rootTitle="Model" active={tab === 'model'} root={<ModelRoot />} key="m" />
      </div>
      <nav className="tabbar" aria-label="Tabs">
        {TABS.map((t) => (
          <button key={t} className={tab === t ? 'on' : ''} onClick={() => setTab(t)} aria-current={tab === t}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={tab === t ? 2.2 : 1.8} strokeLinecap="round" strokeLinejoin="round"><path d={icons[t]} /></svg>
            <span>{t[0].toUpperCase() + t.slice(1)}</span>
          </button>
        ))}
      </nav>
    </div>
  )
}

// Scan stack needs live props in its root and a way to push a result when analysis finishes.
function ScanStack({ active, items, onFile, busy, error, pushed }: { active: boolean; items: Analysis[]; onFile: (f: File) => void; busy: boolean; error: string | null; pushed: { a: Analysis; n: number } | null }) {
  return (
    <Stack rootTitle="Scan" active={active} root={<ScanRootLive items={items} onFile={onFile} busy={busy} error={error} pushed={pushed} />} />
  )
}
function ScanRootLive(p: { items: Analysis[]; onFile: (f: File) => void; busy: boolean; error: string | null; pushed: { a: Analysis; n: number } | null }) {
  const { push } = useNav()
  const last = useRef<number>(0)
  useEffect(() => {
    if (p.pushed && p.pushed.n !== last.current) { last.current = p.pushed.n; push(p.pushed.a.name, <SlideDetail a={p.pushed.a} />) }
  }, [p.pushed, push])
  return <ScanRoot items={p.items} onFile={p.onFile} busy={p.busy} error={p.error} />
}
