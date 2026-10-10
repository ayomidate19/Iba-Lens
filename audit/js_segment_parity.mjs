// Runs the mobile TS segment() in node (type-stripping) on demo slides, compares to mobile reference.json (Python)
import fs from 'node:fs'
import { segment } from '../mobile/src/lib/segment.ts'
const ref = JSON.parse(fs.readFileSync('mobile/public/demo/reference.json'))
const out = {}
for (const id of Object.keys(ref)) {
  const [W, H] = ref[id].size
  const buf = fs.readFileSync(`audit/scratch/${id}.rgba`)
  const t0 = performance.now()
  const cells = segment({ width: W, height: H, data: new Uint8ClampedArray(buf) })
  const ms = performance.now() - t0
  const py = ref[id].cells
  let mism = 0, maxd = 0
  cells.forEach((c, i) => { const p = py[i]; if (!p) { mism++; return } const d = Math.max(...c.bbox.map((v, k) => Math.abs(v - p.bbox[k]))); maxd = Math.max(maxd, d); if (d) mism++ })
  out[id] = { js: cells.length, py: py.length, bboxMismatch: mism, maxBboxDiff: maxd, ms: Math.round(ms) }
  fs.writeFileSync(`audit/scratch/${id}.jscrops.bin`, Buffer.concat(cells.map((c) => Buffer.from(c.crop))))
}
console.log(JSON.stringify(out, null, 1))
