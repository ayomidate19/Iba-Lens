// Dev check: compare on-device pipeline against the Python dump (public/demo/reference.json).
import { analyse } from './analyse'

export async function parity() {
  const ref = await (await fetch(import.meta.env.BASE_URL + 'demo/reference.json')).json()
  const rows = []
  for (const id of Object.keys(ref)) {
    const a = await analyse(id, id, `${import.meta.env.BASE_URL}demo/${id}.png`, 'demo')
    const r = ref[id]
    let maxDiff = 0, boxMismatch = 0
    r.cells.forEach((c: { bbox: number[]; prob: number }, i: number) => {
      const m = a.cells[i]
      if (!m || m.bbox.some((v, k) => v !== c.bbox[k])) boxMismatch++
      else maxDiff = Math.max(maxDiff, Math.abs(m.prob - c.prob))
    })
    rows.push({ id, py: r.cells.length, js: a.cells.length, boxMismatch, maxProbDiff: +maxDiff.toFixed(5),
      pyBand: r.summary.band, jsBand: a.summary.band, pyPos: r.summary.flagged_positive, jsPos: a.summary.flagged_positive })
  }
  return rows
}
