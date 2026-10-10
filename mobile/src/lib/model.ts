import * as ort from 'onnxruntime-web/wasm'
import mjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url'
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url'
import { SIZE } from './segment'

const base = import.meta.env.BASE_URL
ort.env.wasm.wasmPaths = { mjs: mjsUrl, wasm: wasmUrl }
ort.env.wasm.numThreads = 1 // no cross-origin isolation needed (works in WKWebView / plain hosting)

export interface Thresholds {
  t_low: number
  t_high: number
  slide_k_pos: number
  slide_k_neg: number
  cell_fp_rate: number
}

let session: Promise<ort.InferenceSession> | null = null
let thresholds: Promise<Thresholds> | null = null

export function loadModel() {
  session ??= ort.InferenceSession.create(base + 'model/iba_mobilenetv2_96.onnx', { executionProviders: ['wasm'] })
  return session
}
export function loadThresholds() {
  thresholds ??= fetch(base + 'model/thresholds.json').then((r) => r.json())
  return thresholds
}

/** crops: list of SIZE*SIZE*3 RGB uint8 arrays -> probability parasitized each */
export async function predict(crops: Uint8Array[], batch = 64): Promise<number[]> {
  const s = await loadModel()
  const out: number[] = []
  const plane = SIZE * SIZE
  for (let b = 0; b < crops.length; b += batch) {
    const part = crops.slice(b, b + batch)
    const x = new Float32Array(part.length * 3 * plane)
    part.forEach((c, n) => {
      for (let i = 0; i < plane; i++)
        for (let ch = 0; ch < 3; ch++) x[(n * 3 + ch) * plane + i] = c[i * 3 + ch] / 255
    })
    const res = await s.run({ image: new ort.Tensor('float32', x, [part.length, 3, SIZE, SIZE]) })
    out.push(...Array.from(res.prob.data as Float32Array))
    await new Promise((r) => setTimeout(r)) // yield so the UI stays responsive
  }
  return out
}

export type Band = 'likely positive' | 'unclear' | 'likely negative'
export const cellBand = (p: number, t: Thresholds): Band =>
  p >= t.t_high ? 'likely positive' : p < t.t_low ? 'likely negative' : 'unclear'

export interface Summary {
  cells: number
  flagged_positive: number
  flagged_unclear: number
  parasitemia_pct: number
  raw_flagged_pct: number
  parasitemia_upper_pct: number
  band: Band
  max_prob: number
}
export function summarise(probs: number[], t: Thresholds): Summary {
  const n = probs.length
  const pos = probs.filter((p) => p >= t.t_high).length
  const unc = probs.filter((p) => p >= t.t_low && p < t.t_high).length
  const band: Band = pos >= t.slide_k_pos ? 'likely positive' : pos + unc <= t.slide_k_neg ? 'likely negative' : 'unclear'
  return {
    cells: n,
    flagged_positive: pos,
    flagged_unclear: unc,
    parasitemia_pct: n ? Math.max(0, (100 * pos) / n - 100 * t.cell_fp_rate) : 0,
    raw_flagged_pct: n ? (100 * pos) / n : 0,
    parasitemia_upper_pct: n ? (100 * (pos + unc)) / n : 0,
    band,
    max_prob: n ? Math.max(...probs) : 0,
  }
}

/** Occlusion sensitivity: how much does hiding each patch lower the parasitized probability?
 *  Returns SIZE*SIZE map in 0..1 (on-device stand-in for Grad-CAM). */
export async function occlusion(crop: Uint8Array, base_p: number, patch = 24, stride = 8): Promise<Float32Array> {
  // fill colour = mean of non-black pixels so the patch looks like cell, not background
  let r = 0, g = 0, b = 0, n = 0
  for (let i = 0; i < SIZE * SIZE; i++) {
    if (crop[i * 3] + crop[i * 3 + 1] + crop[i * 3 + 2] > 30) {
      r += crop[i * 3]; g += crop[i * 3 + 1]; b += crop[i * 3 + 2]; n++
    }
  }
  const fill = n ? [r / n, g / n, b / n] : [0, 0, 0]
  const pos: [number, number][] = []
  for (let y = 0; y <= SIZE - patch; y += stride) for (let x = 0; x <= SIZE - patch; x += stride) pos.push([y, x])
  const variants = pos.map(([y0, x0]) => {
    const v = crop.slice()
    for (let y = y0; y < y0 + patch; y++)
      for (let x = x0; x < x0 + patch; x++) for (let c = 0; c < 3; c++) v[(y * SIZE + x) * 3 + c] = fill[c]
    return v
  })
  const probs = await predict(variants, 64)
  const map = new Float32Array(SIZE * SIZE)
  const cnt = new Float32Array(SIZE * SIZE)
  pos.forEach(([y0, x0], k) => {
    const drop = Math.max(0, base_p - probs[k])
    for (let y = y0; y < y0 + patch; y++) for (let x = x0; x < x0 + patch; x++) {
      map[y * SIZE + x] += drop
      cnt[y * SIZE + x]++
    }
  })
  for (let i = 0; i < map.length; i++) map[i] = cnt[i] ? map[i] / cnt[i] : 0
  // soften block edges: two box-blur passes
  const tmp = new Float32Array(map.length)
  for (let pass = 0; pass < 2; pass++) {
    for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
      let s = 0, n = 0
      for (let dy = -4; dy <= 4; dy++) for (let dx = -4; dx <= 4; dx++) {
        const yy = y + dy, xx = x + dx
        if (yy >= 0 && yy < SIZE && xx >= 0 && xx < SIZE) { s += map[yy * SIZE + xx]; n++ }
      }
      tmp[y * SIZE + x] = s / n
    }
    map.set(tmp)
  }
  let mx = 0
  for (let i = 0; i < map.length; i++) if (map[i] > mx) mx = map[i]
  if (mx > 0) for (let i = 0; i < map.length; i++) map[i] /= mx
  return map
}
