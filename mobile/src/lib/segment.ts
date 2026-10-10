// Port of segment() in src/smearscan.py: Otsu threshold -> opening -> hole fill -> connected
// components -> square 96x96 crops with the background blacked out (same as the training data).
export const SIZE = 96

export type Box = [number, number, number, number] // row0, col0, row1, col1
export interface Segmented {
  bbox: Box
  crop: Uint8Array // SIZE*SIZE*3 RGB
}

function otsu(values: Float64Array): number {
  const n = values.length
  let lo = Infinity
  let hi = -Infinity
  for (let i = 0; i < n; i++) {
    if (values[i] < lo) lo = values[i]
    if (values[i] > hi) hi = values[i]
  }
  const nb = 256
  const counts = new Float64Array(nb)
  const w = (hi - lo) / nb
  for (let i = 0; i < n; i++) counts[Math.min(nb - 1, Math.floor((values[i] - lo) / w))]++
  const centers = Float64Array.from({ length: nb }, (_, i) => lo + w * (i + 0.5))
  const w1 = new Float64Array(nb)
  const w2 = new Float64Array(nb)
  const s1 = new Float64Array(nb)
  const s2 = new Float64Array(nb)
  let a = 0
  let b = 0
  for (let i = 0; i < nb; i++) {
    a += counts[i]
    b += counts[i] * centers[i]
    w1[i] = a
    s1[i] = b
  }
  a = 0
  b = 0
  for (let i = nb - 1; i >= 0; i--) {
    a += counts[i]
    b += counts[i] * centers[i]
    w2[i] = a
    s2[i] = b
  }
  let best = -1
  let bestI = 0
  for (let i = 0; i < nb - 1; i++) {
    const m1 = s1[i] / w1[i]
    const m2 = s2[i + 1] / w2[i + 1]
    const v = w1[i] * w2[i + 1] * (m1 - m2) ** 2
    if (v > best) {
      best = v
      bestI = i
    }
  }
  return centers[bestI]
}

// disk(2) structuring element offsets
const DISK: [number, number][] = []
for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) if (dx * dx + dy * dy <= 4) DISK.push([dy, dx])

function morph(src: Uint8Array, W: number, H: number, erode: boolean): Uint8Array {
  const out = new Uint8Array(W * H)
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      let v = erode ? 1 : 0
      for (const [dy, dx] of DISK) {
        const yy = y + dy
        const xx = x + dx
        const s = yy < 0 || yy >= H || xx < 0 || xx >= W ? 0 : src[yy * W + xx]
        if (erode ? !s : s) {
          v = erode ? 0 : 1
          break
        }
      }
      out[y * W + x] = v
    }
  return out
}

function fillHoles(mask: Uint8Array, W: number, H: number): Uint8Array {
  const bg = new Uint8Array(W * H) // background reachable from the border (4-connected)
  const stack: number[] = []
  const push = (i: number) => {
    if (!mask[i] && !bg[i]) {
      bg[i] = 1
      stack.push(i)
    }
  }
  for (let x = 0; x < W; x++) {
    push(x)
    push((H - 1) * W + x)
  }
  for (let y = 0; y < H; y++) {
    push(y * W)
    push(y * W + W - 1)
  }
  while (stack.length) {
    const i = stack.pop()!
    const x = i % W
    const y = (i - x) / W
    if (x > 0) push(i - 1)
    if (x < W - 1) push(i + 1)
    if (y > 0) push(i - W)
    if (y < H - 1) push(i + W)
  }
  const out = new Uint8Array(W * H)
  for (let i = 0; i < out.length; i++) out[i] = bg[i] ? 0 : 1
  return out
}

// PIL-style bilinear resize (triangle filter, support scales when shrinking), uint8 per pass.
function coeffs(inSize: number, outSize: number) {
  const scale = inSize / outSize
  const fscale = Math.max(1, scale)
  const support = fscale
  const res: { min: number; w: Float64Array }[] = []
  for (let x = 0; x < outSize; x++) {
    const center = (x + 0.5) * scale
    const min = Math.max(0, Math.trunc(center - support + 0.5))
    const max = Math.min(inSize, Math.trunc(center + support + 0.5))
    const w = new Float64Array(max - min)
    let sum = 0
    for (let k = 0; k < w.length; k++) {
      const t = Math.abs((k + min - center + 0.5) / fscale)
      w[k] = t < 1 ? 1 - t : 0
      sum += w[k]
    }
    if (sum) for (let k = 0; k < w.length; k++) w[k] /= sum
    res.push({ min, w })
  }
  return res
}

export function resizeBilinear(src: Uint8Array, sw: number, sh: number, dw: number, dh: number): Uint8Array {
  const cx = coeffs(sw, dw)
  const cy = coeffs(sh, dh)
  const tmp = new Uint8Array(dw * sh * 3)
  for (let y = 0; y < sh; y++)
    for (let x = 0; x < dw; x++) {
      const { min, w } = cx[x]
      for (let c = 0; c < 3; c++) {
        let s = 0
        for (let k = 0; k < w.length; k++) s += w[k] * src[(y * sw + min + k) * 3 + c]
        tmp[(y * dw + x) * 3 + c] = Math.max(0, Math.min(255, Math.round(s)))
      }
    }
  const out = new Uint8Array(dw * dh * 3)
  for (let y = 0; y < dh; y++) {
    const { min, w } = cy[y]
    for (let x = 0; x < dw; x++)
      for (let c = 0; c < 3; c++) {
        let s = 0
        for (let k = 0; k < w.length; k++) s += w[k] * tmp[((min + k) * dw + x) * 3 + c]
        out[(y * dw + x) * 3 + c] = Math.max(0, Math.min(255, Math.round(s)))
      }
  }
  return out
}

export function segment(img: ImageData, minArea = 900, maxArea = 40000): Segmented[] {
  const { width: W, height: H, data } = img
  const gray = new Float64Array(W * H)
  const valid: number[] = []
  for (let i = 0; i < W * H; i++) {
    const g = (data[i * 4] + data[i * 4 + 1] + data[i * 4 + 2]) / 3
    gray[i] = g
    if (g > 10) valid.push(g)
  }
  const t = valid.length ? otsu(Float64Array.from(valid)) : 128
  const raw = new Uint8Array(W * H)
  for (let i = 0; i < W * H; i++) raw[i] = gray[i] < t && gray[i] > 10 ? 1 : 0
  const mask = fillHoles(morph(morph(raw, W, H, true), W, H, false), W, H)

  // 8-connected labelling
  const labels = new Int32Array(W * H)
  const regions: { id: number; area: number; r0: number; c0: number; r1: number; c1: number }[] = []
  const stack: number[] = []
  let next = 0
  for (let s = 0; s < W * H; s++) {
    if (!mask[s] || labels[s]) continue
    const id = ++next
    const reg = { id, area: 0, r0: H, c0: W, r1: 0, c1: 0 }
    labels[s] = id
    stack.push(s)
    while (stack.length) {
      const i = stack.pop()!
      const x = i % W
      const y = (i - x) / W
      reg.area++
      if (y < reg.r0) reg.r0 = y
      if (y + 1 > reg.r1) reg.r1 = y + 1
      if (x < reg.c0) reg.c0 = x
      if (x + 1 > reg.c1) reg.c1 = x + 1
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const yy = y + dy
          const xx = x + dx
          if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue
          const j = yy * W + xx
          if (mask[j] && !labels[j]) {
            labels[j] = id
            stack.push(j)
          }
        }
    }
    regions.push(reg)
  }

  const cells: Segmented[] = []
  for (const r of regions) {
    if (r.area < minArea || r.area > maxArea) continue
    const h = r.r1 - r.r0
    const w = r.c1 - r.c0
    const s = Math.max(h, w)
    const sq = new Uint8Array(s * s * 3)
    const oy = (s - h) >> 1
    const ox = (s - w) >> 1
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = (r.r0 + y) * W + r.c0 + x
        if (labels[i] !== r.id) continue
        const o = ((y + oy) * s + x + ox) * 3
        sq[o] = data[i * 4]
        sq[o + 1] = data[i * 4 + 1]
        sq[o + 2] = data[i * 4 + 2]
      }
    cells.push({ bbox: [r.r0, r.c0, r.r1, r.c1], crop: resizeBilinear(sq, s, s, SIZE, SIZE) })
  }
  cells.sort((a, b) => Math.floor(a.bbox[0] / 60) - Math.floor(b.bbox[0] / 60) || a.bbox[1] - b.bbox[1])
  return cells
}
