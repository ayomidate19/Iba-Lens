import { cellBand, loadThresholds, predict, summarise, type Band, type Summary } from './lib/model'
import { segment, type Box } from './lib/segment'

export interface Cell {
  bbox: Box
  crop: Uint8Array
  prob: number
  band: Band
}
export interface Analysis {
  id: string
  name: string
  url: string
  width: number
  height: number
  cells: Cell[]
  summary: Summary
  source: 'demo' | 'scan'
}

const MAX_SIDE = 1600

export async function toImageData(url: string): Promise<ImageData> {
  const img = new Image()
  img.src = url
  await img.decode()
  const s = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight))
  const c = document.createElement('canvas')
  c.width = Math.round(img.naturalWidth * s)
  c.height = Math.round(img.naturalHeight * s)
  const ctx = c.getContext('2d', { willReadFrequently: true })!
  ctx.drawImage(img, 0, 0, c.width, c.height)
  return ctx.getImageData(0, 0, c.width, c.height)
}

export async function analyse(id: string, name: string, url: string, source: Analysis['source']): Promise<Analysis> {
  const [img, t] = await Promise.all([toImageData(url), loadThresholds()])
  await new Promise((r) => setTimeout(r))
  const seg = segment(img)
  const probs = await predict(seg.map((s) => s.crop))
  const cells = seg.map((s, i) => ({ ...s, prob: probs[i], band: cellBand(probs[i], t) }))
  return { id, name, url, width: img.width, height: img.height, cells, summary: summarise(probs, t), source }
}
