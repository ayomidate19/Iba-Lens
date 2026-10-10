// Cache-first runtime cache: after one online visit the app, model and WASM work offline.
const CACHE = 'iba-lens-v1'
self.addEventListener('install', (e) => self.skipWaiting())
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()))
self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET' || new URL(e.request.url).origin !== location.origin) return
  e.respondWith(caches.open(CACHE).then(async (c) => {
    const hit = await c.match(e.request)
    if (hit) return hit
    const res = await fetch(e.request)
    if (res.ok) c.put(e.request, res.clone())
    return res
  }))
})
