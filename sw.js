const CACHE='quet-pdf-v19';
const FILES=['./','./index.html','./docdet.js','./jspdf.umd.min.js','./manifest.webmanifest','./icon-192.png','./icon-512.png','./apple-touch-icon.png','./vendor/ort/ort.wasm.min.js','./vendor/ort/ort-wasm-simd-threaded.mjs'];
// mô hình AI nặng: tải 1 lần rồi giữ trong bộ nhớ đệm riêng (không xoá khi app cập nhật)
const MODEL_CACHE='quet-pdf-models-v2';
self.addEventListener('install',e=>{e.waitUntil(caches.open(CACHE).then(c=>c.addAll(FILES)).then(()=>self.skipWaiting()));});
self.addEventListener('activate',e=>{e.waitUntil(caches.keys().then(ks=>Promise.all(ks.filter(k=>k!==CACHE&&k!==MODEL_CACHE).map(k=>caches.delete(k)))).then(()=>self.clients.claim()));});
self.addEventListener('fetch',e=>{
  const r=e.request; if(r.method!=='GET') return;
  const url=new URL(r.url);
  if(url.origin!==location.origin){ e.respondWith(caches.match(r).then(m=>m||fetch(r).then(res=>{const cp=res.clone();caches.open(CACHE).then(c=>c.put(r,cp));return res;}).catch(()=>m))); return; }
  if(/\/vendor\/(models\/|ort\/.*\.gz$)/.test(url.pathname)){
    // ưu tiên bản đã lưu; chưa có thì tải rồi lưu
    e.respondWith(caches.open(MODEL_CACHE).then(c=>c.match(r).then(m=>m||fetch(r).then(res=>{ if(res.ok) c.put(r,res.clone()); return res; }))));
    return;
  }
  // file của app: lấy bản mới khi có mạng, mất mạng thì dùng bản đã lưu
  e.respondWith(fetch(r).then(res=>{const cp=res.clone();caches.open(CACHE).then(c=>c.put(r,cp));return res;}).catch(()=>caches.match(r).then(m=>m||caches.match('./index.html'))));
});
