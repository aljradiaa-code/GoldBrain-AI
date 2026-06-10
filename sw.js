// MVT-VRL Trader Service Worker
const CACHE_NAME = 'mvt-trader-v3.1';
const ASSETS = ['./index.html', './manifest.json'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE_NAME).then(c => c.addAll(ASSETS)));
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys =>
    Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
  ));
  self.clients.claim();
});

self.addEventListener('fetch', e => {
  const url = e.request.url;

  // استراتيجية التحديث: "الشبكة أولاً" لملفات النموذج والبيانات الحساسة
  // هذا يضمن تحميل أحدث أوزان فور رفعها من بوت التدريب
  if (url.includes('model.json') || url.includes('model_weights.bin') || 
      url.includes('api.') || url.includes('binance') || url.includes('twelvedata')) {
    e.respondWith(
      fetch(e.request)
        .then(response => {
          return caches.open(CACHE_NAME).then(cache => {
            cache.put(e.request, response.clone());
            return response;
          });
        })
        .catch(() => caches.match(e.request))
    );
  } else {
    // استراتيجية "الكاش أولاً" للملفات الثابتة (التصميم، الأيقونات)
    e.respondWith(
      caches.match(e.request).then(cached => cached || fetch(e.request))
    );
  }
});
