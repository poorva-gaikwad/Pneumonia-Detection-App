/* ============================================
   SERVICE WORKER - PWA OFFLINE SUPPORT
   Handles caching, background sync, offline mode
   ============================================ */

const CACHE_VERSION = "v1";
const STATIC_CACHE = `pneumonia-static-${CACHE_VERSION}`;
const DYNAMIC_CACHE = `pneumonia-dynamic-${CACHE_VERSION}`;
const IMAGE_CACHE = `pneumonia-images-${CACHE_VERSION}`;

// Static assets to cache on install
const staticAssets = [
  "/",
  "/login",
  "/register",
  "/static/js/app.js",
  "/static/css/responsive.css",
  "/static/icons/icon-192.png",
  "/static/icons/icon-512.png",
];

// Dynamic routes to cache on demand
const dynamicRoutes = [
  "/patient_dashboard",
  "/doctor_dashboard",
  "/admin_dashboard",
  "/scan_predict",
  "/patient_history",
  "/history",
  "/report",
  "/analytics",
  "/settings",
  "/about",
  "/prevention",
];

/* ─────────────────────────────────────────
   INSTALL EVENT
   Cache all static assets
   ───────────────────────────────────────── */
self.addEventListener("install", (event) => {
  console.log("[Service Worker] Installing...");

  event.waitUntil(
    caches
      .open(STATIC_CACHE)
      .then((cache) => {
        console.log("[Service Worker] Caching static assets");
        return cache.addAll(staticAssets).catch((err) => {
          console.warn("[Service Worker] Failed to cache some assets", err);
          // Don't fail install if some assets are missing
        });
      })
      .then(() => {
        self.skipWaiting(); // Activate immediately
      })
  );
});

/* ─────────────────────────────────────────
   ACTIVATE EVENT
   Clean old cache versions
   ───────────────────────────────────────── */
self.addEventListener("activate", (event) => {
  console.log("[Service Worker] Activating...");

  event.waitUntil(
    caches
      .keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames
            .filter(
              (name) =>
                name.startsWith("pneumonia-") &&
                name !== STATIC_CACHE &&
                name !== DYNAMIC_CACHE &&
                name !== IMAGE_CACHE
            )
            .map((name) => {
              console.log("[Service Worker] Deleting old cache:", name);
              return caches.delete(name);
            })
        );
      })
      .then(() => {
        return self.clients.claim(); // Take control of all clients
      })
  );
});

/* ─────────────────────────────────────────
   FETCH EVENT
   Smart caching strategy:
   - Static: cache first
   - API: network first with cache fallback
   - Images: cache first
   ───────────────────────────────────────── */
self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip cross-origin requests
  if (url.origin !== location.origin) {
    return;
  }

  // Static assets: Cache first, fallback to network
  if (request.method === "GET" && isStaticAsset(url.pathname)) {
    event.respondWith(
      caches.match(request).then((response) => {
        if (response) {
          return response;
        }
        return fetch(request)
          .then((response) => {
            // Only cache successful responses
            if (response.ok && response.status === 200) {
              const cache = request.url.includes("/static/")
                ? STATIC_CACHE
                : DYNAMIC_CACHE;
              const responseClone = response.clone();
              caches.open(cache).then((cache) => {
                cache.put(request, responseClone);
              });
            }
            return response;
          })
          .catch(() => {
            // Return offline page if available
            return createOfflineFallback(request);
          });
      })
    );
    return;
  }

  // Images: Cache first
  if (request.method === "GET" && isImage(url.pathname)) {
    event.respondWith(
      caches.match(request).then((response) => {
        if (response) {
          return response;
        }
        return fetch(request)
          .then((response) => {
            if (response.ok && response.status === 200) {
              const responseClone = response.clone();
              caches.open(IMAGE_CACHE).then((cache) => {
                cache.put(request, responseClone);
              });
            }
            return response;
          })
          .catch(() => {
            return createImageFallback();
          });
      })
    );
    return;
  }

  // API/Dynamic: Network first, cache fallback
  if (request.method === "GET") {
    event.respondWith(
      fetch(request)
        .then((response) => {
          // Cache successful responses
          if (response.ok && response.status === 200) {
            const responseClone = response.clone();
            caches.open(DYNAMIC_CACHE).then((cache) => {
              cache.put(request, responseClone);
            });
          }
          return response;
        })
        .catch(() => {
          // Try cache
          return caches.match(request).then((response) => {
            return response || createOfflineFallback(request);
          });
        })
    );
    return;
  }

  // POST/PUT/DELETE: Network only, no caching
  event.respondWith(
    fetch(request).catch(() => {
      return new Response(
        JSON.stringify({ error: "Offline - Cannot process this request" }),
        {
          status: 503,
          statusText: "Service Unavailable",
          headers: { "Content-Type": "application/json" },
        }
      );
    })
  );
});

/* ─────────────────────────────────────────
   HELPER FUNCTIONS
   ───────────────────────────────────────── */

/**
 * Check if URL is a static asset
 */
function isStaticAsset(pathname) {
  return (
    pathname.includes("/static/css/") ||
    pathname.includes("/static/js/") ||
    pathname.includes("/static/fonts/") ||
    pathname.endsWith(".woff") ||
    pathname.endsWith(".woff2")
  );
}

/**
 * Check if URL is an image
 */
function isImage(pathname) {
  return pathname.match(/\.(png|jpg|jpeg|gif|svg|webp)$/i);
}

/**
 * Create offline HTML fallback
 */
function createOfflineFallback(request) {
  return new Response(
    `
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Offline Mode</title>
      <style>
        * {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }
        body {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
          background: #e3f2fd;
          color: #0a2540;
          display: flex;
          align-items: center;
          justify-content: center;
          min-height: 100vh;
          padding: 20px;
        }
        .offline-container {
          background: white;
          padding: 40px;
          border-radius: 12px;
          text-align: center;
          max-width: 500px;
          box-shadow: 0 4px 16px rgba(13, 71, 161, 0.12);
        }
        .offline-icon {
          font-size: 64px;
          margin-bottom: 20px;
        }
        h1 {
          color: #1976d2;
          margin-bottom: 16px;
          font-size: 24px;
        }
        p {
          color: #5a6c7d;
          margin-bottom: 24px;
          line-height: 1.6;
        }
        .actions {
          display: flex;
          gap: 12px;
          flex-wrap: wrap;
          justify-content: center;
        }
        button {
          padding: 12px 24px;
          border-radius: 8px;
          border: none;
          font-size: 16px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.3s ease;
        }
        .btn-primary {
          background: #1976d2;
          color: white;
        }
        .btn-primary:hover {
          background: #1565c0;
        }
        .btn-secondary {
          background: #e3f2fd;
          color: #1976d2;
          border: 2px solid #1976d2;
        }
        .btn-secondary:hover {
          background: #bbdefb;
        }
        .cached-content {
          margin-top: 24px;
          padding-top: 24px;
          border-top: 1px solid #e0e0e0;
          text-align: left;
        }
        .cached-item {
          padding: 12px;
          background: #f9fbff;
          border-radius: 8px;
          margin-bottom: 8px;
          text-decoration: none;
          color: #1976d2;
          display: block;
          transition: background 0.3s ease;
        }
        .cached-item:hover {
          background: #e3f2fd;
        }
      </style>
    </head>
    <body>
      <div class="offline-container">
        <div class="offline-icon">📡</div>
        <h1>No Connection</h1>
        <p>You're currently offline. Some features may not be available. Please check your internet connection.</p>
        <div class="actions">
          <button class="btn-primary" onclick="location.reload()">Try Again</button>
          <button class="btn-secondary" onclick="history.back()">Go Back</button>
        </div>
      </div>
    </body>
    </html>
  `,
    {
      status: 200,
      statusText: "OK",
      headers: { "Content-Type": "text/html" },
    }
  );
}

/**
 * Create offline image fallback
 */
function createImageFallback() {
  // Return a placeholder SVG
  return new Response(
    `
    <svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200">
      <rect width="200" height="200" fill="#e3f2fd"/>
      <text x="50%" y="50%" text-anchor="middle" dy=".3em" fill="#1976d2" font-family="Arial" font-size="14">
        Image not available
      </text>
    </svg>
  `,
    {
      status: 200,
      headers: { "Content-Type": "image/svg+xml" },
    }
  );
}

/* ─────────────────────────────────────────
   BACKGROUND SYNC (Optional)
   Push notifications when back online
   ───────────────────────────────────────── */
self.addEventListener("sync", (event) => {
  if (event.tag === "sync-scans") {
    event.waitUntil(
      // Notify clients that we're back online
      self.clients.matchAll().then((clients) => {
        clients.forEach((client) => {
          client.postMessage({
            type: "SYNC_COMPLETE",
            message: "Connected! Your data is being synced.",
          });
        });
      })
    );
  }
});

console.log("[Service Worker] Loaded successfully");
