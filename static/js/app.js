/* ============================================
   PNEUMONIA APP - CORE FUNCTIONALITY
   PWA support, offline detection, notifications
   ============================================ */

/* ─────────────────────────────────────────
   SERVICE WORKER REGISTRATION
   ───────────────────────────────────────── */
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker
      .register("/static/service-worker.js")
      .then((reg) => {
        console.log("[App] Service Worker registered:", reg);

        // Check for updates
        reg.addEventListener("updatefound", () => {
          const newWorker = reg.installing;
          newWorker.addEventListener("statechange", () => {
            if (
              newWorker.state === "installed" &&
              navigator.serviceWorker.controller
            ) {
              showToast("New version available! Refresh to update.", "info");
            }
          });
        });

        // Listen for messages from Service Worker
        navigator.serviceWorker.addEventListener("message", (event) => {
          const { type, message } = event.data;
          if (type === "SYNC_COMPLETE") {
            showToast(message, "success");
          }
        });
      })
      .catch((err) => console.error("[App] SW registration failed:", err));
  });
}

/* ─────────────────────────────────────────
   PWA INSTALL PROMPT HANDLER
   ───────────────────────────────────────── */
let deferredPrompt;

window.addEventListener("beforeinstallprompt", (e) => {
  e.preventDefault();
  deferredPrompt = e;

  // Show install button if it exists
  const installBtn = document.getElementById("installBtn");
  if (installBtn) {
    installBtn.style.display = "inline-block";
    installBtn.addEventListener("click", handleInstallClick);
  }

  // Auto-show on mobile
  if (isMobileDevice()) {
    showToast("📱 Install this app for quick access!", "info", 5000);
  }
});

function handleInstallClick() {
  if (!deferredPrompt) {
    showToast("App is already installed", "info");
    return;
  }

  deferredPrompt.prompt();
  deferredPrompt.userChoice.then((choice) => {
    if (choice.outcome === "accepted") {
      showToast("✅ App installed successfully!", "success");
      console.log("[App] App installed");
    } else {
      console.log("[App] Install cancelled");
    }
    deferredPrompt = null;
  });
}

window.addEventListener("appinstalled", () => {
  console.log("[App] PWA installed");
  showToast("✅ App installed successfully!", "success");
});

/* ─────────────────────────────────────────
   OFFLINE/ONLINE DETECTION
   ───────────────────────────────────────── */
const offlineBanner =
  document.querySelector(".offline-banner") || createOfflineBanner();

function updateOnlineStatus() {
  const isOnline = navigator.onLine;

  if (isOnline) {
    offlineBanner.classList.remove("active");
    console.log("[App] Back online");
    showToast("✅ Connection restored", "success", 3000);
  } else {
    offlineBanner.classList.add("active");
    console.log("[App] Offline mode");
    showToast("📡 You're offline. Using cached data.", "warning", 5000);
  }
}

function createOfflineBanner() {
  const banner = document.createElement("div");
  banner.className = "offline-banner";
  banner.innerHTML = "📡 You're offline. Data may be cached.";
  document.body.appendChild(banner);
  return banner;
}

window.addEventListener("online", updateOnlineStatus);
window.addEventListener("offline", updateOnlineStatus);
document.addEventListener("DOMContentLoaded", updateOnlineStatus);

/* ─────────────────────────────────────────
   TOAST NOTIFICATION SYSTEM
   ───────────────────────────────────────── */
function showToast(message, type = "info", duration = 4000) {
  const container = getToastContainer();

  const toast = document.createElement("div");
  toast.className = `toast toast--${type} fade-in`;

  const content = document.createElement("div");
  content.style.flex = "1";
  content.textContent = message;

  const closeBtn = document.createElement("button");
  closeBtn.className = "toast-close";
  closeBtn.innerHTML = "×";
  closeBtn.setAttribute("aria-label", "Close notification");
  closeBtn.addEventListener("click", () => removeToast(toast));

  toast.appendChild(content);
  toast.appendChild(closeBtn);
  container.appendChild(toast);

  // Auto-remove after duration
  if (duration > 0) {
    setTimeout(() => removeToast(toast), duration);
  }

  return toast;
}

function removeToast(toast) {
  toast.classList.add("fade-out");
  setTimeout(() => toast.remove(), 300);
}

function getToastContainer() {
  let container = document.querySelector(".toast-container");
  if (!container) {
    container = document.createElement("div");
    container.className = "toast-container";
    document.body.appendChild(container);
  }
  return container;
}

/* ─────────────────────────────────────────
   DARK MODE SUPPORT
   ───────────────────────────────────────── */
function initDarkMode() {
  const darkModeToggle = document.getElementById("darkModeToggle");
  if (!darkModeToggle) return;

  const isDarkMode =
    localStorage.getItem("darkMode") === "true" ||
    window.matchMedia("(prefers-color-scheme: dark)").matches;

  setDarkMode(isDarkMode);

  darkModeToggle.addEventListener("change", (e) => {
    setDarkMode(e.target.checked);
  });
}

function setDarkMode(enabled) {
  if (enabled) {
    document.documentElement.style.colorScheme = "dark";
    localStorage.setItem("darkMode", "true");
  } else {
    document.documentElement.style.colorScheme = "light";
    localStorage.setItem("darkMode", "false");
  }
}

/* ─────────────────────────────────────────
   FORM ENHANCEMENTS
   ───────────────────────────────────────── */
function initFormHandlers() {
  // Confirm before leaving page if form has changes
  const forms = document.querySelectorAll("form");
  forms.forEach((form) => {
    let isDirty = false;

    const inputs = form.querySelectorAll("input, textarea, select");
    inputs.forEach((input) => {
      input.addEventListener("change", () => {
        isDirty = true;
      });
    });

    form.addEventListener("submit", () => {
      isDirty = false;
    });

    window.addEventListener("beforeunload", (e) => {
      if (isDirty) {
        e.preventDefault();
        e.returnValue = "";
      }
    });
  });
}

/* ─────────────────────────────────────────
   DRAG & DROP FILE UPLOAD
   ───────────────────────────────────────── */
function initDragDrop() {
  const dropZones = document.querySelectorAll("[data-drop-zone]");

  dropZones.forEach((zone) => {
    ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
      zone.addEventListener(eventName, preventDefaults, false);
    });

    ["dragenter", "dragover"].forEach((eventName) => {
      zone.addEventListener(eventName, highlight, false);
    });

    ["dragleave", "drop"].forEach((eventName) => {
      zone.addEventListener(eventName, unhighlight, false);
    });

    zone.addEventListener("drop", (e) => handleDrop(e, zone), false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  function highlight(e) {
    preventDefaults(e);
    this.style.backgroundColor = "#bbdefb";
    this.style.borderColor = "#1976d2";
  }

  function unhighlight(e) {
    preventDefaults(e);
    this.style.backgroundColor = "";
    this.style.borderColor = "";
  }

  function handleDrop(e, zone) {
    const dt = e.dataTransfer;
    const files = dt.files;
    const fileInput = zone.querySelector('input[type="file"]');

    if (fileInput) {
      fileInput.files = files;
      showToast(`📁 ${files.length} file(s) selected`, "success");
    }
  }
}

/* ─────────────────────────────────────────
   RESPONSIVE NAVBAR MENU
   ───────────────────────────────────────── */
function initNavbar() {
  const toggle = document.querySelector(".navbar-toggle");
  const navLinks = document.querySelector(".navbar-links");

  if (toggle && navLinks) {
    toggle.addEventListener("click", () => {
      navLinks.classList.toggle("active");
    });

    // Close menu when link is clicked
    navLinks.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", () => {
        navLinks.classList.remove("active");
      });
    });
  }
}

/* ─────────────────────────────────────────
   ACCESSIBILITY ENHANCEMENTS
   ───────────────────────────────────────── */
function initAccessibility() {
  // Skip to main content
  const skipLink = document.querySelector(".skip-to-main");
  if (skipLink) {
    skipLink.addEventListener("click", (e) => {
      e.preventDefault();
      const main =
        document.querySelector("main") || document.querySelector("body");
      main.focus();
      main.scrollIntoView();
    });
  }

  // Announce live regions
  const alerts = document.querySelectorAll("[role='alert']");
  alerts.forEach((alert) => {
    alert.setAttribute("aria-live", "polite");
    alert.setAttribute("aria-atomic", "true");
  });
}

/* ─────────────────────────────────────────
   ACCORDION (FAQ / Expandable Sections)
   Accessible, keyboard-friendly accordion helper
   ───────────────────────────────────────── */
function initAccordions() {
  const accordions = document.querySelectorAll("[data-accordion]");
  accordions.forEach((acc) => {
    const items = acc.querySelectorAll("[data-accordion-item]");
    items.forEach((item) => {
      const header = item.querySelector("[data-accordion-header]");
      const panel = item.querySelector("[data-accordion-panel]");
      if (!header || !panel) return;

      header.setAttribute("role", "button");
      header.setAttribute("tabindex", "0");
      header.setAttribute("aria-expanded", "false");
      panel.setAttribute("hidden", "");

      function toggle() {
        const open = header.getAttribute("aria-expanded") === "true";
        header.setAttribute("aria-expanded", String(!open));
        if (open) {
          panel.setAttribute("hidden", "");
        } else {
          panel.removeAttribute("hidden");
        }
      }

      header.addEventListener("click", toggle);
      header.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          toggle();
        }
      });
    });
  });
}

/* ─────────────────────────────────────────
   UTILITY FUNCTIONS
   ───────────────────────────────────────── */
function isMobileDevice() {
  return /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini/i.test(
    navigator.userAgent.toLowerCase()
  );
}

function formatDate(date) {
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(date));
}

function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

/* ─────────────────────────────────────────
   INITIALIZE ON PAGE LOAD
   ───────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  console.log("[App] Initializing...");

  initNavbar();
  initFormHandlers();
  initDragDrop();
  initDarkMode();
  initAccessibility();
  initAccordions();

  console.log("[App] Ready");
});

// Export functions for global use
window.showToast = showToast;
window.handleInstallClick = handleInstallClick;
