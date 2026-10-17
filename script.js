// ---- sidebar
const menuBtn = document.getElementById('menuBtn');
const sidebar = document.getElementById('sidebar');
const scrim = document.getElementById('scrim');
function setMenu(open) {
  sidebar.classList.toggle('open', open);
  scrim.classList.toggle('open', open);
  menuBtn.setAttribute('aria-expanded', open);
  sidebar.setAttribute('aria-hidden', !open);
}
if (menuBtn) {
  menuBtn.addEventListener('click', () => setMenu(!sidebar.classList.contains('open')));
  scrim.addEventListener('click', () => setMenu(false));
  document.addEventListener('keydown', e => e.key === 'Escape' && setMenu(false));
}

// ---- upload page
const fileInput = document.getElementById('xrayFile');
if (fileInput) {
  const preview = document.getElementById('preview');
  const img = document.getElementById('previewImg');
  const scanBtn = document.getElementById('scanBtn');
  const err = document.getElementById('uploadError');
  let file = null;
  const showError = msg => { err.textContent = msg; err.hidden = false; };

  fileInput.addEventListener('change', () => {
    err.hidden = true;
    file = fileInput.files[0];
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      preview.hidden = true;
      return showError('This image is invalid. Please use a chest X-ray image.');
    }
    img.src = URL.createObjectURL(file);
    preview.hidden = false;
  });

  scanBtn.addEventListener('click', async () => {
    if (!file) return;
    scanBtn.disabled = true;
    scanBtn.textContent = 'Checking image...';
    err.hidden = true;
    const fd = new FormData();
    fd.append('xray', file);
    try {
      const res = await fetch('/api/scan', { method: 'POST', body: fd });
      const data = await res.json();
      if (data.ok) { window.location.href = data.redirect; return; }
      showError(data.error);
    } catch (e) {
      showError('Something went wrong. Please try again.');
    }
    scanBtn.disabled = false;
    scanBtn.textContent = 'Start Scanning';
  });
}

// ---- scanning page (about 5 seconds, then go to results)
const scanPage = document.getElementById('scanPage');
if (scanPage) {
  const fill = document.getElementById('fill');
  const pct = document.getElementById('pct');
  const status = document.getElementById('status');
  const steps = ['Preparing image...', 'Extracting lung features...', 'Running CNN analysis...', 'Calculating probabilities...'];
  const DURATION = 5000, t0 = performance.now();
  (function tick(now) {
    const p = Math.min((now - t0) / DURATION, 1);
    fill.style.width = (p * 100) + '%';
    pct.textContent = Math.round(p * 100) + '%';
    status.textContent = steps[Math.min(Math.floor(p * steps.length), steps.length - 1)];
    if (p < 1) requestAnimationFrame(tick);
    else window.location.href = scanPage.dataset.next;
  })(t0);
}
