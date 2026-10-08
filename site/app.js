// findmybc.app: find the right Business Central app.
// Routing: problem text -> need group (bge-small embeddings in the browser, keyword fallback).
// Questions: pre-baked per need group + a few generic ones built from listing facts.
// Facts are only what each AppSource listing states; silence is "not stated", never "no".

const $ = (s, el = document) => el.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const view = $('#view');

const S = {
  groups: [], apps: [], stm: [], byGroup: new Map(), byGroup2: new Map(),
  query: '', ranked: [], mentions: [],
  country: load('af-country') || '',
  flow: {},            // per group: {asked:[qid], answers:{qid:[optIdx]}}
  shape: 'all',
};
function load(k){ try { return localStorage.getItem(k); } catch(e){ return null; } }
function save(k,v){ try { localStorage.setItem(k,v); } catch(e){} }

// Free-text labels compare case- and space-insensitively everywhere
const labelKey = v => String(v).trim().replace(/\s+/g, ' ').toLowerCase();
const SHAPES = {process:'Apps', connector:'Connectors', usability:'Tools', addon:'Add-ons', vertical:'Industry solutions', localization:'Localizations', language:'Languages', platform:'Hosted platforms', other:'Other'};
const PRICING = {free:'Free', free_trial:'Free trial', subscription:'Subscription', per_user:'Per user', per_company:'Per company', per_transaction:'Per transaction', one_time:'One-time', contact:'Contact publisher', not_stated:'Not mentioned'};
const AREA_ORDER = ['Purchasing & payables','Sales & receivables','Finance & reporting','Inventory & warehouse','Manufacturing','Projects','Field service','CRM & quotes','Customer service','People & payroll','Planning & budgets','Products & items','Fixed assets & maintenance','Marketing','Administration & IT','Working faster in BC','Connecting other systems','Country requirements','Industry solutions'];
const EXAMPLES = [
  'We type every supplier invoice in by hand',
  'Our Shopify orders need to flow into BC',
  'Bank reconciliation takes forever',
  'Customers pay late and nobody chases them',
];
let regionName = c => c, langName = c => c;
try { const dl = new Intl.DisplayNames(['en'], {type:'language'}); langName = c => { try { return dl.of(c) || c; } catch(e){ return c; } }; } catch(e){}
try { const dn = new Intl.DisplayNames(['en'], {type:'region'}); regionName = c => { try { return dn.of(c) || c; } catch(e){ return c; } }; } catch(e){}

/* ---------------- data ---------------- */
async function boot(){
  view.innerHTML = `<div class="thinking"><span class="dot3"></span>Loading…</div>`;
  const [g, a, st] = await Promise.all(['groups','apps','statements'].map(f => fetch(`data/${f}.json`).then(r => r.json())));
  S.groups = g; S.apps = a; S.stm = st;
  a.forEach(app => {
    app.text = (app.n + ' ' + app.s + ' ' + app.k.join(' ')).toLowerCase();
    app.g.forEach(gi => { if (!S.byGroup.has(gi)) S.byGroup.set(gi, []); S.byGroup.get(gi).push(app); });
    app.g2.forEach(gi => { if (!S.byGroup2.has(gi)) S.byGroup2.set(gi, []); S.byGroup2.get(gi).push(app); });
  });
  g.forEach(gr => gr.q.forEach(q => q.options.forEach(o => {
    o.re = (o.patterns || []).map(p => { try { return new RegExp(p, 'i'); } catch(e){ return null; } }).filter(Boolean);
  })));
  // named systems for the hybrid lookup
  const sys = new Map();
  a.forEach(app => app.x.forEach(x => { const k = labelKey(x); if (k.length > 2) { const e = sys.get(k) || {n: 0, label: x.trim()}; e.n++; sys.set(k, e); } }));
  S.systems = [...sys].filter(([, e]) => e.n >= 3).map(([, e]) => e.label);
  initCountry();
  initEngine();
  window.addEventListener('hashchange', route);
  route();
  coverageNotice();
}

/* ---------------- matcher ---------------- */
const M = { mode: 'loading', V: null, D: 0, owner: null, extractor: null, idf: null, toks: null };
async function initEngine(){
  M.owner = Int32Array.from(S.stm.map(s => s[0]));
  buildKeyword();
  setStatus('Matcher: keyword matching (loading the language model…)');
  try {
    const r = await fetch('data/vectors.bin');
    if (!r.ok) throw new Error('no vectors');
    M.V = new Float32Array(await r.arrayBuffer());
    M.D = M.V.length / S.stm.length;
    if (!Number.isInteger(M.D) || M.D < 64) throw new Error('vectors do not match statements; rebuild them');
    const T = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2');
    T.env.allowLocalModels = false;
    M.extractor = await T.pipeline('feature-extraction', 'Xenova/bge-small-en-v1.5');
    M.mode = 'embed';
    setStatus('Matcher: language model running in your browser');
  } catch(e) {
    M.mode = 'keyword';
    setStatus('Matcher: keyword matching (language model not available)');
    console.warn('Embedding engine unavailable:', e);
  }
}
function setStatus(t){ const el = $('#engine-status'); if (el) el.textContent = t; else console.info(t); }
const STOP = new Set('a an the and or of to for in on at is are we our us it its be with from that this by as do does not no can cant don t i my me you your they them their have has had into out up so all any some too very just than then there here what which who how when where'.split(' '));
const tok = s => s.toLowerCase().replace(/[^a-z0-9æøåäöü\- ]/g,' ').split(/\s+/).filter(w => w.length > 2 && !STOP.has(w)).map(w => w.replace(/(ing|ed|es|s)$/,''));
function buildKeyword(){
  const df = new Map(); M.toks = S.stm.map(s => { const t = new Set(tok(s[1])); t.forEach(w => df.set(w, (df.get(w)||0)+1)); return t; });
  const N = S.stm.length; M.idf = w => Math.log(1 + N / (1 + (df.get(w) || 0)));
}
async function scoreGroups(text){
  const n = S.groups.length, sims = new Float32Array(S.stm.length);
  if (M.mode === 'embed') {
    const out = await M.extractor(text, {pooling:'cls', normalize:true});
    const q = out.data, D = M.D, V = M.V;
    for (let i = 0; i < S.stm.length; i++){ let s = 0; const o = i*D; for (let d = 0; d < D; d++) s += V[o+d]*q[d]; sims[i] = s; }
  } else {
    const qt = new Set(tok(text)); let qn = 0; qt.forEach(w => qn += M.idf(w));
    M.toks.forEach((t, i) => { let s = 0; qt.forEach(w => { if (t.has(w)) s += M.idf(w); }); sims[i] = qn ? s / Math.sqrt(qn * Math.max(1, t.size)) : 0; });
  }
  const top = Array.from({length:n}, () => [-1,-1,-1]);
  for (let i = 0; i < sims.length; i++){ const t = top[M.owner[i]], v = sims[i];
    if (v > t[0]) { t[2]=t[1]; t[1]=t[0]; t[0]=v; } else if (v > t[1]) { t[2]=t[1]; t[1]=v; } else if (v > t[2]) t[2]=v; }
  return top.map((t, gi) => ({gi, s: (t[0] + Math.max(t[1],0) + Math.max(t[2],0)) / 3}))
            .filter(x => (S.byGroup.get(x.gi) || []).length)
            .sort((a, b) => b.s - a.s);
}

/* ---------------- routing ---------------- */
function go(h){ if (location.hash === h) route(); else location.hash = h; }
function route(){
  const h = location.hash.slice(1) || '/';
  const [path, qstr] = h.split('?');
  const [, p, id] = path.split('/');
  closeDrawer();
  if (!p) return renderHome();
  if (p === 'match') return S.query ? renderMatch() : go('#/');
  if (p === 'need') return renderQuestions(+id);
  if (p === 'results') return renderResults(fromShare(id, qstr));
  if (p === 'browse') return renderBrowse();
  if (p === 'about') return renderAbout();
  renderHome();
}

/* ---------------- shareable shortlists ----------------
   #/results/<need id>?a=<qid>.<opt>-<opt>~<qid>.<opt>&t=<shape>&c=<country>
   Uses the need's id, not its index, so links survive a rebuild. */
function fromShare(id, qstr){
  const gi = /^\d+$/.test(id || '') ? +id : S.groups.findIndex(g => g.id === id);
  if (gi < 0 || !qstr || location.hash === S.lastShare) return gi;   // our own replaceState: nothing to apply
  const ps = new URLSearchParams(qstr);
  const c = ps.get('c') || '';
  if (c !== S.country){ S.country = c; const l = $('#country-label'); if (l) l.textContent = c ? regionName(c) : 'Any'; }
  S.shape = ps.get('t') || 'all';
  const qs = allQuestions(gi), answers = {};
  (ps.get('a') || '').split('~').filter(Boolean).forEach(part => {
    const [qid, ix] = part.split('.'); const q = qs.find(x => x.id === qid); if (!q || !ix) return;
    const sel = ix.split('-').map(Number).filter(n => Number.isInteger(n) && q.options[n]);
    if (sel.length) answers[qid] = sel;
  });
  S.flow[gi] = {asked: Object.keys(answers), answers};
  return gi;
}
function shareHash(gi){
  const f = S.flow[gi] || {answers:{}}, ps = new URLSearchParams();
  const a = Object.entries(f.answers).filter(([, sel]) => sel.length).map(([q, sel]) => q + '.' + sel.join('-')).join('~');
  if (a) ps.set('a', a);
  if (S.shape !== 'all') ps.set('t', S.shape);
  if (S.country) ps.set('c', S.country);
  const qs = ps.toString();
  return '#/results/' + S.groups[gi].id + (qs ? '?' + qs : '');
}

/* ---------------- about ---------------- */
const REPO = 'https://github.com/soren-alexandersen/findmybcapp';
function renderAbout(){
  view.innerHTML = `
    <article class="prose">
      <h2>About findmybc.app</h2>
      <p>A neutral, non-commercial finder for Business Central apps on the Microsoft Marketplace. Describe what you need or what isn't working, answer a few questions, and get a short list of apps whose listings say they handle it.</p>
      <h3>Where the information comes from</h3>
      <p>Only from each app's own Microsoft Marketplace listing. We read every listing and describe it in our own words, with a link to the original. If a listing doesn't mention something, we show it as "not mentioned": we can't tell whether the app does it. The catalog is refreshed every month.</p>
      <p>We don't reproduce Marketplace listings. App names belong to their publishers; every summary is our own interpretation.</p>
      <p>The order of results is not a rating. Apps that match more of your answers come first, then apps whose listings describe more capabilities.</p>
      <h3>Privacy</h3>
      <ul>
        <li>No account, no cookies, no tracking, no analytics.</li>
        <li>What you type stays in your browser. The matching runs on your own device and nothing you type is sent to us.</li>
        <li>The site is hosted on GitHub Pages. To match on your device, your browser also downloads a small language model from jsDelivr and Hugging Face. Like any web host, these services see your IP address.</li>
        <li>Your country choice is saved in your browser only.</li>
      </ul>
      <h3>For publishers</h3>
      <p>The best way to be found is a clear Marketplace listing: what your app does, for which countries and languages, and what it works with. You can also add countries, languages and related apps for your app on <a href="${REPO}" target="_blank" rel="noopener">GitHub</a>.</p>
      <p><b>Your logo is your choice.</b> We show a publisher's logo only when the publisher asks us to. Until then, your apps show the first letter of their name. To show your logo, set <code>showLogo</code> to true and add a <code>logoUrl</code> to your app's file in <a href="${REPO}/tree/main/publisher-data" target="_blank" rel="noopener">publisher-data</a> on GitHub, then open a pull request.</p>
      <h3>Contact</h3>
      <p>Something wrong or missing? <a href="${REPO}/issues" target="_blank" rel="noopener">Open an issue on GitHub</a>.</p>
      <p class="quiet">Not affiliated with or endorsed by Microsoft.</p>
    </article>`;
}

/* ---------------- home ---------------- */
function renderHome(){
  view.innerHTML = `
    <section class="hero">
      <h1><span class="hl">Describe the business need.</span> <span class="h1sub">Find the Business Central solutions that claim to solve it.</span></h1>
      <p>You don't need to know what the app is called. It's free, independent and takes about two minutes.</p>
      <form class="ask" id="ask">
        <textarea id="q" rows="1" placeholder="Describe what you need, or what isn't working" aria-label="Describe your business need">${esc(S.query)}</textarea>
        <button class="btn" type="submit">Find apps</button>
      </form>
      <p class="where">Showing apps for <b>${S.country ? esc(regionName(S.country)) : 'any country'}</b> · <button type="button" id="where-change">${S.country ? 'Change' : 'Choose your country'}</button></p>
      <div class="examples"><span class="rlabel">Try</span>${EXAMPLES.map(e => `<button class="ex" type="button">${esc(e)}</button>`).join('')}</div>
      <div class="showcase">
        <ol class="how">
          <li><b>Describe it</b><span>In your own words, the way you'd tell a colleague.</span></li>
          <li><b>Answer up to 5 quick questions</b><span>Each one narrows the list. Skip any you don't care about.</span></li>
          <li><b>See what each app claims</b><span>And what its listing doesn't mention, so you know what to ask.</span></li>
        </ol>
        <div class="demo" aria-hidden="true">
          <div class="demo-tag">Example result</div>
          <div class="app demo-card"><span class="icon">E</span><span style="min-width:0">
            <div class="nm">Example Invoice Capture</div><div class="pb">Example Publisher</div>
            <div class="sm">Reads supplier invoices from email and drafts them in BC</div>
            <div class="hits"><span class="score">Matches 3 of 4</span><span class="hit">✓ PDF attachments by email</span><span class="hit">✓ E-invoices</span><span class="hit">✓ Approve on phone</span><span class="hit unk">Not mentioned: Auto-approve small invoices</span></div>
          </span></div>
        </div>
      </div>
      <div class="proof">
        <p><b>${S.apps.length.toLocaleString('en')}</b> solutions for Business Central from <b>${new Set(S.apps.map(a => a.p)).size.toLocaleString('en')}</b> publishers: ${S.apps.filter(a => !a.h).length.toLocaleString('en')} apps and ${S.apps.filter(a => a.h).length.toLocaleString('en')} hosted platforms, mapped to the business needs they serve.</p>
        <ul class="trust"><li>No sign-up</li><li>No paid rankings</li><li>Refreshed monthly from Microsoft Marketplace</li></ul>
        <p class="partners"><b>Partners:</b> run it together with a customer, then send them the shortlist link.</p>
      </div>
    </section>`;
  $('#where-change').onclick = e => { e.stopPropagation(); $('#country-btn').click(); };
  const ta = $('#q');
  const fit = () => { ta.style.height = 'auto'; ta.style.height = ta.scrollHeight + 'px'; };
  ta.addEventListener('input', fit); fit();
  ta.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); submit(); } });
  $('#ask').addEventListener('submit', e => { e.preventDefault(); submit(); });
  view.querySelectorAll('.ex').forEach(b => b.onclick = () => { ta.value = b.textContent; submit(); });
  ta.focus();
  function submit(){ const v = ta.value.trim(); if (!v) return ta.focus(); S.query = v; S.ranked = []; go('#/match'); }
}

/* ---------------- match ---------------- */
async function renderMatch(){
  view.innerHTML = `<div class="thinking"><span class="dot3"></span>Working out what you need…</div>`;
  if (!S.ranked.length) S.ranked = await scoreGroups(S.query);
  const lq = S.query.toLowerCase();
  S.mentions = S.systems.filter(s => new RegExp('\\b' + s.toLowerCase().replace(/[.*+?^${}()|[\]\\]/g,'\\$&') + '\\b').test(lq));
  const r = S.ranked, clear = r.length > 1 && (r[0].s - r[1].s) >= (M.mode === 'embed' ? 0.035 : 0.08);
  const shown = clear ? r.slice(0, 1) : r.slice(0, 3);
  const card = (x, top) => { const g = S.groups[x.gi], n = (S.byGroup.get(x.gi) || []).length;
    return `<button class="need ${top ? 'top' : ''}" data-g="${x.gi}"><div><div class="tag">${esc(g.area)}</div><div class="lbl">${esc(g.label)}</div></div><span class="go">${n} apps →</span><div class="meta">${esc(g.desc)}</div></button>`; };
  view.innerHTML = `
    <section class="step">
      <button class="back" onclick="location.hash='#/'">← Change what you wrote</button>
      <p class="you-said">You said <q>${esc(S.query)}</q></p>
      <h2>${clear ? 'Sounds like this is what you need' : 'Which of these is closest?'}</h2>
      <div class="needs">${shown.map((x, i) => card(x, i === 0 && clear)).join('')}</div>
      <div class="more">
        ${clear ? `<button class="btn ghost small" id="others">Not quite? Show other matches</button>` : ''}
        <a class="btn ghost small" href="#/browse" style="text-decoration:none">None of these, browse all process needs</a>
      </div>
      <div id="others-list"></div>
    </section>`;
  view.querySelectorAll('.need').forEach(b => b.onclick = () => startNeed(+b.dataset.g));
  const o = $('#others'); if (o) o.onclick = () => { $('#others-list').innerHTML = `<div class="needs">${r.slice(1, 5).map(x => card(x, false)).join('')}</div>`; o.remove();
    view.querySelectorAll('#others-list .need').forEach(b => b.onclick = () => startNeed(+b.dataset.g)); };
}
function startNeed(gi){
  const f = S.flow[gi] = {asked: [], answers: {}};
  // pre-answer the system question when the user named a system we know
  if (S.mentions.length){ const q = genericQuestions(gi).find(q => q.id === 'g-system');
    if (q){ const idx = q.options.findIndex(o => !o.any && S.mentions.some(m => labelKey(m) === labelKey(o.label))); if (idx >= 0){ f.asked.push(q.id); f.answers[q.id] = [idx]; } } }
  go('#/need/' + gi);
}

/* ---------------- questions ---------------- */
function pool(gi){ return (S.byGroup.get(gi) || []).filter(a => !S.country || !a.c.length || a.c.includes(S.country)); }
function optMatch(o, app){
  if (o.any) return false;
  if (o.fn) return o.fn(app);
  return o.re.some(r => r.test(app.text));
}
const genericCache = new Map();
function genericQuestions(gi){
  if (genericCache.has(gi)) return genericCache.get(gi);
  const apps = S.byGroup.get(gi) || [], qs = [];
  // Free-text labels are grouped case-insensitively: "Manufacturing" and "manufacturing" are one option
  const top = (key) => {
    const m = new Map();
    apps.forEach(a => new Set((a[key] || []).map(labelKey)).forEach(k => {
      const e = m.get(k) || {n: 0, label: (a[key] || []).find(v => labelKey(v) === k)};
      e.n++; m.set(k, e);
    }));
    return [...m].filter(([, e]) => e.n >= 3).sort((a, b) => b[1].n - a[1].n).slice(0, 5).map(([k, e]) => ({k, label: e.label}));
  };
  const has = (a, key, k) => (a[key] || []).some(v => labelKey(v) === k);
  const sys = top('x');
  if (sys.length >= 2) qs.push({id:'g-system', text:'Does it need to work with a particular system?', help:'Pick the ones you use.', multi:true,
    options:[...sys.map(s => ({label:s.label, fn:a => has(a, 'x', s.k)})), {label:'No particular system', any:true}]});
  const ind = top('ind');
  if (ind.length >= 2) qs.push({id:'g-industry', text:'Which industry are you in?', multi:false,
    options:[...ind.map(s => ({label:s.label, fn:a => has(a, 'ind', s.k)})), {label:'None of these / not important', any:true}]});
  qs.push({id:'g-trial', text:'Do you want to try it for free first?', multi:false,
    options:[{label:'Yes, free or with a free trial', fn:a => a.pr === 'free' || a.pr === 'free_trial'}, {label:"Doesn't matter", any:true}]});
  genericCache.set(gi, qs); return qs;
}
function allQuestions(gi){ return [...S.groups[gi].q.map(q => ({...q, specific:true})), ...genericQuestions(gi)]; }
function appStatus(app, gi){
  const f = S.flow[gi] || {answers:{}}, qs = allQuestions(gi); let yes = 0, unk = 0; const hits = [], misses = [];
  for (const [qid, sel] of Object.entries(f.answers)){
    const q = qs.find(x => x.id === qid); if (!q) continue;
    const real = sel.map(i => q.options[i]).filter(o => !o.any); if (!real.length) continue;
    const hit = real.filter(o => optMatch(o, app));
    if (hit.length){ yes++; hits.push(hit.map(o => o.label).join(' / ')); } else { unk++; misses.push(real.map(o => o.label).join(' / ')); }
  }
  return {yes, unk, hits, misses};
}
function pickNext(gi){
  const f = S.flow[gi], apps = pool(gi); if (f.asked.length >= 5 || apps.length <= 3) return null;
  let best = null;
  for (const q of allQuestions(gi)){
    if (f.asked.includes(q.id)) continue;
    const real = q.options.filter(o => !o.any);
    const cov = apps.filter(a => real.some(o => optMatch(o, a))).length / apps.length;
    if (cov < 0.05) continue;
    const score = Math.min(cov, 1 - cov) + (q.specific ? 0.5 : 0) + (cov > 0.95 ? -0.3 : 0);
    if (!best || score > best.score) best = {q, score};
  }
  return best && best.q;
}
function renderQuestions(gi){
  const g = S.groups[gi]; if (!g) return go('#/');
  const f = S.flow[gi] || (S.flow[gi] = {asked:[], answers:{}});
  const q = pickNext(gi);
  if (!q) return go('#/results/' + gi);
  const apps = pool(gi), confirmed = apps.filter(a => { const st = appStatus(a, gi); return st.unk === 0; }).length;
  const counts = q.options.map(o => o.any ? null : apps.filter(a => optMatch(o, a)).length);
  const answered = Object.entries(f.answers).map(([qid, sel]) => { const qq = allQuestions(gi).find(x => x.id === qid); return qq ? sel.map(i => qq.options[i].label).join(', ') : ''; }).filter(Boolean);
  view.innerHTML = `
    <section class="step">
      <button class="back" id="bk">← Back</button>
      <div class="tag" style="margin-top:14px">${esc(g.area)}</div>
      <h2>${esc(g.label)}</h2>
      <div class="progress">${Object.keys(f.answers).length ? `<span class="count" id="cnt">${apps.length} apps</span><span>for this need · ${confirmed} mention everything you've answered so far</span>` : `<span class="count" id="cnt">${apps.length} apps</span><span>Answer a few questions to narrow it down</span>`}</div>
      ${answered.length ? `<div class="answers">${answered.map(a => `<span class="ans">${esc(a)}</span>`).join('')}</div>` : ''}
      <div class="qwrap">
        <div class="q">${esc(q.text)}</div>
        ${q.help ? `<p class="qhelp">${esc(q.help)}${q.multi ? ' You can pick more than one.' : ''}</p>` : (q.multi ? `<p class="qhelp">You can pick more than one.</p>` : '<p class="qhelp"></p>')}
        <div class="opts">${q.options.map((o, i) => `<button class="opt ${o.any ? 'any' : ''}" data-i="${i}" aria-pressed="false"><span>${esc(o.label)}</span>${counts[i] != null ? `<span class="n">${counts[i]} of ${apps.length} mention this</span>` : ''}</button>`).join('')}</div>
        <div class="qactions">
          ${q.multi ? `<button class="btn" id="next" disabled>Continue</button>` : ''}
          <button class="btn ghost small" id="skip">Skip this question</button>
          <button class="btn ghost small" id="show">Show the ${apps.length} apps now</button>
        </div>
      </div>
    </section>`;
  const sel = new Set();
  view.querySelectorAll('.opt').forEach(b => b.onclick = () => {
    const i = +b.dataset.i, o = q.options[i];
    if (!q.multi || o.any){ answer([i]); return; }
    sel.has(i) ? sel.delete(i) : sel.add(i); b.setAttribute('aria-pressed', sel.has(i));
    $('#next').disabled = !sel.size;
  });
  const nx = $('#next'); if (nx) nx.onclick = () => answer([...sel]);
  $('#skip').onclick = () => { f.asked.push(q.id); renderQuestions(gi); };
  $('#show').onclick = () => go('#/results/' + gi);
  $('#bk').onclick = () => {
    const last = f.asked.pop();
    if (last){ delete f.answers[last]; renderQuestions(gi); } else history.back();
  };
  function answer(ix){ f.asked.push(q.id); f.answers[q.id] = ix; renderQuestions(gi); const c = $('#cnt'); if (c){ c.classList.add('bump'); setTimeout(() => c.classList.remove('bump'), 250); } }
}

/* ---------------- results ---------------- */
function renderResults(gi){
  const g = S.groups[gi]; if (!g) return go('#/');
  const f = S.flow[gi] || (S.flow[gi] = {asked:[], answers:{}});
  const apps = pool(gi), excluded = (S.byGroup.get(gi) || []).length - apps.length;
  const anyAnswers = Object.values(f.answers).some(sel => sel.length);
  const withSt = apps.map(a => ({a, st: appStatus(a, gi)}));
  const shapesPresent = [...new Set(apps.map(a => a.t))];
  const filt = x => S.shape === 'all' || x.a.t === S.shape;
  const rank = (x, y) => (y.st.yes - x.st.yes) || (y.a.k.length - x.a.k.length);
  const scored = anyAnswers && withSt.some(x => x.st.yes + x.st.unk > 0);
  const side = (S.byGroup2.get(gi) || []).filter(a => !apps.includes(a)).filter(a => !S.country || !a.c.length || a.c.includes(S.country));
  const qsAll = allQuestions(gi);
  const chips = Object.entries(f.answers).map(([qid, sel]) => { const qq = qsAll.find(x => x.id === qid);
    return qq && sel.length ? {qid, q: qq.text, opts: sel.map(i => ({i, label: qq.options[i].label}))} : null; }).filter(Boolean);
  const L = withSt.filter(filt).sort(rank);
  const sortNote = scored ? 'Best match first. Ties go to the listing that describes the most capabilities.'
                          : 'Sorted by how many capabilities each Marketplace listing describes. Not a quality rating.';
  view.innerHTML = `
    <section class="step">
      <button class="back" id="bk">← Back to the questions</button>
      <div class="rhead"><div><div class="tag" style="margin-top:14px">${esc(g.area)}</div><h2>${esc(g.label)}</h2></div>
        <div class="ract"><button class="btn ghost small" id="share">Share shortlist</button><button class="btn ghost small" id="restart">Start over</button></div></div>
      ${chips.length ? `<div class="answers"><span class="rlabel">Your answers</span>${chips.map(c => `<span class="ans" title="${esc(c.q)}">${c.opts.map(o => `<span class="ans-o">${esc(o.label)}<button class="ans-x" data-q="${esc(c.qid)}" data-i="${o.i}" aria-label="Remove ${esc(o.label)}">×</button></span>`).join('<span class="ans-or">or</span>')}</span>`).join('')}</div>` : ''}
      ${!anyAnswers && apps.length < 8 ? `<p class="quiet">Only ${apps.length} app${apps.length === 1 ? '' : 's'} for this process need, so there's nothing to narrow down.</p>` : ''}
      ${S.country ? `<p class="quiet">Showing apps for ${esc(regionName(S.country))} and apps whose listing names no country.${excluded ? ` ${excluded} apps for other countries are hidden.` : ''}</p>` : ''}
      ${shapesPresent.length > 1 ? `<div class="shapes"><span class="rlabel">Show</span>${['all', ...shapesPresent].map(s => `<button class="shape" data-s="${s}" aria-pressed="${S.shape === s}">${s === 'all' ? 'All' : SHAPES[s] || s}</button>`).join('')}</div>` : ''}
      <div class="section">
        <p class="sortnote"><b>${L.length} app${L.length === 1 ? '' : 's'}.</b> ${L.length > 1 ? sortNote : ''}</p>
        ${L.length ? `<div class="apps" id="list">${L.slice(0, 10).map(x => appCard(x.a, scored ? x.st : null)).join('')}</div>${L.length > 10 ? `<button class="btn ghost small showmore">Show ${L.length - 10} more</button>` : ''}`
                   : `<div class="empty">No apps of this kind. Try All.</div>`}
      </div>
      ${side.length ? `<details class="section"><summary class="quiet" style="cursor:pointer">Also covers this as a side feature (${side.length})</summary><div class="apps" style="margin-top:10px">${side.slice(0, 30).map(a => appCard(a, null)).join('')}</div></details>` : ''}
    </section>`;
  S.lastShare = shareHash(gi); history.replaceState(null, '', S.lastShare);
  $('#share').onclick = async () => {
    const b = $('#share'), url = location.href;
    if (navigator.share && matchMedia('(pointer:coarse)').matches){ try { await navigator.share({title: g.label + ' | findmybc.app', url}); } catch(e){} return; }
    try { await navigator.clipboard.writeText(url); b.textContent = 'Link copied'; } catch(e){ b.textContent = 'Copy the address bar'; }
    setTimeout(() => { b.textContent = 'Share shortlist'; }, 2000);
  };
  const more = view.querySelector('.showmore');
  if (more) more.onclick = () => { $('#list').innerHTML = L.map(x => appCard(x.a, scored ? x.st : null)).join(''); more.remove(); wire(); };
  view.querySelectorAll('.ans-x').forEach(b => b.onclick = () => {
    const qid = b.dataset.q, i = +b.dataset.i, rest = (f.answers[qid] || []).filter(x => x !== i);
    if (rest.length) f.answers[qid] = rest; else delete f.answers[qid];
    renderResults(gi);
  });
  view.querySelectorAll('.shape').forEach(b => b.onclick = () => { S.shape = b.dataset.s; renderResults(gi); });
  $('#bk').onclick = () => { const last = f.asked.pop(); if (last) delete f.answers[last]; go('#/need/' + gi); };
  $('#restart').onclick = () => { S.flow[gi] = {asked:[], answers:{}}; S.shape = 'all'; go('#/need/' + gi); };
  wire();
  function wire(){ view.querySelectorAll('.app').forEach(b => b.onclick = () => openApp(+b.dataset.i, gi)); }
}
function iconHtml(a, big){ const ch = esc((a.n || '?').trim()[0] || '?');
  return `<span class="icon">${a.ic ? `<img src="${esc(a.ic)}" alt="" loading="lazy" onerror="this.replaceWith(document.createTextNode('${ch}'))">` : ch}</span>`; }
function appCard(a, st){
  const total = st ? st.yes + st.unk : 0;
  const hits = st ? [...new Set(st.hits)] : [];
  const fit = total ? `<div class="hits"><span class="score${st.unk ? '' : ' all'}">Matches ${st.yes} of ${total}</span>${hits.map(h => `<span class="hit">✓ ${esc(h)}</span>`).join('')}${st.misses.map(m => `<span class="hit unk">Not mentioned: ${esc(m)}</span>`).join('')}</div>` : '';
  return `<button class="app" data-i="${a.i}">${iconHtml(a)}<span style="min-width:0"><div class="nm">${esc(a.n)}</div><div class="pb">${esc(a.p)}${a.t !== 'process' ? ' · ' + esc(SHAPES[a.t] || a.t) : ''}</div><div class="sm">${esc(a.s)}</div>${fit}</span><span class="arrow">›</span></button>`;
}

/* ---------------- app drawer ---------------- */
function openApp(i, gi){
  const a = S.apps.find(x => x.i === i); if (!a) return;
  const st = gi != null ? appStatus(a, gi) : null;
  const qs = gi != null ? allQuestions(gi) : [];
  const f = gi != null ? (S.flow[gi] || {answers:{}}) : {answers:{}};
  const chosen = Object.entries(f.answers).flatMap(([qid, sel]) => { const q = qs.find(x => x.id === qid); return q ? sel.map(k => q.options[k]).filter(o => !o.any && o.re) : []; });
  const marks = c => chosen.some(o => o.re.some(r => r.test(c)));
  const needs = a.g.map(k => S.groups[k].label);
  $('#drawer-body').innerHTML = `
    <div class="dhead">${iconHtml(a, true)}<div><h2 id="d-title">${esc(a.n)}</h2><div class="quiet">${esc(a.p)}</div></div></div>
    <p>${esc(a.s)}</p>
    ${st && (st.hits.length || st.misses.length) ? `<div class="why">${st.hits.length ? `<div><b>The listing mentions:</b> ${esc([...new Set(st.hits)].join('; '))}</div>` : ''}${st.misses.length ? `<div style="margin-top:4px">Not mentioned in the listing: ${esc(st.misses.join('; '))}</div>` : ''}</div>` : ''}
    <dl class="facts">
      <dt>Good for</dt><dd>${esc(needs.join(' · ') || '—')}</dd>
      ${a.x.length ? `<dt>Works with</dt><dd>${esc(a.x.join(', '))}</dd>` : ''}
      <dt>Countries</dt><dd>${own(a, 'c').length ? esc(own(a, 'c').map(regionName).join(', ')) : (a.pub && a.pub.c ? 'Not mentioned on the Marketplace' : 'None mentioned')}</dd>
      ${own(a, 'l').length ? `<dt>Languages</dt><dd>${esc(own(a, 'l').map(langName).join(', '))}</dd>` : ''}
      <dt>Pricing</dt><dd>${esc(PRICING[a.pr] || a.pr)}</dd>
      ${a.req && a.req.length ? `<dt>Requires</dt><dd>${esc(a.req.join(', '))}</dd>` : ''}
    </dl>
    ${pubBlock(a)}
    ${a.k.length ? `<div class="lbl2">What the listing says it does (${a.k.length})</div>
    <ul class="claims">${a.k.map(c => `<li class="${marks(c) ? 'm' : ''}">${esc(c)}</li>`).join('')}</ul>
    <p class="quiet">These lines summarise the Marketplace listing. Check the details there before you decide.</p>` : '<p class="quiet">The listing makes no specific claims we could extract.</p>'}
    <p style="margin-top:24px"><a class="btn" href="${esc(a.u)}" target="_blank" rel="noopener" style="text-decoration:none;display:inline-block">Open on Microsoft Marketplace ↗</a></p>
`;
  $('#drawer-body').querySelectorAll('[data-open]').forEach(b => b.onclick = () => openApp(+b.dataset.open, gi));
  $('#drawer').hidden = false; $('.panel .x').focus();
}
// What the listing itself states, without the publisher's additions.
function own(a, key){ const add = (a.pub && a.pub[key]) || []; return (a[key] || []).filter(x => !add.includes(x)); }
// Facts the publisher added where AppSource falls short. Shown apart, and attributed.
function pubBlock(a){
  const p = a.pub; if (!p) return '';
  const link = i => S.apps[i] ? `<button class="linkish" data-open="${i}">${esc(S.apps[i].n)}</button>` : '';
  const rows = [
    p.c ? `<dt>Also countries</dt><dd>${esc(p.c.map(regionName).join(', '))}</dd>` : '',
    p.l ? `<dt>Also languages</dt><dd>${esc(p.l.map(langName).join(', '))}</dd>` : '',
    p.ext ? `<dt>Extends</dt><dd>${p.ext.map(link).join(', ')}</dd>` : '',
    p.www ? `<dt>Works well with</dt><dd>${p.www.map(link).join(', ')}</dd>` : '',
  ].join('');
  return `<div class="lbl2">Added by the publisher</div><dl class="facts pubfacts">${rows}</dl>`;
}
function closeDrawer(){ $('#drawer').hidden = true; }
document.addEventListener('click', e => { if (e.target.closest('[data-close]')) closeDrawer(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape'){ closeDrawer(); $('#country-pop').hidden = true; } });

/* ---------------- browse ---------------- */
function renderBrowse(){
  const areas = new Map(); S.groups.forEach((g, gi) => { const n = (S.byGroup.get(gi) || []).length; if (!n) return;
    if (!areas.has(g.area)) areas.set(g.area, []); areas.get(g.area).push({g, gi, n}); });
  const ix = a => { const k = AREA_ORDER.indexOf(a); return k < 0 ? 999 : k; };
  const order = [...areas.keys()].sort((a, b) => ix(a) - ix(b));
  view.innerHTML = `
    <section class="step">
      <button class="back" onclick="location.hash='#/'">← Describe your need instead</button>
      <h2>Browse by process need</h2>
      <input class="search" id="bs" placeholder="Search process needs or app names" autocomplete="off">
      <div id="bres"></div>
      <div class="areas" id="areas">${order.map(ar => `<div class="area"><h3>${esc(ar)}</h3><ul>${areas.get(ar).sort((a, b) => b.n - a.n).map(x => `<li><button data-g="${x.gi}">${esc(x.g.label)}<span>${x.n}</span></button></li>`).join('')}</ul></div>`).join('')}</div>
    </section>`;
  view.querySelectorAll('[data-g]').forEach(b => b.onclick = () => startNeed(+b.dataset.g));
  const bs = $('#bs'); bs.focus();
  bs.oninput = () => { const v = bs.value.trim().toLowerCase();
    if (v.length < 2){ $('#bres').innerHTML = ''; $('#areas').hidden = false; return; }
    $('#areas').hidden = true;
    const gs = S.groups.map((g, gi) => ({g, gi})).filter(x => (S.byGroup.get(x.gi) || []).length && (x.g.label + ' ' + x.g.desc).toLowerCase().includes(v)).slice(0, 6);
    const as = S.apps.filter(a => (a.n + ' ' + a.p).toLowerCase().includes(v)).slice(0, 20);
    $('#bres').innerHTML = `${gs.length ? `<div class="section"><div class="sh"><h3>Process needs</h3></div><div class="needs">${gs.map(x => `<button class="need" data-g="${x.gi}"><div><div class="tag">${esc(x.g.area)}</div><div class="lbl">${esc(x.g.label)}</div></div><span class="go">${(S.byGroup.get(x.gi)||[]).length} apps →</span></button>`).join('')}</div></div>` : ''}
      ${as.length ? `<div class="section"><div class="sh"><h3>Apps</h3></div><div class="apps">${as.map(a => appCard(a, null)).join('')}</div></div>` : ''}
      ${!gs.length && !as.length ? '<div class="empty" style="margin-top:16px">Nothing found. Try fewer words.</div>' : ''}`;
    $('#bres').querySelectorAll('[data-g]').forEach(b => b.onclick = () => startNeed(+b.dataset.g));
    $('#bres').querySelectorAll('.app').forEach(b => b.onclick = () => openApp(+b.dataset.i, null));
  };
}

/* ---------------- country ---------------- */
function initCountry(){
  const cs = new Map(); S.apps.forEach(a => a.c.forEach(c => cs.set(c, (cs.get(c) || 0) + 1)));
  const list = [...cs.keys()].filter(c => /^[A-Z]{2}$/.test(c)).map(c => [c, regionName(c)]).sort((a, b) => a[1].localeCompare(b[1]));
  const lab = () => $('#country-label').textContent = S.country ? regionName(S.country) : 'Any';
  lab();
  const pop = $('#country-pop'), inp = $('#country-search'), box = $('#country-list');
  const draw = () => { const v = inp.value.toLowerCase();
    box.innerHTML = `<button data-c="" aria-selected="${!S.country}">Any country</button>` + list.filter(([, n]) => n.toLowerCase().includes(v)).map(([c, n]) => `<button data-c="${c}" aria-selected="${S.country === c}">${esc(n)}</button>`).join('');
    box.querySelectorAll('button').forEach(b => b.onclick = () => { S.country = b.dataset.c; save('af-country', S.country); lab(); pop.hidden = true; route(); }); };
  $('#country-btn').onclick = e => { e.stopPropagation(); pop.hidden = !pop.hidden; if (!pop.hidden){ inp.value = ''; draw(); inp.focus(); } };
  inp.oninput = draw;
  document.addEventListener('click', e => { if (!pop.hidden && !pop.contains(e.target)) pop.hidden = true; });
}

/* ---------------- coverage notice ----------------
   A callout under the "Browse" link in the top bar, shown once per browser until dismissed.
   Remove when the full catalog arrives. */
function coverageNotice(){
  const KEY = 'af-notice-coverage-1';
  const link = document.querySelector('.topnav a[href="#/browse"]');
  if (load(KEY) || !link) return;
  const el = document.createElement('aside');
  el.className = 'notice';
  el.setAttribute('role', 'note');
  el.innerHTML = `<button class="x" aria-label="Close">×</button>
    <b>Missing an app?</b>
    <p>Microsoft's public catalog currently gives us fewer than half of the Business Central apps on Marketplace, so some aren't here yet. We've asked Microsoft for full access.</p>
    <p><b>Publishers:</b> search for your app by name under Browse first. If it doesn't show up, it hasn't reached us yet.</p>
    <div class="acts"><a class="btn small" href="#/browse">Search by name</a><button class="btn ghost small">Got it</button></div>`;
  document.body.appendChild(el);
  const place = () => {
    const r = link.getBoundingClientRect(), w = el.offsetWidth, vw = document.documentElement.clientWidth;
    const left = Math.max(12, Math.min(vw - w - 12, r.left + r.width / 2 - w + 40));
    el.style.top = (r.bottom + 12) + 'px';
    el.style.left = left + 'px';
    el.style.setProperty('--arrow', Math.max(16, Math.min(w - 16, r.left + r.width / 2 - left)) + 'px');
  };
  place();
  window.addEventListener('resize', place);
  const close = () => { save(KEY, '1'); window.removeEventListener('resize', place); el.remove(); };
  el.querySelector('.x').onclick = close;
  el.querySelector('.ghost').onclick = close;
  el.querySelector('a').onclick = close;
  link.addEventListener('click', close);
}

boot().catch(e => { view.innerHTML = `<div class="empty" style="margin-top:40px">Couldn't load the data. Please reload the page.<br><br>${esc(e.message)}</div>`; });
