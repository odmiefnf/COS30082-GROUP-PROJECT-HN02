"""Single-file Light Mint-Emerald web UI for the local Face Attendance server.

Served at http://127.0.0.1:8000 by attendance_app.web with zero external CDN or
Node.js dependencies so it runs offline on any local machine.
"""

HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Face Attendance · COS30082</title>
<style>
  :root {
    --bg: #f1f3f7;
    --card: #ffffff;
    --surface: #f7f9fb;
    --border: #e2e8f0;
    --border-strong: #cbd5e1;
    --ink: #111827;
    --muted: #6b7280;
    --subtle: #9ca3af;
    --mint: #12b896;
    --mint-hover: #0e9f81;
    --mint-soft: #e6f9f4;
    --mint-ink: #0b7a64;
    --amber: #f59e0b;
    --amber-soft: #fffbeb;
    --amber-ink: #92400e;
    --rose: #ef4444;
    --rose-soft: #fef2f2;
    --rose-ink: #991b1b;
    --radius: 16px;
    --radius-sm: 10px;
    --shadow: 0 1px 2px rgba(15, 23, 42, .04), 0 6px 20px rgba(15, 23, 42, .04);
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
    font-size: 14px;
    line-height: 1.45;
    -webkit-font-smoothing: antialiased;
  }
  /* Header */
  .topbar {
    position: sticky; top: 0; z-index: 30;
    background: rgba(255, 255, 255, .94);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--border);
  }
  .topbar-inner {
    max-width: 1380px; margin: 0 auto;
    padding: 12px 24px;
    display: flex; align-items: center; gap: 20px; flex-wrap: wrap;
  }
  .brand { display: flex; align-items: center; gap: 12px; margin-right: auto; }
  .brand-title { font-size: 17px; font-weight: 700; letter-spacing: -.01em; display: flex; align-items: center; gap: 8px; }
  .brand-badge { font-size: 11px; font-weight: 700; color: var(--mint-ink); background: var(--mint-soft); padding: 2px 9px; border-radius: 999px; }
  .brand-sub { font-size: 12px; color: var(--muted); }
  .nav {
    display: flex; align-items: center; gap: 4px;
    background: var(--surface); padding: 4px; border-radius: 12px;
    border: 1px solid var(--border);
  }
  .nav-btn {
    border: 0; background: transparent; color: var(--muted);
    font: 600 13px/1 inherit; padding: 8px 16px; border-radius: 9px;
    cursor: pointer; transition: all .15s ease;
  }
  .nav-btn:hover { color: var(--ink); }
  .nav-btn.active { background: var(--card); color: var(--mint-ink); box-shadow: 0 1px 3px rgba(15,23,42,.08); }
  .status-pill {
    display: inline-flex; align-items: center; gap: 8px;
    background: var(--surface); border: 1px solid var(--border);
    padding: 7px 14px; border-radius: 999px;
    font-size: 12px; font-weight: 600; color: var(--ink);
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--subtle); flex: 0 0 auto; }
  .dot.ok { background: var(--mint); box-shadow: 0 0 0 3px var(--mint-soft); }
  .dot.warn { background: var(--amber); box-shadow: 0 0 0 3px var(--amber-soft); }
  .dot.err { background: var(--rose); box-shadow: 0 0 0 3px var(--rose-soft); }

  /* Layout */
  .wrap { max-width: 1380px; margin: 0 auto; padding: 18px 24px 28px; }
  .page { display: none; }
  .page.active { display: block; }
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius); box-shadow: var(--shadow);
    padding: 18px 20px;
  }
  .card-title { font-size: 15px; font-weight: 700; margin: 0; }
  .caption { font-size: 11px; font-weight: 700; letter-spacing: .05em; color: var(--muted); text-transform: uppercase; }

  /* Toolbar */
  .toolbar {
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
    padding: 13px 18px; margin-bottom: 16px;
  }
  .field-inline { display: flex; align-items: center; gap: 8px; }
  .divider-v { width: 1px; height: 26px; background: var(--border); margin: 0 4px; }
  input[type="text"], select {
    height: 38px; padding: 0 12px; border-radius: var(--radius-sm);
    border: 1px solid var(--border-strong); background: var(--card);
    color: var(--ink); font: 400 13.5px inherit; outline: none;
    transition: border-color .15s, box-shadow .15s;
  }
  input[type="text"]:focus, select:focus {
    border-color: var(--mint); box-shadow: 0 0 0 3px rgba(18, 184, 150, .15);
  }
  input[type="text"]:disabled, select:disabled { background: var(--surface); color: var(--subtle); }
  .btn {
    height: 38px; padding: 0 16px; border-radius: var(--radius-sm);
    border: 1px solid transparent; font: 600 13px inherit;
    cursor: pointer; display: inline-flex; align-items: center; justify-content: center; gap: 7px;
    transition: all .15s ease; white-space: nowrap;
  }
  .btn:disabled { opacity: .45; cursor: not-allowed; }
  .btn-primary { background: var(--mint); color: #fff; }
  .btn-primary:hover:not(:disabled) { background: var(--mint-hover); transform: translateY(-1px); }
  .btn-secondary { background: var(--surface); color: var(--ink); border-color: var(--border); }
  .btn-secondary:hover:not(:disabled) { background: #e8edf3; }
  .btn-danger { background: var(--rose-soft); color: var(--rose-ink); border-color: #fecaca; }
  .btn-danger:hover:not(:disabled) { background: #fee2e2; }
  .btn-ghost { background: transparent; color: var(--mint-ink); padding: 0 10px; }
  .btn-ghost:hover:not(:disabled) { background: var(--mint-soft); }
  .btn-sm { height: 32px; padding: 0 12px; font-size: 12px; border-radius: 8px; }
  .btn-full { width: 100%; }

  /* Live workspace grid */
  .live-grid { display: grid; grid-template-columns: minmax(0, 1.85fr) minmax(330px, 1fr); gap: 16px; align-items: start; }
  @media (max-width: 980px) { .live-grid { grid-template-columns: 1fr; } }

  /* Camera card */
  .cam-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
  .fps-tag { font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; }
  .viewport {
    position: relative; width: 100%; aspect-ratio: 16 / 10; min-height: 300px; max-height: 490px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; overflow: hidden;
    display: flex; align-items: center; justify-content: center;
  }
  .viewport.live { background: #0f172a; border-color: #1e293b; }
  .viewport img { width: 100%; height: 100%; object-fit: contain; display: none; }
  .viewport.live img { display: block; }
  .viewport.live .placeholder { display: none; }
  .hud-pill {
    position: absolute; top: 12px; left: 12px; z-index: 5;
    display: none; align-items: center; gap: 7px;
    padding: 6px 12px; border-radius: 999px;
    background: rgba(15, 23, 42, .74); backdrop-filter: blur(6px);
    color: #fff; font-size: 12.5px; font-weight: 600;
    border: 1px solid rgba(255,255,255,.14);
  }
  .viewport.live .hud-pill.visible { display: inline-flex; }
  .hud-pill.ok { background: rgba(6, 95, 70, .86); border-color: rgba(52, 211, 153, .45); }
  .hud-pill.err { background: rgba(153, 27, 27, .86); border-color: rgba(252, 165, 165, .45); }
  .hud-pill.warn { background: rgba(146, 64, 14, .86); border-color: rgba(253, 230, 138, .45); }
  .step-strip { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 12px 0 8px; }
  .step-dot {
    text-align: center; font-size: 11.5px; font-weight: 700; padding: 9px 8px;
    border-radius: 10px; background: var(--surface); color: var(--muted);
    border: 1px solid var(--border); transition: all .2s;
    display: flex; flex-direction: column; align-items: center; gap: 3px;
  }
  .step-dot .step-num { font-size: 10.5px; font-weight: 800; color: var(--subtle); text-transform: uppercase; letter-spacing: .04em; }
  .step-dot.current { background: #f0fdf9; color: var(--mint-ink); border-color: var(--mint); box-shadow: 0 0 0 2px rgba(18,184,150,.12); }
  .step-dot.current .step-num { color: var(--mint-ink); }
  .step-dot.done { background: var(--mint-soft); color: var(--mint-ink); border-color: #a7f3d0; }
  .step-dot.done .step-num { color: var(--mint-ink); }

  /* Dedicated Register page */
  .register-grid { display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(350px, 1fr); gap: 16px; align-items: start; }
  @media (max-width: 980px) { .register-grid { grid-template-columns: 1fr; } }
  .guide-oval {
    position: absolute; inset: 0; pointer-events: none; z-index: 4;
    display: none; align-items: center; justify-content: center;
  }
  .viewport.live .guide-oval { display: flex; }
  .guide-oval-ring {
    width: 36%; max-width: 220px; aspect-ratio: 3 / 4;
    border-radius: 50%; border: 2px dashed rgba(255, 255, 255, .55);
    box-shadow: 0 0 0 9999px rgba(15, 23, 42, .18);
    transition: border-color .2s, box-shadow .2s;
  }
  .guide-oval.is-ready .guide-oval-ring {
    border-style: solid; border-color: rgba(52, 211, 153, .92);
    box-shadow: 0 0 0 9999px rgba(15, 23, 42, .12), 0 0 0 4px rgba(18, 184, 150, .25);
  }
  .pose-pill {
    position: absolute; bottom: 14px; left: 50%; transform: translateX(-50%); z-index: 5;
    display: none; align-items: center; gap: 8px;
    padding: 7px 14px; border-radius: 999px;
    background: rgba(15, 23, 42, .80); backdrop-filter: blur(6px);
    color: #fff; font-size: 12.5px; font-weight: 600; white-space: nowrap;
    border: 1px solid rgba(255, 255, 255, .16);
  }
  .viewport.live .pose-pill { display: inline-flex; }
  .pose-pill.ok { background: rgba(6, 95, 70, .88); border-color: rgba(52, 211, 153, .45); }
  .pose-pill.warn { background: rgba(146, 64, 14, .88); border-color: rgba(253, 230, 138, .45); }
  .pose-pill.err { background: rgba(153, 27, 27, .88); border-color: rgba(252, 165, 165, .45); }
  .ready-callout {
    display: flex; align-items: center; gap: 8px;
    padding: 9px 12px; border-radius: 10px; margin-bottom: 12px;
    background: var(--surface); border: 1px solid var(--border);
    font-size: 12.5px; font-weight: 600; color: var(--muted);
  }
  .ready-callout.ok { background: var(--mint-soft); border-color: #a7f3d0; color: var(--mint-ink); }
  .ready-callout.warn { background: var(--amber-soft); border-color: #fde68a; color: var(--amber-ink); }
  .ready-callout.err { background: var(--rose-soft); border-color: #fecaca; color: var(--rose-ink); }
  .alt-option {
    display: flex; align-items: center; justify-content: space-between; gap: 12px;
    padding: 10px 0; border-bottom: 1px solid #f1f5f9;
  }
  .alt-option:last-of-type { border-bottom: 0; }
  .alt-option-title { font-size: 13px; font-weight: 700; color: var(--ink); }
  .alt-option-sub { font-size: 12px; color: var(--muted); margin-top: 2px; }
  .cam-flash {
    position: absolute; inset: 0; z-index: 6; pointer-events: none;
    background: rgba(255, 255, 255, .65); opacity: 0;
  }
  .cam-flash.pop { animation: camFlash .26s ease-out; }
  @keyframes camFlash { 0% { opacity: .85; } 100% { opacity: 0; } }
  .capture-bar {
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
    margin-top: 12px;
  }
  .capture-count {
    font-size: 11.5px; font-weight: 700; padding: 2px 8px; border-radius: 999px;
    background: rgba(255, 255, 255, .24); font-variant-numeric: tabular-nums;
  }
  .shoot-hint { font-size: 12.5px; color: var(--muted); flex: 1; min-width: 170px; }
  .shoot-hint.warn { color: var(--amber-ink); font-weight: 600; }
  .shoot-hint.err { color: var(--rose-ink); font-weight: 600; }
  .shots-head { display: flex; align-items: center; justify-content: space-between; margin: 12px 0 8px; }
  .shots-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; }
  .shot-card {
    position: relative; aspect-ratio: 3 / 4; border-radius: 10px; overflow: hidden;
    background: var(--surface); border: 1.5px dashed var(--border-strong);
    display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 2px;
    text-align: center; padding: 4px; transition: all .18s ease;
  }
  .shot-card.next { border-style: solid; border-color: var(--mint); background: #f0fdf9; box-shadow: 0 0 0 2px rgba(18,184,150,.13); }
  .shot-card.filled { border-style: solid; border-color: #a7f3d0; background: #0f172a; padding: 0; }
  .shot-card img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .shot-num { font-size: 13px; font-weight: 800; color: var(--muted); font-variant-numeric: tabular-nums; }
  .shot-card.next .shot-num { color: var(--mint-ink); }
  .shot-lbl { font-size: 10.5px; font-weight: 600; color: var(--subtle); }
  .shot-card.next .shot-lbl { color: var(--mint-ink); }
  .shot-badge {
    position: absolute; left: 0; right: 0; bottom: 0;
    background: linear-gradient(transparent, rgba(15, 23, 42, .82));
    color: #fff; font-size: 10px; font-weight: 700; padding: 8px 4px 3px; text-align: center;
  }
  .shot-rm {
    position: absolute; top: 4px; right: 4px; width: 20px; height: 20px;
    border-radius: 50%; border: 0; background: rgba(15, 23, 42, .75); color: #fff;
    font: 700 12px/1 inherit; cursor: pointer; display: flex; align-items: center; justify-content: center;
  }
  .shot-rm:hover { background: var(--rose); }
  .shot-card.not-live { border-color: var(--rose); box-shadow: 0 0 0 2px rgba(239,68,68,.25); }
  .shot-flag {
    position: absolute; top: 4px; left: 4px; padding: 2px 5px; border-radius: 6px;
    background: var(--rose); color: #fff; font-size: 9.5px; font-weight: 700;
  }
  .dup-box {
    display: none; margin: 12px 0; padding: 12px 13px; border-radius: 12px;
    background: var(--amber-soft); border: 1px solid #fde68a; font-size: 12.5px; color: var(--amber-ink);
  }
  .dup-box.open { display: block; }
  .dup-box input[type="text"] { height: 34px; font-size: 12.5px; margin-top: 8px; width: 100%; }
  .btn-auto-on { background: var(--mint-soft); color: var(--mint-ink); border-color: var(--mint); }
  .match-row { display: grid; grid-template-columns: 1fr auto; gap: 6px; align-items: center; margin-bottom: 6px; font-size: 12px; }
  .match-bar { grid-column: 1 / -1; height: 5px; background: #edf1f5; border-radius: 999px; overflow: hidden; }
  .match-fill { height: 100%; border-radius: 999px; background: #94a3b8; transition: width .2s; }
  .match-fill.pass { background: var(--mint); }
  .act-item { display: flex; align-items: flex-start; gap: 8px; padding: 7px 4px; border-bottom: 1px solid #f1f5f9; font-size: 12.5px; }
  .act-item:last-child { border-bottom: 0; }
  .act-time { font-size: 11px; color: var(--subtle); font-variant-numeric: tabular-nums; flex: 0 0 auto; margin-top: 2px; }
  .kpi-strip-5 { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 16px; }
  @media (max-width: 820px) { .kpi-strip-5 { grid-template-columns: repeat(2, 1fr); } }
  .settings-grid { display: grid; grid-template-columns: 1fr 1.3fr; gap: 16px; margin-bottom: 16px; }
  @media (max-width: 880px) { .settings-grid { grid-template-columns: 1fr; } }
  .setting-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 10px 0; border-bottom: 1px solid #f1f5f9; }
  .setting-row:last-child { border-bottom: 0; }
  .switch { display: inline-flex; align-items: center; gap: 8px; cursor: pointer; font-weight: 600; font-size: 12.5px; }
  .switch input { width: 17px; height: 17px; accent-color: var(--mint); cursor: pointer; }
  .focus-dim { opacity: .42; transition: opacity .2s; }
  .placeholder {
    text-align: center; padding: 24px; max-width: 380px;
    display: flex; flex-direction: column; align-items: center; gap: 10px;
  }
  .placeholder h3 { margin: 4px 0 0; font-size: 17px; font-weight: 700; color: var(--ink); }
  .placeholder p { margin: 0; font-size: 13px; color: var(--muted); }
  .scan-wave { animation: scanPulse 2.8s ease-in-out infinite; transform-origin: center; }
  @keyframes scanPulse { 0%, 100% { transform: translateY(-8px); opacity: .75; } 50% { transform: translateY(8px); opacity: 1; } }

  /* Quality chips */
  .chips { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 12px; }
  @media (max-width: 640px) { .chips { grid-template-columns: repeat(2, 1fr); } }
  .chip {
    display: flex; align-items: center; gap: 7px;
    padding: 7px 11px; border-radius: 10px;
    background: var(--surface); border: 1px solid var(--border);
    font-size: 12px; font-weight: 600; color: var(--muted);
    transition: all .2s ease;
  }
  .chip .dot { width: 7px; height: 7px; box-shadow: none; }
  .chip.ok { background: var(--mint-soft); border-color: #a7f3d0; color: var(--mint-ink); }
  .chip.ok .dot { background: var(--mint); }
  .chip.warn { background: var(--amber-soft); border-color: #fde68a; color: var(--amber-ink); }
  .chip.warn .dot { background: var(--amber); }
  .chip.err { background: var(--rose-soft); border-color: #fecaca; color: var(--rose-ink); }
  .chip.err .dot { background: var(--rose); }

  /* Progress + feedback banner */
  .progress-track {
    height: 6px; background: #e5eaf0; border-radius: 999px; overflow: hidden; margin-top: 12px;
  }
  .progress-fill {
    height: 100%; width: 0%; background: var(--mint); border-radius: 999px;
    transition: width .25s ease, background-color .25s ease;
  }
  .banner {
    margin-top: 10px; padding: 12px 15px; border-radius: 12px;
    background: var(--surface); border-left: 4px solid var(--border-strong);
    transition: background-color .2s, border-color .2s;
  }
  .banner.info { background: #eefbf7; border-left-color: var(--mint); }
  .banner.success { background: #ecfdf5; border-left-color: #10b981; }
  .banner.warning { background: var(--amber-soft); border-left-color: var(--amber); }
  .banner.danger { background: var(--rose-soft); border-left-color: var(--rose); }
  .banner-title { font-size: 14px; font-weight: 700; color: var(--ink); }
  .banner.success .banner-title { color: #065f46; }
  .banner.info .banner-title { color: var(--mint-ink); }
  .banner.warning .banner-title { color: var(--amber-ink); }
  .banner.danger .banner-title { color: var(--rose-ink); }
  .banner-sub { font-size: 12.5px; color: var(--muted); margin-top: 2px; }

  /* 3 KPI cards */
  .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 14px; }
  @media (max-width: 640px) { .stats-grid { grid-template-columns: 1fr; } }
  .stat-card {
    background: var(--card); border: 1px solid var(--border); border-radius: 14px;
    box-shadow: var(--shadow); overflow: hidden;
  }
  .stat-strip { height: 4px; background: var(--border-strong); transition: background-color .2s; }
  .stat-body { padding: 12px 15px; }
  .stat-val {
    font-size: 15px; font-weight: 700; color: var(--subtle); margin-top: 5px;
    display: flex; align-items: center; gap: 6px; word-break: break-word; min-height: 24px;
  }
  .stat-card.ok .stat-strip { background: #10b981; }
  .stat-card.ok .stat-val { color: #065f46; }
  .stat-card.info .stat-strip { background: var(--mint); }
  .stat-card.info .stat-val { color: var(--mint-ink); }
  .stat-card.warn .stat-strip { background: var(--amber); }
  .stat-card.warn .stat-val { color: var(--amber-ink); }
  .stat-card.err .stat-strip { background: var(--rose); }
  .stat-card.err .stat-val { color: var(--rose-ink); }

  /* Right sidebar */
  .sidebar { display: flex; flex-direction: column; gap: 14px; }
  .seg-tabs {
    display: grid; grid-template-columns: 1fr 1fr; gap: 4px;
    background: var(--surface); border: 1px solid var(--border);
    padding: 4px; border-radius: 11px; margin-bottom: 14px;
  }
  .seg-btn {
    border: 0; background: transparent; color: var(--muted);
    font: 600 12.5px inherit; padding: 7px 10px; border-radius: 8px; cursor: pointer;
    transition: all .15s;
  }
  .seg-btn.active { background: var(--card); color: var(--mint-ink); box-shadow: 0 1px 2px rgba(15,23,42,.07); }
  .sub-panel { display: none; }
  .sub-panel.active { display: block; }
  .form-group { margin-bottom: 10px; }
  .form-group label { display: block; margin-bottom: 5px; }
  .btn-row { display: flex; gap: 8px; margin-top: 8px; }
  .btn-row > * { flex: 1; }
  .divider-h { height: 1px; background: var(--border); margin: 14px 0 10px; }
  .meta-row { display: flex; align-items: center; justify-content: space-between; font-size: 12.5px; color: var(--muted); }

  /* Telemetry bars */
  .meter { margin-bottom: 10px; }
  .meter-head { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; }
  .meter-head strong { color: var(--ink); }
  .meter-head span { color: var(--muted); font-variant-numeric: tabular-nums; }
  .meter-track { height: 7px; background: #edf1f5; border-radius: 999px; overflow: hidden; }
  .meter-fill { height: 100%; width: 0%; background: var(--mint); border-radius: 999px; transition: width .2s ease; }
  .emo-row { display: grid; grid-template-columns: 68px 1fr 38px; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 12px; }
  .emo-name { color: var(--muted); }
  .emo-pct { text-align: right; color: var(--subtle); font-variant-numeric: tabular-nums; }
  .emo-row.top .emo-name, .emo-row.top .emo-pct { color: var(--mint-ink); font-weight: 700; }
  .emo-track { height: 7px; background: #edf1f5; border-radius: 999px; overflow: hidden; }
  .emo-fill { height: 100%; width: 0%; background: #a7d9cd; border-radius: 999px; transition: width .2s ease; }
  .emo-row.top .emo-fill { background: var(--mint); }

  /* Tables */
  .table-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
  .count-pill { background: var(--mint-soft); color: var(--mint-ink); font-size: 12px; font-weight: 700; padding: 3px 10px; border-radius: 999px; }
  .table-wrap { border: 1px solid var(--border); border-radius: 11px; overflow: auto; max-height: 290px; }
  .table-wrap.tall { max-height: 460px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th {
    position: sticky; top: 0; background: var(--surface); color: var(--muted);
    font-size: 11px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;
    text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--border);
  }
  td { padding: 9px 12px; border-bottom: 1px solid #f1f5f9; color: var(--ink); }
  tr:last-child td { border-bottom: 0; }
  tbody tr { cursor: pointer; transition: background-color .1s; }
  tbody tr:hover { background: #f8fbfa; }
  tbody tr.selected { background: var(--mint-soft); }
  .tag { display: inline-block; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 999px; }
  .tag-auto { background: var(--mint-soft); color: var(--mint-ink); }
  .tag-manual { background: var(--amber-soft); color: var(--amber-ink); }
  .tag-legacy { background: var(--surface); color: var(--muted); }
  .empty-row { text-align: center; color: var(--subtle); padding: 22px 12px; cursor: default; }

  /* Profiles / History / Diagnostics layouts */
  .split-2 { display: grid; grid-template-columns: 1.15fr 1fr; gap: 16px; }
  @media (max-width: 900px) { .split-2 { grid-template-columns: 1fr; } }
  .kpi-strip { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }
  @media (max-width: 720px) { .kpi-strip { grid-template-columns: repeat(2, 1fr); } }
  .kpi-box { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 14px 16px; box-shadow: var(--shadow); }
  .kpi-num { font-size: 24px; font-weight: 700; color: var(--ink); margin-top: 4px; }
  .diag-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 16px; }
  @media (max-width: 860px) { .diag-grid { grid-template-columns: 1fr; } }
  .diag-card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 14px 16px; box-shadow: var(--shadow); }
  .diag-top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
  .diag-detail { font-size: 12px; color: var(--muted); word-break: break-all; }
  .mono-log {
    font-family: Consolas, "Courier New", monospace; font-size: 12.5px;
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    padding: 14px; white-space: pre-wrap; color: var(--ink); min-height: 120px; margin: 0;
  }
  .footer-note { margin-top: 14px; font-size: 11.5px; color: var(--subtle); }

  /* Modals & Toasts */
  .backdrop {
    position: fixed; inset: 0; z-index: 50;
    background: rgba(15, 23, 42, .45); backdrop-filter: blur(3px);
    display: none; align-items: center; justify-content: center; padding: 20px;
  }
  .backdrop.open { display: flex; }
  .modal {
    background: var(--card); border-radius: 18px; width: 100%; max-width: 540px;
    box-shadow: 0 20px 50px rgba(15, 23, 42, .22); padding: 22px 24px;
    max-height: 90vh; overflow-y: auto;
  }
  .modal h2 { margin: 0 0 6px; font-size: 18px; }
  .modal-sub { margin: 0 0 14px; font-size: 13px; color: var(--muted); }
  .crop-row { display: flex; gap: 14px; align-items: center; margin-bottom: 14px; }
  .crop-img { width: 136px; height: 106px; object-fit: cover; border-radius: 12px; border: 1px solid var(--border); background: var(--surface); }
  .countdown-pill {
    display: inline-block; background: var(--mint-soft); color: var(--mint-ink);
    font-weight: 700; font-size: 12.5px; padding: 6px 12px; border-radius: 9px; margin-bottom: 10px;
  }
  .countdown-pill.err { background: var(--rose-soft); color: var(--rose-ink); }
  .check-line { display: flex; align-items: flex-start; gap: 9px; font-size: 13px; margin: 12px 0; cursor: pointer; }
  .check-line input { margin-top: 3px; accent-color: var(--mint); }
  .err-text { color: var(--rose-ink); font-size: 12.5px; font-weight: 600; min-height: 18px; margin-bottom: 6px; }
  .toast {
    position: fixed; right: 22px; bottom: 22px; z-index: 60;
    background: #111827; color: #fff; padding: 11px 16px; border-radius: 12px;
    font-size: 13px; font-weight: 500; box-shadow: 0 10px 30px rgba(15,23,42,.25);
    opacity: 0; transform: translateY(10px); pointer-events: none; transition: all .2s ease;
    max-width: 420px;
  }
  .toast.show { opacity: 1; transform: translateY(0); }
</style>
</head>
<body>

<header class="topbar">
  <div class="topbar-inner">
    <div class="brand">
      <svg width="38" height="38" viewBox="0 0 40 40" fill="none" aria-hidden="true">
        <polygon points="20,2 36,11 36,29 20,38 4,29 4,11" fill="#e6f9f4" stroke="#12b896" stroke-width="2"/>
        <circle cx="20" cy="16.5" r="5" stroke="#0b7a64" stroke-width="1.8"/>
        <path d="M12.5 29c1.8-4 4.4-5.8 7.5-5.8s5.7 1.8 7.5 5.8" stroke="#0b7a64" stroke-width="1.8" stroke-linecap="round"/>
        <path d="M9 20c4-2.2 18-2.2 22 0" stroke="#12b896" stroke-width="1.8" stroke-linecap="round"/>
      </svg>
      <div>
        <div class="brand-title">Face Attendance <span class="brand-badge">Live Verification</span></div>
        <div class="brand-sub">Identity verification · Anti-spoofing liveness · Facial expression</div>
      </div>
    </div>
    <nav class="nav" aria-label="Main sections">
      <button class="nav-btn active" data-page="live">Live</button>
      <button class="nav-btn" data-page="register">Register</button>
      <button class="nav-btn" data-page="profiles">Profiles</button>
      <button class="nav-btn" data-page="attendance">Attendance</button>
      <button class="nav-btn" data-page="diagnostics">Settings &amp; Info</button>
    </nav>
    <div class="status-pill"><span class="dot" id="hdrDot"></span><span id="hdrState">Loading models…</span></div>
  </div>
</header>

<main class="wrap">
  <!-- PAGE 1: LIVE -->
  <section class="page active" id="page-live">
    <div class="card toolbar">
      <div class="field-inline">
        <span class="caption">Session</span>
        <input type="text" id="sessionInput" style="width:190px" placeholder="YYYY-MM-DD">
      </div>
      <button class="btn btn-primary" id="btnStart" disabled>▶ Start camera</button>
      <button class="btn btn-danger" id="btnStop" disabled>■ Stop</button>
      <div style="margin-right:auto;"></div>
      <button class="btn btn-ghost" onclick="switchPage('register')">+ Register person</button>
      <button class="btn btn-ghost" onclick="switchPage('attendance')">History / CSV →</button>
    </div>

    <div class="live-grid">
      <div>
        <div class="card">
          <div class="cam-head">
            <div style="display:flex;align-items:center;gap:10px;">
              <h2 class="card-title">Live camera</h2>
              <select id="focusSelect" style="height:28px;font-size:12px;padding:0 8px;" title="Highlight one model for demo">
                <option value="all">Focus: All 3 models</option>
                <option value="liveness">Focus: Liveness (PAD)</option>
                <option value="identity">Focus: Identity</option>
                <option value="expression">Focus: Expression</option>
              </select>
            </div>
            <span class="fps-tag" id="fpsLabel">FPS not measured</span>
          </div>
          <div class="viewport" id="viewport">
            <div class="hud-pill" id="hudPill"><span class="dot"></span><span id="hudText">Scanning…</span></div>
            <img id="videoFeed" alt="Live camera stream">
            <div class="placeholder">
              <svg width="150" height="140" viewBox="0 0 150 140" fill="none" aria-hidden="true">
                <path d="M18 34V18h16M132 34V18h-16M18 106v16h16M132 106v16h-16" stroke="#cbd5e1" stroke-width="2.5" stroke-linecap="round"/>
                <polygon points="75,20 116,44 116,94 75,118 34,94 34,44" fill="#ffffff" stroke="#94a3b8" stroke-width="2"/>
                <ellipse cx="75" cy="60" rx="18" ry="22" stroke="#475569" stroke-width="2"/>
                <path d="M47 100c5-12 15-18 28-18s23 6 28 18" stroke="#475569" stroke-width="2" stroke-linecap="round"/>
                <path class="scan-wave" d="M26 70c18-10 80-10 98 0" stroke="#12b896" stroke-width="2.8" stroke-linecap="round"/>
              </svg>
              <h3>Live face detection</h3>
              <p>Click <strong>Start camera</strong> to scan your face and verify attendance.</p>
            </div>
          </div>

          <div class="chips">
            <div class="chip" id="chip-face"><span class="dot"></span>Single face</div>
            <div class="chip" id="chip-lighting"><span class="dot"></span>Good lighting</div>
            <div class="chip" id="chip-focus"><span class="dot"></span>Sharp focus</div>
            <div class="chip" id="chip-liveness"><span class="dot"></span>Liveness</div>
          </div>

          <div class="progress-track"><div class="progress-fill" id="scanFill"></div></div>

          <div class="banner" id="feedbackBanner">
            <div class="banner-title" id="feedbackTitle">Click Start camera when you are ready</div>
            <div class="banner-sub" id="feedbackSub">Keep one face centered in good lighting. Identity, liveness and expression update live.</div>
          </div>
        </div>

        <div class="stats-grid">
          <div class="stat-card" id="cardIdentity">
            <div class="stat-strip"></div>
            <div class="stat-body">
              <div class="caption">Identity</div>
              <div class="stat-val" id="valIdentity">—</div>
            </div>
          </div>
          <div class="stat-card" id="cardLiveness">
            <div class="stat-strip"></div>
            <div class="stat-body">
              <div class="caption">Liveness</div>
              <div class="stat-val" id="valLiveness">—</div>
            </div>
          </div>
          <div class="stat-card" id="cardExpression">
            <div class="stat-strip"></div>
            <div class="stat-body">
              <div class="caption">Expression</div>
              <div class="stat-val" id="valExpression">—</div>
            </div>
          </div>
        </div>
      </div>

      <aside class="sidebar">
        <div class="card">
          <div class="table-head" style="margin-bottom:10px;">
            <h2 class="card-title">Live model scores</h2>
            <span id="liveProfileCount" style="font-size:12px;color:var(--muted);">0 profiles</span>
          </div>
          <div class="caption" style="margin-bottom:8px;">Liveness (PAD) &amp; Identity</div>
          <div class="meter">
            <div class="meter-head"><strong>Liveness score</strong><span id="padScoreTxt">Waiting for frame</span></div>
            <div class="meter-track"><div class="meter-fill" id="padBar"></div></div>
          </div>
          <div class="meter">
            <div class="meter-head"><strong>Identity cosine</strong><span id="idScoreTxt">Waiting for frame</span></div>
            <div class="meter-track"><div class="meter-fill" id="idBar"></div></div>
          </div>
          <div class="caption" style="margin:10px 0 6px;">Top-3 Gallery Matches</div>
          <div id="topMatches" style="font-size:12px;color:var(--subtle);margin-bottom:6px;">No live identity probe yet</div>
          <div class="divider-h"></div>
          <div class="caption" style="margin-bottom:8px;">Expressions (FER2013 · Display only)</div>
          <div id="emotionBars"></div>
          <div class="divider-h"></div>
          <div class="btn-row" style="margin-top:0;">
            <button class="btn btn-secondary btn-sm" id="btnReview" disabled>Manual review</button>
            <button class="btn btn-ghost btn-sm" id="btnLiveGoRegister" onclick="switchPage('register')">+ Register face →</button>
          </div>
        </div>

        <div class="card">
          <div class="table-head" style="margin-bottom:8px;">
            <div style="display:flex;gap:6px;">
              <button class="btn btn-sm btn-secondary" id="btnLogPresent" onclick="switchSideList('present')">Present</button>
              <button class="btn btn-sm btn-ghost" id="btnLogActivity" onclick="switchSideList('activity')">Activity <span id="spoofBadge"></span></button>
            </div>
            <span class="count-pill" id="presentCount">0 present</span>
          </div>
          <div id="sidePresent">
            <div class="table-wrap">
              <table>
                <thead><tr><th>Name / Student ID</th><th>Time</th><th>Source</th></tr></thead>
                <tbody id="sessionBody"><tr><td colspan="3" class="empty-row">No attendance recorded yet</td></tr></tbody>
              </table>
            </div>
            <div style="font-size:11.5px;color:var(--subtle);margin-top:8px;">One record per person per session · Click a row for details</div>
          </div>
          <div id="sideActivity" style="display:none;">
            <div id="activityList" style="max-height:250px;overflow-y:auto;">
              <div class="empty-row">No check-ins or blocked spoof events yet</div>
            </div>
          </div>
        </div>
      </aside>
    </div>
    <div class="footer-note">Stored locally · Camera images are not saved · Expressions are display only · Educational prototype (COS30082)</div>
  </section>

  <!-- PAGE 2: REGISTER (HYBRID GUIDED LIVE ENROLLMENT) -->
  <section class="page" id="page-register">
    <div class="register-grid">
      <div class="card">
        <div class="cam-head">
          <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
            <h2 class="card-title">Enrollment camera</h2>
            <span class="fps-tag" id="regFpsLabel">Camera off</span>
          </div>
          <div style="display:flex;gap:8px;">
            <button class="btn btn-primary btn-sm" id="btnRegStart" disabled>▶ Start camera</button>
            <button class="btn btn-danger btn-sm" id="btnRegStop" disabled>■ Stop</button>
          </div>
        </div>

        <div class="viewport" id="regViewport">
          <div class="cam-flash" id="regFlash"></div>
          <div class="guide-oval" id="regGuideOval" aria-hidden="true"><div class="guide-oval-ring"></div></div>
          <div class="pose-pill" id="regPosePill"><span id="regPoseText">1/10 · Look straight at the camera</span></div>
          <div class="placeholder">
            <svg width="150" height="140" viewBox="0 0 150 140" fill="none" aria-hidden="true">
              <path d="M18 34V18h16M132 34V18h-16M18 106v16h16M132 106v16h-16" stroke="#cbd5e1" stroke-width="2.5" stroke-linecap="round"/>
              <ellipse cx="75" cy="70" rx="34" ry="44" fill="#ffffff" stroke="#12b896" stroke-width="2" stroke-dasharray="6 5"/>
              <circle cx="75" cy="58" r="13" stroke="#475569" stroke-width="2"/>
              <path d="M52 94c4.5-10 13-15 23-15s18.5 5 23 15" stroke="#475569" stroke-width="2" stroke-linecap="round"/>
            </svg>
            <h3>Guided live registration</h3>
            <p>Click <strong>Start camera</strong>, follow the 10 pose hints, and capture live shots (or press <strong>Space</strong>).</p>
          </div>
        </div>

        <div class="capture-bar">
          <button type="button" class="btn btn-primary" id="btnTakeShot" disabled>
            📷 Capture shot <span class="capture-count" id="shotCountBadge">0/10</span>
          </button>
          <button type="button" class="btn btn-secondary" id="btnAutoCapture" disabled title="Automatically capture shots whenever all live checks pass">
            ⚡ Auto-capture
          </button>
          <span class="shoot-hint" id="regShootHint">Start the camera to begin capturing live shots.</span>
        </div>

        <div class="chips" aria-label="Registration readiness checks">
          <div class="chip" id="reg-chip-face"><span class="dot"></span>Single face</div>
          <div class="chip" id="reg-chip-lighting"><span class="dot"></span>Good lighting</div>
          <div class="chip" id="reg-chip-focus"><span class="dot"></span>Sharp focus</div>
          <div class="chip" id="reg-chip-liveness"><span class="dot"></span>Liveness</div>
        </div>
      </div>

      <aside class="sidebar">
        <div class="card">
          <h2 class="card-title" id="regTitle" style="font-size:17px;margin-bottom:4px;">Register a person</h2>
          <p style="margin:0 0 12px;font-size:12.5px;color:var(--muted);">
            Capture 10 guided shots (2 straight, 2 left, 2 right, 2 tilt up, 2 tilt down). Shots that fail liveness are marked Not live and are not saved. Previews stay in RAM only — only embeddings are saved.
          </p>
          <div class="seg-tabs" role="tablist" aria-label="Registration type">
            <button type="button" class="seg-btn active" id="regModeNew" role="tab">New person</button>
            <button type="button" class="seg-btn" id="regModeLook" role="tab">Add appearance</button>
          </div>

          <form id="registerForm" autocomplete="off">
            <div id="regNewFields">
              <div class="form-group">
                <label class="caption" for="regFullName">Full name</label>
                <input type="text" id="regFullName" style="width:100%" placeholder="e.g. Nguyen Van A" required>
              </div>
              <div class="form-group">
                <label class="caption" for="regStudentId">Student ID <span style="font-weight:500;text-transform:none;letter-spacing:0;color:var(--subtle);">(optional)</span></label>
                <input type="text" id="regStudentId" style="width:100%" placeholder="e.g. 104123456">
              </div>
            </div>
            <div id="regLookFields" style="display:none;">
              <div class="form-group">
                <label class="caption" for="regLookProfile">Existing profile</label>
                <select id="regLookProfile" style="width:100%"><option value="">Select a person…</option></select>
              </div>
              <div class="form-group">
                <label class="caption" for="regLookLabel">Appearance label</label>
                <input type="text" id="regLookLabel" style="width:100%" placeholder="e.g. Without glasses">
              </div>
              <label class="check-line"><input type="checkbox" id="regLookConfirm"> I checked that the person in front of the camera is the selected profile.</label>
            </div>

            <div class="shots-head">
              <span class="caption" id="shotsHeading">Live shots · 0/10</span>
              <button type="button" class="btn btn-ghost btn-sm" id="btnClearShots" style="display:none;height:24px;padding:0 8px;">Retake all</button>
            </div>
            <div class="shots-grid" id="shotsGrid" aria-label="Captured live shots"></div>
            <div class="progress-track" style="margin-top:8px;margin-bottom:10px;">
              <div class="progress-fill" id="regStepFill"></div>
            </div>

            <div class="err-text" id="regFormErr"></div>

            <div class="dup-box" id="regDupBox" role="alert">
              <div id="regDupMsg">This face already matches an existing profile.</div>
              <input type="text" id="dupAppLabel" placeholder="Appearance label (e.g. Without glasses)">
              <div class="btn-row" style="margin-top:8px;">
                <button type="button" class="btn btn-primary btn-sm" id="btnDupAddApp">Add as appearance</button>
                <button type="button" class="btn btn-secondary btn-sm" id="btnDupForce">Register separately</button>
                <button type="button" class="btn btn-ghost btn-sm" id="btnDupCancel" style="flex:0 0 auto;">Cancel</button>
              </div>
            </div>

            <div class="btn-row">
              <button type="submit" class="btn btn-primary" id="btnRegister" disabled>Register (10 more shots)</button>
            </div>
          </form>
        </div>

        <div class="card">
          <div class="caption" style="margin-bottom:6px;">Existing person &amp; offline options</div>
          <div class="alt-option">
            <div>
              <div class="alt-option-title">Import 3 reference photos</div>
              <div class="alt-option-sub" id="regImportSub">Create a profile from 3 clear photos while the camera is stopped.</div>
            </div>
            <button class="btn btn-secondary btn-sm" id="btnRegPhotoImport" disabled>Import…</button>
          </div>
          <div class="divider-h" style="margin-top:10px;"></div>
          <div class="meta-row">
            <span id="profileCount">0 registered profiles</span>
            <button class="btn btn-ghost btn-sm" onclick="switchPage('profiles')">Manage profiles →</button>
          </div>
        </div>
      </aside>
    </div>
    <div class="footer-note">Stored locally · Camera images are never saved to disk · Live PAD anti-spoofing verified per shot</div>
  </section>

  <!-- PAGE 2: PROFILES -->
  <section class="page" id="page-profiles">
    <div class="card toolbar">
      <div class="field-inline" style="flex:1;min-width:220px;">
        <span class="caption">Search</span>
        <input type="text" id="profileSearch" style="flex:1" placeholder="Filter by name or student ID…">
      </div>
      <button class="btn btn-secondary" id="btnRefreshProfiles">Refresh</button>
      <button class="btn btn-primary" id="btnNewPhotoImport">+ New person from photos…</button>
    </div>
    <div id="profilesNotice" style="margin-bottom:12px;font-size:12.5px;color:var(--muted);">Stop the camera before editing profiles or importing reference photos.</div>
    <div class="split-2">
      <div class="card">
        <div class="table-head"><h2 class="card-title">Registered people</h2><span class="count-pill" id="profilesTotal">0 people</span></div>
        <div class="table-wrap tall">
          <table>
            <thead><tr><th>Name / Student ID</th><th>Looks</th><th>Samples</th></tr></thead>
            <tbody id="profilesBody"><tr><td colspan="3" class="empty-row">No profiles loaded</td></tr></tbody>
          </table>
        </div>
      </div>
      <div class="card">
        <h2 class="card-title" id="selProfileTitle">No profile selected</h2>
        <div id="selProfileSub" style="font-size:12.5px;color:var(--muted);margin:3px 0 12px;">Select a person from the list to inspect or update their profile.</div>
        <div class="btn-row" style="margin-top:0;margin-bottom:14px;">
          <input type="text" id="renameInput" placeholder="New profile name" disabled style="flex:1">
          <button class="btn btn-secondary" id="btnRename" disabled style="flex:0 0 auto;">Rename</button>
        </div>
        <div class="table-wrap" style="max-height:240px;">
          <table>
            <thead><tr><th>Appearance</th><th>Samples</th><th></th></tr></thead>
            <tbody id="appearancesBody"><tr><td colspan="3" class="empty-row">—</td></tr></tbody>
          </table>
        </div>
        <div class="btn-row" style="margin-top:12px;">
          <button class="btn btn-secondary" id="btnImportLook" disabled>Add appearance with camera…</button>
          <button class="btn btn-danger" id="btnDeleteProfile" disabled>Delete profile</button>
        </div>
      </div>
    </div>
  </section>

  <!-- PAGE 3: ATTENDANCE -->
  <section class="page" id="page-attendance">
    <div class="kpi-strip-5">
      <div class="kpi-box"><div class="caption">Present</div><div class="kpi-num" style="color:var(--mint-ink);" id="kpiRecords">0</div></div>
      <div class="kpi-box"><div class="caption">Absent</div><div class="kpi-num" style="color:var(--muted);" id="kpiAbsent">0</div></div>
      <div class="kpi-box"><div class="caption">Blocked spoofs</div><div class="kpi-num" style="color:var(--rose-ink);" id="kpiSpoofs">0</div></div>
      <div class="kpi-box"><div class="caption">Automatic</div><div class="kpi-num" id="kpiAuto">0</div></div>
      <div class="kpi-box"><div class="caption">Manual review</div><div class="kpi-num" style="color:var(--amber-ink);" id="kpiManual">0</div></div>
    </div>
    <div class="card toolbar">
      <div class="field-inline">
        <span class="caption">Session</span>
        <select id="histSession" style="min-width:170px"><option value="">All sessions</option></select>
      </div>
      <div class="field-inline">
        <span class="caption">Source</span>
        <select id="histSource">
          <option value="">All sources</option>
          <option value="automatic">Automatic</option>
          <option value="manual_review">Manual</option>
          <option value="legacy_unrecorded">Unknown (legacy)</option>
        </select>
      </div>
      <input type="text" id="histSearch" style="flex:1;min-width:180px" placeholder="Search name or session…">
      <button class="btn btn-secondary" id="btnHistRefresh">Refresh</button>
      <button class="btn btn-primary" id="btnExportCsv" disabled>Export CSV</button>
    </div>
    <div class="card">
      <div class="table-head">
        <div style="display:flex;gap:6px;">
          <button class="btn btn-sm btn-secondary" id="btnAttPresent" onclick="switchAttTab('present')">Present records</button>
          <button class="btn btn-sm btn-ghost" id="btnAttAbsent" onclick="switchAttTab('absent')">Absent</button>
          <button class="btn btn-sm btn-ghost" id="btnAttSpoofs" onclick="switchAttTab('spoofs')">Blocked spoofs</button>
        </div>
        <span id="attScopeNote" style="font-size:12px;color:var(--muted);"></span>
      </div>
      <div id="attTabPresent" class="table-wrap tall">
        <table>
          <thead><tr><th>Session</th><th>Name / Student ID</th><th>Recorded (Local)</th><th>Source</th><th>Reviewer</th></tr></thead>
          <tbody id="histBody"><tr><td colspan="5" class="empty-row">No saved records</td></tr></tbody>
        </table>
      </div>
      <div id="attTabAbsent" class="table-wrap tall" style="display:none;">
        <table>
          <thead><tr><th>Registered Name / Student ID</th><th>Enrolled</th><th>Status</th></tr></thead>
          <tbody id="absentBody"><tr><td colspan="3" class="empty-row">Everyone registered is present</td></tr></tbody>
        </table>
      </div>
      <div id="attTabSpoofs" class="table-wrap tall" style="display:none;">
        <table>
          <thead><tr><th>Time</th><th>Session</th><th>PAD details (in-memory log · no images saved)</th></tr></thead>
          <tbody id="spoofBody"><tr><td colspan="3" class="empty-row">No blocked spoof attempts in this run</td></tr></tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- PAGE 4: SETTINGS & DIAGNOSTICS -->
  <section class="page" id="page-diagnostics">
    <div class="settings-grid">
      <div class="card">
        <h2 class="card-title" style="margin-bottom:4px;">Live pipeline controls</h2>
        <div style="font-size:12.5px;color:var(--muted);margin-bottom:10px;">Safe display &amp; confirmation options. Calibrated model thresholds stay locked to validation checkpoints.</div>
        <div class="setting-row">
          <div><strong>Mirror camera preview</strong><div style="font-size:12px;color:var(--muted);">Flip display horizontally (analyzed crop is unchanged)</div></div>
          <label class="switch"><input type="checkbox" id="optMirror" checked></label>
        </div>
        <div class="setting-row">
          <div><strong>Live HUD overlay badge</strong><div style="font-size:12px;color:var(--muted);">Show verification status pill over the camera feed</div></div>
          <label class="switch"><input type="checkbox" id="optHud" checked></label>
        </div>
        <div class="setting-row">
          <div><strong>Confirmation frames</strong><div style="font-size:12px;color:var(--muted);">Consecutive live verified frames before auto check-in</div></div>
          <select id="optStable" style="width:105px;">
            <option value="1">1 frame</option><option value="2">2 frames</option>
            <option value="3" selected>3 frames</option><option value="4">4 frames</option><option value="5">5 frames</option>
          </select>
        </div>
        <div class="setting-row">
          <div><strong>Liveness smoothing</strong><div style="font-size:12px;color:var(--muted);">Average PAD live score over recent frames to reduce single-frame flicker</div></div>
          <select id="optSmooth" style="width:105px;">
            <option value="1">Off (1 frame)</option>
            <option value="3" selected>3 frames</option>
            <option value="5">5 frames</option>
          </select>
        </div>
        <div style="margin-top:12px;display:flex;gap:8px;align-items:center;">
          <button class="btn btn-primary btn-sm" id="btnRetrySetup" disabled>Retry model setup</button>
          <span id="diagRuntime" style="font-size:12px;color:var(--muted);">Checking runtime…</span>
        </div>
      </div>

      <div class="card">
        <h2 class="card-title" style="margin-bottom:4px;">Loaded models &amp; calibrated thresholds</h2>
        <div style="font-size:12.5px;color:var(--muted);margin-bottom:10px;">Validation-calibrated operating points across all three trained models.</div>
        <div class="table-wrap" style="max-height:220px;">
          <table>
            <thead><tr><th>Model</th><th>Checkpoint</th><th>Operating point</th></tr></thead>
            <tbody id="modelInfoBody"><tr><td colspan="3" class="empty-row">Loading…</td></tr></tbody>
          </table>
        </div>
      </div>
    </div>
    <div class="diag-grid" id="diagGrid"></div>
    <div class="card">
      <div class="caption" style="margin-bottom:8px;">Latest frame telemetry (memory only)</div>
      <pre class="mono-log" id="diagFrame">No frame processed in this camera run.</pre>
    </div>
  </section>
</main>

<!-- MODAL: MANUAL REVIEW -->
<div class="backdrop" id="modalReview">
  <div class="modal">
    <h2>Manual review</h2>
    <p class="modal-sub">Keep one person in front of the camera. Select a profile only after checking it in person.</p>
    <div class="crop-row">
      <img id="reviewCrop" class="crop-img" alt="Temporary face crop">
      <div>
        <div class="countdown-pill" id="reviewCountdown">Request valid for 30s</div>
        <div id="reviewTime" style="font-size:12px;color:var(--muted);">Temporary crop held in memory only.</div>
      </div>
    </div>
    <div class="form-group">
      <label class="caption" for="reviewProfile">Profile you checked</label>
      <select id="reviewProfile" style="width:100%"></select>
    </div>
    <div class="form-group">
      <label class="caption" for="reviewOperator">Reviewer name</label>
      <input type="text" id="reviewOperator" style="width:100%" placeholder="Operator name">
    </div>
    <div class="form-group">
      <label class="caption" for="reviewReason">Reason for confirmation</label>
      <input type="text" id="reviewReason" style="width:100%" placeholder="e.g. Checked student ID in person">
    </div>
    <label class="check-line"><input type="checkbox" id="reviewConfirmBox"> I have checked the person against the selected profile in person.</label>
    <div class="err-text" id="reviewErr"></div>
    <div class="btn-row">
      <button class="btn btn-primary" id="btnReviewSubmit">Record manual attendance</button>
      <button class="btn btn-secondary" id="btnReviewRetry" style="flex:0 0 auto;display:none;">Retry</button>
      <button class="btn btn-secondary" id="btnReviewCancel" style="flex:0 0 auto;">Cancel</button>
    </div>
  </div>
</div>

<!-- MODAL: ADD APPEARANCE -->
<div class="backdrop" id="modalAppearance">
  <div class="modal">
    <h2>Add appearance to existing profile</h2>
    <p class="modal-sub">Use this for a verified person with a different look (e.g. without glasses). The camera will capture 3 live samples.</p>
    <div class="countdown-pill" id="appCountdown">Request valid for 30s</div>
    <div class="form-group">
      <label class="caption" for="appProfile">Existing profile</label>
      <select id="appProfile" style="width:100%"></select>
    </div>
    <div class="form-group">
      <label class="caption" for="appLabel">Appearance label</label>
      <input type="text" id="appLabel" style="width:100%" placeholder="e.g. Without glasses">
    </div>
    <label class="check-line"><input type="checkbox" id="appConfirmBox"> I have verified this is the person in the selected profile.</label>
    <div class="err-text" id="appErr"></div>
    <div class="btn-row">
      <button class="btn btn-primary" id="btnAppSubmit">Begin capture</button>
      <button class="btn btn-secondary" id="btnAppRetry" style="flex:0 0 auto;display:none;">Retry</button>
      <button class="btn btn-secondary" id="btnAppCancel" style="flex:0 0 auto;">Cancel</button>
    </div>
  </div>
</div>

<!-- MODAL: PHOTO IMPORT -->
<div class="backdrop" id="modalImport">
  <div class="modal">
    <h2 id="importTitle">Import reference photos</h2>
    <p class="modal-sub">Choose exactly 3 clear photos of the same person and look. Checks do not prove live presence or record attendance.</p>
    <div class="form-group">
      <label class="caption" id="importFieldLabel" for="importTarget">Name / student ID</label>
      <input type="text" id="importTarget" style="width:100%">
    </div>
    <div class="form-group">
      <input type="file" id="importFiles" accept="image/*" multiple>
    </div>
    <div class="table-wrap" style="max-height:160px;margin-bottom:10px;">
      <table>
        <thead><tr><th>Photo</th><th>Check result</th></tr></thead>
        <tbody id="importBody"><tr><td colspan="2" class="empty-row">Select 3 photos, then click Check photos</td></tr></tbody>
      </table>
    </div>
    <label class="check-line"><input type="checkbox" id="importConfirmBox"> I verified all 3 photos show this person and may be used as reference images.</label>
    <div class="err-text" id="importErr"></div>
    <div class="btn-row">
      <button class="btn btn-secondary" id="btnImportCheck">Check photos</button>
      <button class="btn btn-primary" id="btnImportCommit" disabled>Import checked photos</button>
      <button class="btn btn-secondary" id="btnImportClose" style="flex:0 0 auto;">Cancel</button>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
const EMOTIONS = ["happy", "neutral", "surprise", "sad", "angry", "fear", "disgust"];
const $ = id => document.getElementById(id);
let lastState = null, lastNoticeId = 0, videoActive = false, currentPage = "live";
let profilesCache = [], selectedProfileId = null, histRows = [];
let activeReviewId = null, activeAppId = null;
let importPerson = null, importToken = null, importOpId = null;
const GUIDED_MIN = 10, GUIDED_MAX = 10, GUIDED_MIN_LIVE = 3;  // GUIDED_MIN_LIVE mirrors service.GUIDED_MIN_SHOTS
const GUIDED_POSES = [
  {short: "Front 1", hint: "Look straight at the camera"},
  {short: "Front 2", hint: "Look straight, hold steady"},
  {short: "Left 1", hint: "Turn your head slightly left"},
  {short: "Left 2", hint: "Turn a little more to the left"},
  {short: "Right 1", hint: "Turn your head slightly right"},
  {short: "Right 2", hint: "Turn a little more to the right"},
  {short: "Up 1", hint: "Tilt your chin slightly up"},
  {short: "Up 2", hint: "Tilt your chin a little higher"},
  {short: "Down 1", hint: "Tilt your chin slightly down"},
  {short: "Down 2", hint: "Tilt your chin a little lower"},
];
let guidedShots = [], regBusy = false, regDuplicate = null, autoCapTimer = null;
let regMode = "new";  // "new" person or "look" (appearance of an existing profile)

// Build emotion rows once
$("emotionBars").innerHTML = EMOTIONS.map(e => `
  <div class="emo-row" id="emo-${e}">
    <span class="emo-name">${e[0].toUpperCase() + e.slice(1)}</span>
    <div class="emo-track"><div class="emo-fill" id="emobar-${e}"></div></div>
    <span class="emo-pct" id="emopct-${e}">—</span>
  </div>`).join("");

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove("show"), 3600);
}

async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)
  };
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok || data.error) {
    const err = new Error(data.error || ("HTTP " + res.status));
    err.status = res.status;
    err.data = data;
    throw err;
  }
  return data;
}

function syncVideoMount(running, forceReload = false) {
  const img = $("videoFeed");
  const liveVp = $("viewport"), regVp = $("regViewport");
  const target = currentPage === "register" ? regVp : liveVp;
  const other = currentPage === "register" ? liveVp : regVp;
  let moved = false;
  if (img.parentElement !== target) {
    target.insertBefore(img, target.querySelector(".placeholder"));
    moved = true;
  }
  other.classList.remove("live");
  target.classList.toggle("live", !!running);
  if (running && (!videoActive || moved || forceReload)) {
    videoActive = true;
    // Register shows only the oval guide, so it asks for frames without face boxes.
    img.src = "/api/video?" + (currentPage === "register" ? "boxes=0&" : "") + "t=" + Date.now();
  } else if (!running && videoActive) {
    videoActive = false;
    img.removeAttribute("src");
  }
}

function stopAutoCapture() {
  if (autoCapTimer) { clearInterval(autoCapTimer); autoCapTimer = null; }
  const btn = $("btnAutoCapture");
  if (btn) {
    btn.classList.remove("btn-auto-on");
    btn.textContent = "⚡ Auto-capture";
  }
}

function switchPage(page) {
  if (currentPage === "register" && page !== "register") stopAutoCapture();
  currentPage = page;
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.toggle("active", b.dataset.page === page));
  document.querySelectorAll(".page").forEach(p => p.classList.toggle("active", p.id === "page-" + page));
  if (page === "live" || page === "register") syncVideoMount(!!lastState?.running);
  if (page === "register") {
    api("/api/register/shots").then(d => { guidedShots = d.shots || []; renderShotsGrid(); updateRegisterReadiness(lastState); }).catch(() => {});
  }
  if (page === "profiles") loadProfiles();
  if (page === "attendance") loadHistory();
}
document.querySelectorAll(".nav-btn").forEach(b => b.addEventListener("click", () => switchPage(b.dataset.page)));

function setChip(id, state) {
  const cls = "chip" + (state === "success" ? " ok" : state === "warning" ? " warn" : state === "danger" ? " err" : "");
  const el = $("chip-" + id);
  if (el) el.className = cls;
  const regEl = $("reg-chip-" + id);
  if (regEl) regEl.className = cls;
}

function setStat(id, text, tone) {
  const card = $("card" + id);
  card.className = "stat-card" + (tone === "success" ? " ok" : tone === "info" ? " info" : tone === "warning" ? " warn" : tone === "danger" ? " err" : "");
  const badge = (id === "Identity" && tone === "success") ? '<span style="color:#10b981;">✓</span>' : "";
  $("val" + id).innerHTML = badge + escapeHtml(text || "—");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

function switchSideList(which) {
  $("sidePresent").style.display = which === "present" ? "block" : "none";
  $("sideActivity").style.display = which === "activity" ? "block" : "none";
  $("btnLogPresent").className = "btn btn-sm " + (which === "present" ? "btn-secondary" : "btn-ghost");
  $("btnLogActivity").className = "btn btn-sm " + (which === "activity" ? "btn-secondary" : "btn-ghost");
}

function switchAttTab(which) {
  $("attTabPresent").style.display = which === "present" ? "block" : "none";
  $("attTabAbsent").style.display = which === "absent" ? "block" : "none";
  $("attTabSpoofs").style.display = which === "spoofs" ? "block" : "none";
  $("btnAttPresent").className = "btn btn-sm " + (which === "present" ? "btn-secondary" : "btn-ghost");
  $("btnAttAbsent").className = "btn btn-sm " + (which === "absent" ? "btn-secondary" : "btn-ghost");
  $("btnAttSpoofs").className = "btn btn-sm " + (which === "spoofs" ? "btn-secondary" : "btn-ghost");
}

$("focusSelect").onchange = () => {
  const f = $("focusSelect").value;
  $("cardIdentity").classList.toggle("focus-dim", f !== "all" && f !== "identity");
  $("cardLiveness").classList.toggle("focus-dim", f !== "all" && f !== "liveness");
  $("cardExpression").classList.toggle("focus-dim", f !== "all" && f !== "expression");
};

function nextMissingSlot() {
  const used = new Set(guidedShots.map(x => x.slot));
  for (let i = 0; i < GUIDED_MAX; i++) if (!used.has(i)) return i;
  return -1;
}

function renderShotsGrid() {
  const bySlot = new Map(guidedShots.map(x => [x.slot, x]));
  const nextSlot = nextMissingSlot();
  $("shotsHeading").textContent = `Live shots · ${guidedShots.length}/${GUIDED_MAX}`;
  $("shotCountBadge").textContent = `${guidedShots.length}/${GUIDED_MAX}`;
  $("btnClearShots").style.display = guidedShots.length ? "inline-flex" : "none";
  $("regStepFill").style.width = Math.round((guidedShots.length / GUIDED_MAX) * 100) + "%";
  $("shotsGrid").innerHTML = GUIDED_POSES.map((pose, idx) => {
    const shot = bySlot.get(idx);
    if (shot) {
      const notLive = shot.is_live === false;
      return `<div class="shot-card filled ${notLive ? 'not-live' : ''}" title="${notLive ? 'Did not pass liveness; not saved' : escapeHtml(pose.hint)}">
        <img src="${shot.thumb_url}" alt="Shot ${idx + 1} (${escapeHtml(pose.short)})">
        ${notLive ? '<span class="shot-flag">Not live</span>' : ''}
        <button type="button" class="shot-rm" onclick="removeGuidedShot('${escapeHtml(shot.id)}')" aria-label="Remove shot ${idx + 1}">×</button>
        <div class="shot-badge">${idx + 1} · ${escapeHtml(pose.short)}</div>
      </div>`;
    }
    const isNext = idx === nextSlot;
    return `<div class="shot-card ${isNext ? 'next' : ''}" title="${escapeHtml(pose.hint)}">
      <span class="shot-num">${idx + 1}</span>
      <span class="shot-lbl">${escapeHtml(pose.short)}</span>
    </div>`;
  }).join("");
}
renderShotsGrid();

function hideDuplicateBox() {
  regDuplicate = null;
  $("regDupBox").classList.remove("open");
}

function showDuplicateBox(dup) {
  regDuplicate = dup;
  $("regDupMsg").innerHTML = `This face already matches <strong>${escapeHtml(dup.name)}</strong> (cosine ${Number(dup.cosine).toFixed(3)}). Add these shots as a new appearance, or register separately?`;
  $("dupAppLabel").value = "";
  $("regDupBox").classList.add("open");
}

function popCameraFlash() {
  const fl = $("regFlash");
  fl.classList.remove("pop");
  void fl.offsetWidth;
  fl.classList.add("pop");
}

// The camera worker owns the staged shots; another tab or a late reply can change
// them, so re-read them whenever the polled count disagrees with this page.
let shotsSyncing = false;
function syncGuidedShots() {
  if (shotsSyncing) return;
  shotsSyncing = true;
  api("/api/register/shots").then(d => {
    guidedShots = d.shots || [];
    if (!guidedShots.length) hideDuplicateBox();
    renderShotsGrid();
  }).catch(() => {}).finally(() => { shotsSyncing = false; });
}

function updateRegisterReadiness(s) {
  if (!s) return;
  if (!regBusy && typeof s.guided_shots_count === "number" && s.guided_shots_count !== guidedShots.length) {
    syncGuidedShots();
  }
  if (!s.running) stopAutoCapture();
  const fullName = $("regFullName").value.trim();
  const q = s.quality || {};
  const ready = !!s.running && !!s.can_capture_shot;
  const count = guidedShots.length;
  const full = count >= GUIDED_MAX;
  const enough = count >= GUIDED_MIN;
  // Only shots that passed liveness are saved; the server needs at least GUIDED_MIN_LIVE of them.
  const liveCount = guidedShots.filter(x => x.is_live !== false).length;
  const liveOk = liveCount >= GUIDED_MIN_LIVE;
  const nextSlot = nextMissingSlot();

  $("regGuideOval").classList.toggle("is-ready", ready && !full);
  $("btnTakeShot").disabled = !ready || full || regBusy;
  $("btnAutoCapture").disabled = !s.running || full || regBusy;
  if (full && autoCapTimer) stopAutoCapture();

  // Submit button label & state
  const regBtn = $("btnRegister");
  if (regBusy) {
    regBtn.disabled = true;
    regBtn.textContent = "Saving profile…";
  } else if (!enough) {
    const need = GUIDED_MIN - count;
    regBtn.disabled = true;
    regBtn.textContent = `Register (${need} more shot${need === 1 ? "" : "s"} needed)`;
  } else if (!liveOk) {
    const need = GUIDED_MIN_LIVE - liveCount;
    regBtn.disabled = true;
    regBtn.textContent = `Register (need ${need} more live shot${need === 1 ? "" : "s"})`;
  } else {
    regBtn.disabled = regMode === "look" ? !lookFormReady() : !fullName;
    regBtn.textContent = regMode === "look"
      ? `Save appearance (${liveCount}/${count} live shots saved)`
      : `Register profile (${liveCount}/${count} live shots saved)`;
  }

  // Pose overlay pill + capture hint
  const posePill = $("regPosePill"), poseText = $("regPoseText"), hint = $("regShootHint");
  if (!s.running) {
    posePill.className = "pose-pill";
    poseText.textContent = "Camera off";
    hint.className = "shoot-hint";
    hint.textContent = s.models_ready ? "Click Start camera to begin capturing live shots." : "Loading models… please wait.";
  } else if (full) {
    posePill.className = "pose-pill ok";
    poseText.textContent = `✓ All ${GUIDED_MAX}/${GUIDED_MAX} poses captured`;
    hint.className = liveOk ? "shoot-hint" : "shoot-hint warn";
    hint.textContent = !liveOk
      ? `Only ${liveCount} shot${liveCount === 1 ? "" : "s"} passed liveness — click × on the Not live shots and retake them.`
      : regMode === "look"
        ? (lookFormReady() ? "All shots taken — click Save appearance, or click × to retake any shot."
                           : "All shots taken — choose the profile, enter a label and tick the check on the right.")
      : fullName ? "All shots taken — click Register profile, or click × to retake any shot." : "All shots taken — enter Full name on the right to save.";
  } else {
    const pose = GUIDED_POSES[nextSlot >= 0 ? nextSlot : 0];
    if (ready) {
      posePill.className = "pose-pill ok";
      poseText.textContent = `${nextSlot + 1}/${GUIDED_MAX} · ${pose.hint}`;
      const liveNow = q.liveness === "success";
      hint.className = liveNow ? "shoot-hint" : "shoot-hint warn";
      hint.textContent = autoCapTimer
        ? `Auto-capturing pose ${nextSlot + 1}/${GUIDED_MAX} (${pose.short})…`
        : liveNow ? "Ready — click Capture shot or press Space."
        : "Ready — but liveness is not passing now, so this shot will be marked Not live and skipped on save.";
    } else if (q.liveness === "danger") {
      posePill.className = "pose-pill err";
      poseText.textContent = "⚠ Liveness not passed · Adjust front lighting";
      hint.className = "shoot-hint err";
      hint.textContent = "Not ready: liveness check did not pass.";
    } else {
      const blockers = [];
      if (q.face !== "success") blockers.push("single face");
      if (q.lighting !== "success") blockers.push("good lighting");
      if (q.focus !== "success") blockers.push("sharp focus");
      if (q.liveness !== "success") blockers.push("live face");
      const msg = blockers.length ? "Waiting for: " + blockers.join(", ") : "Hold steady inside the oval…";
      posePill.className = "pose-pill warn";
      poseText.textContent = `${nextSlot + 1}/${GUIDED_MAX} · ${pose.hint}`;
      hint.className = "shoot-hint warn";
      hint.textContent = msg;
    }
  }
}

async function captureGuidedShot(silentAuto = false) {
  if (regBusy || !lastState?.running || !lastState?.can_capture_shot) return;
  const slot = nextMissingSlot();
  if (slot < 0) { stopAutoCapture(); return; }
  regBusy = true;
  $("regFormErr").textContent = "";
  updateRegisterReadiness(lastState);
  try {
    const res = await api("/api/register/capture", {slot});
    guidedShots = res.shots || [];
    hideDuplicateBox();
    popCameraFlash();
    renderShotsGrid();
    if (guidedShots.length >= GUIDED_MAX) stopAutoCapture();
  } catch (e) {
    if (!silentAuto) $("regFormErr").textContent = e.message;
    syncGuidedShots();
  } finally {
    regBusy = false;
    updateRegisterReadiness(lastState);
  }
}

async function removeGuidedShot(shotId) {
  if (regBusy) return;
  stopAutoCapture();
  $("regFormErr").textContent = "";
  try {
    const res = await api("/api/register/remove", {shot_id: shotId});
    guidedShots = res.shots || [];
    hideDuplicateBox();
    renderShotsGrid();
    updateRegisterReadiness(lastState);
  } catch (e) { $("regFormErr").textContent = e.message; }
}

async function clearGuidedShots() {
  if (regBusy) return;
  stopAutoCapture();
  $("regFormErr").textContent = "";
  try {
    const res = await api("/api/register/clear", {});
    guidedShots = res.shots || [];
    hideDuplicateBox();
    renderShotsGrid();
    updateRegisterReadiness(lastState);
  } catch (e) { $("regFormErr").textContent = e.message; }
}

async function commitGuidedRegistration(opts = {}) {
  if (regBusy) return;
  stopAutoCapture();
  $("regFormErr").textContent = "";
  const fullName = $("regFullName").value.trim();
  const sid = $("regStudentId").value.trim();
  const combined = sid ? `${fullName} · ${sid}` : fullName;
  if (!opts.appearance_person_id && !fullName) {
    $("regFormErr").textContent = "Please enter a full name.";
    return;
  }
  regBusy = true;
  updateRegisterReadiness(lastState);
  try {
    const res = await api("/api/register/commit", {
      name: combined,
      force: !!opts.force,
      appearance_person_id: opts.appearance_person_id || null,
      appearance_label: opts.appearance_label || "",
    });
    const info = res.result || {};
    guidedShots = [];
    hideDuplicateBox();
    renderShotsGrid();
    $("regFullName").value = "";
    $("regStudentId").value = "";
    $("regLookLabel").value = "";
    $("regLookConfirm").checked = false;
    toast(info.mode === "appearance"
      ? `Added appearance "${info.label}" to ${info.name} (${info.used_shots} live shots).`
      : `Registered ${info.name} from ${info.used_shots} live shots.`);
    switchPage("live");
  } catch (e) {
    if (e.status === 409 && e.data?.duplicate) {
      showDuplicateBox(e.data.duplicate);
    } else {
      $("regFormErr").textContent = e.message;
    }
  } finally {
    regBusy = false;
    updateRegisterReadiness(lastState);
  }
}

function renderState(s) {
  lastState = s;
  if (!$("sessionInput").dataset.init) {
    $("sessionInput").value = s.session;
    $("optMirror").checked = !!s.mirror_preview;
    $("optHud").checked = !!s.show_hud;
    $("optStable").value = String(s.stable_frames_target || 3);
    $("optSmooth").value = String(s.liveness_smooth_window || 5);
    $("sessionInput").dataset.init = "1";
  }
  // Header status pill
  $("hdrState").textContent = s.state_text;
  const st = s.state_text.toLowerCase();
  $("hdrDot").className = "dot " + (st.includes("error") || st.includes("not ready") ? "err" :
    st.includes("running") || st.startsWith("ready") || st.includes("recorded") ? "ok" :
    st.includes("load") || st.includes("open") || st.includes("test") ? "warn" : "");

  // Toolbar controls (Live + Register)
  const locked = s.running || s.camera_starting || s.probe_busy;
  $("sessionInput").disabled = locked;
  $("btnStart").disabled = !s.models_ready || locked;
  $("btnStop").disabled = !s.running;
  $("btnRegStart").disabled = !s.models_ready || locked;
  $("btnRegStop").disabled = !s.running;
  const speedText = s.processing_ms != null ? `${s.speed} · ${s.processing_ms} ms` : s.speed;
  $("fpsLabel").textContent = speedText;
  $("regFpsLabel").textContent = s.running ? speedText : (s.camera_starting ? "Opening camera…" : "Camera off");

  // Video stream lifecycle + HUD pill
  syncVideoMount(s.running);
  const hud = $("hudPill");
  hud.className = "hud-pill" + (s.show_hud ? " visible" : "") +
    (s.feedback_tone === "success" ? " ok" : s.feedback_tone === "danger" ? " err" : s.feedback_tone === "warning" ? " warn" : "");
  $("hudText").textContent = s.identity_tone === "success"
    ? `✓ ${s.identity_text}` + (s.expression_text && s.expression_text !== "—" ? ` · ${s.expression_text}` : "")
    : s.liveness_tone === "danger" ? "⚠ Liveness not passed" : (s.feedback || "Scanning…");

  // Quality chips (shared across Live & Register)
  const q = s.quality || {};
  setChip("face", q.face || "neutral");
  setChip("lighting", q.lighting || "neutral");
  setChip("focus", q.focus || "neutral");
  setChip("liveness", q.liveness || "neutral");

  // Progress bar & banner (Live)
  const pctWidth = Math.round((s.scan_ratio || 0) * 100) + "%";
  const barColor = s.scan_color || "var(--mint)";
  $("scanFill").style.width = pctWidth;
  $("scanFill").style.background = barColor;
  const bannerCls = "banner " + (s.feedback_tone || "neutral");
  $("feedbackBanner").className = bannerCls;
  $("feedbackTitle").textContent = s.feedback;
  $("feedbackSub").textContent = s.guidance;

  // 3 KPI cards
  setStat("Identity", s.identity_text, s.identity_tone);
  setStat("Liveness", s.liveness_text, s.liveness_tone);
  setStat("Expression", s.expression_text, s.expression_tone);

  // Registration, Appearance, Manual Review & Profile counts
  $("btnReview").disabled = !s.can_review;
  const canOfflineImport = !locked && s.models_ready && s.photo_import_available && !s.preview;
  $("btnRegPhotoImport").disabled = !canOfflineImport;
  $("regImportSub").textContent = s.running
    ? "Stop the camera first to import 3 offline reference photos."
    : "Create a profile from 3 clear photos while the camera is stopped.";
  $("profileCount").textContent = s.profile_count + (s.profile_count === 1 ? " registered profile" : " registered profiles");
  $("liveProfileCount").textContent = s.profile_count + (s.profile_count === 1 ? " profile" : " profiles");
  const goRegBtn = $("btnLiveGoRegister");
  goRegBtn.className = "btn btn-sm " + (s.can_register ? "btn-primary" : "btn-ghost");
  goRegBtn.textContent = s.can_register ? "+ Register this face →" : "+ Register face →";
  updateRegisterReadiness(s);

  // Telemetry bars + Top-3 gallery matches
  const fmtPad = v => v < 0.05 ? v.toFixed(4) : v.toFixed(2);
  if (s.pad_score != null && s.pad_threshold != null) {
    const raw = s.pad_raw_score != null && s.pad_raw_score !== s.pad_score ? ` · this frame ${fmtPad(s.pad_raw_score)}` : "";
    $("padScoreTxt").textContent = `${raw ? "avg " : ""}${fmtPad(s.pad_score)}${raw} (thr ${fmtPad(s.pad_threshold)})`;
    const pct = s.pad_threshold < 0.05
      ? Math.min(100, (Math.log10(Math.max(1e-6, s.pad_score)) + 6) / 6 * 100)
      : Math.max(0, Math.min(100, s.pad_score * 100));
    $("padBar").style.width = Math.max(0, pct) + "%";
    $("padBar").style.background = s.liveness_tone === "success" ? "var(--mint)" : "var(--rose)";
  } else {
    $("padScoreTxt").textContent = "Waiting for frame";
    $("padBar").style.width = "0%";
  }
  const targetFrames = s.stable_frames_target || 3;
  if (s.id_cosine != null) {
    const streak = s.stable_frames ? ` · ${s.stable_frames}/${targetFrames}` : "";
    const thrTxt = s.id_threshold != null ? ` (thr ${s.id_threshold.toFixed(2)})` : "";
    $("idScoreTxt").textContent = `cos ${s.id_cosine.toFixed(2)}${thrTxt}${streak}`;
    $("idBar").style.width = Math.max(0, Math.min(100, (s.id_cosine + 1) * 50)) + "%";
    $("idBar").style.background = s.identity_tone === "success" ? "var(--mint)" : "var(--amber)";
  } else {
    $("idScoreTxt").textContent = "Waiting for frame";
    $("idBar").style.width = "0%";
  }
  const matches = s.top_matches || [];
  if (!matches.length) {
    $("topMatches").innerHTML = '<span style="color:var(--subtle);">No live identity probe yet</span>';
  } else {
    const thr = s.id_threshold ?? 0.87;
    $("topMatches").innerHTML = matches.map((m, i) => {
      const pct = Math.max(0, Math.min(100, (m.cosine + 1) * 50));
      const pass = m.cosine >= thr;
      return `<div class="match-row">
        <span><strong>#${i+1}</strong> ${escapeHtml(m.name)}</span>
        <span style="font-variant-numeric:tabular-nums;color:${pass ? 'var(--mint-ink)' : 'var(--muted)'};font-weight:${pass ? '700' : '500'};">${m.cosine.toFixed(3)}</span>
        <div class="match-bar"><div class="match-fill ${pass ? 'pass' : ''}" style="width:${pct.toFixed(1)}%"></div></div>
      </div>`;
    }).join("");
  }
  const scores = s.emotion_scores || {};
  EMOTIONS.forEach(e => {
    const row = $("emo-" + e), bar = $("emobar-" + e), pct = $("emopct-" + e);
    if (e in scores) {
      const v = scores[e];
      row.classList.toggle("top", e === s.emotion_top);
      bar.style.width = Math.round(v * 100) + "%";
      pct.textContent = Math.round(v * 100) + "%";
    } else {
      row.classList.remove("top"); bar.style.width = "0%"; pct.textContent = "—";
    }
  });

  // Present this session table + Live activity feed
  const rows = s.attendance || [];
  $("presentCount").textContent = rows.length + " present";
  $("spoofBadge").textContent = s.spoof_count ? ` (${s.spoof_count} blocked)` : "";
  const body = $("sessionBody");
  if (!rows.length) {
    body.innerHTML = '<tr><td colspan="3" class="empty-row">No attendance recorded yet</td></tr>';
  } else {
    body.innerHTML = rows.map((r, i) => `
      <tr onclick="showSessionDetail(${i})">
        <td><strong>${escapeHtml(r.name)}</strong></td>
        <td style="font-variant-numeric:tabular-nums;">${escapeHtml(r.time_local)}</td>
        <td><span class="tag ${r.source === 'manual_review' ? 'tag-manual' : r.source === 'automatic' ? 'tag-auto' : 'tag-legacy'}">${escapeHtml(r.source_label)}</span></td>
      </tr>`).join("");
  }
  const acts = s.activity || [];
  $("activityList").innerHTML = !acts.length
    ? '<div class="empty-row">No check-ins or blocked spoof events yet</div>'
    : acts.map(a => {
        const pill = a.kind === "spoof"
          ? '<span class="tag" style="background:var(--rose-soft);color:var(--rose-ink);">Spoof</span>'
          : a.kind === "manual" ? '<span class="tag tag-manual">Manual</span>' : '<span class="tag tag-auto">Check-in</span>';
        return `<div class="act-item"><span class="act-time">${escapeHtml(a.time)}</span>${pill}<span>${escapeHtml(a.text)}</span></div>`;
      }).join("");

  // Settings & Diagnostics page
  $("diagRuntime").textContent = s.runtime_summary;
  $("btnRetrySetup").disabled = !s.can_retry_setup;
  $("diagFrame").textContent = s.last_frame_details || "No frame processed in this camera run.";
  const mi = s.model_info || {};
  const idArch = mi.identity_backbone === "resnet18" ? "ResNet18" : "Compact CNN";
  const idMethod = mi.identity_method ? mi.identity_method[0].toUpperCase() + mi.identity_method.slice(1) : "Triplet";
  const padName = mi.liveness_arch === "mobilenetv3_large" ? "Anti-Spoofing (MobileNetV3-Large)" : "Anti-Spoofing (PAD CNN)";
  const padCrop = mi.liveness_crop_scale ? ` · crop ${mi.liveness_crop_scale}×` : "";
  $("modelInfoBody").innerHTML = `
    <tr><td><strong>Identity (${escapeHtml(idArch)} · ${escapeHtml(idMethod)})</strong></td><td>${escapeHtml(mi.identity_checkpoint || "—")}</td><td>Cosine ≥ ${mi.identity_threshold != null ? mi.identity_threshold.toFixed(4) : "—"} · margin ${mi.identity_margin ?? "—"}</td></tr>
    <tr><td><strong>${padName}</strong></td><td>${escapeHtml(mi.liveness_checkpoint || "—")}</td><td>Live score ≥ ${mi.liveness_threshold != null ? mi.liveness_threshold.toFixed(4) : "—"}${padCrop}</td></tr>
    <tr><td><strong>Expression (${mi.emotion_arch === "EmotionMobileNetV3-v1" ? "MobileNetV3" : "FER2013 CNN"})</strong></td><td>${escapeHtml(mi.emotion_checkpoint || "—")}</td><td>7 classes · Display only</td></tr>
    <tr><td><strong>Runtime</strong></td><td>${escapeHtml((mi.device || "auto").toUpperCase())} · ${mi.num_threads || 4} threads</td><td>Auto check-in: ${s.stable_frames_target || 3} frames · Smooth: ${s.liveness_smooth_window || 5}f</td></tr>`;
  $("diagGrid").innerHTML = (s.components || []).map(c => {
    const cls = c.status === "Ready" ? "tag-auto" : c.status === "Error" ? "tag-manual" : "tag-legacy";
    const style = c.status === "Error" ? 'style="background:var(--rose-soft);color:var(--rose-ink);"' : "";
    return `<div class="diag-card">
      <div class="diag-top"><strong>${escapeHtml(c.title)}</strong><span class="tag ${cls}" ${style}>${escapeHtml(c.status)}</span></div>
      <div class="diag-detail">${escapeHtml(c.detail || "—")}</div>
    </div>`;
  }).join("");

  // Active modals synced with backend
  syncReviewModal(s.review_request);
  syncAppearanceModal(s.appearance_request);

  if (s.notice && s.notice.id !== lastNoticeId) {
    lastNoticeId = s.notice.id;
    toast(s.notice.message);
  }
}

function showSessionDetail(idx) {
  const r = (lastState?.attendance || [])[idx];
  if (!r) return;
  toast(`${r.name} · ${r.source_label}` + (r.reviewer ? ` · Reviewer: ${r.reviewer} (${r.reason || "—"})` : ""));
}

function syncReviewModal(req) {
  const modal = $("modalReview");
  if (!req) {
    if (activeReviewId) { activeReviewId = null; modal.classList.remove("open"); }
    return;
  }
  if (req.id !== activeReviewId) {
    activeReviewId = req.id;
    $("reviewCrop").src = req.crop_data_url || "";
    $("reviewTime").textContent = "Face observed at " + req.observed_local + " · Memory only";
    $("reviewProfile").innerHTML = '<option value="">— Select verified profile —</option>' +
      req.profiles.map(p => `<option value="${escapeHtml(p.id)}">${escapeHtml(p.name)}</option>`).join("");
    $("reviewOperator").value = ""; $("reviewReason").value = ""; $("reviewConfirmBox").checked = false;
    $("reviewErr").textContent = "";
    modal.classList.add("open");
  }
  const pill = $("reviewCountdown");
  pill.textContent = req.valid ? `Request valid for ${Math.ceil(req.seconds_left)}s` : "Request expired or invalid";
  pill.classList.toggle("err", !req.valid);
  $("btnReviewSubmit").disabled = !req.valid;
  $("btnReviewRetry").style.display = req.valid ? "none" : "inline-flex";
  if (!req.valid && req.reason) $("reviewErr").textContent = req.reason;
}

function syncAppearanceModal(req) {
  const modal = $("modalAppearance");
  if (!req) {
    if (activeAppId) { activeAppId = null; modal.classList.remove("open"); }
    return;
  }
  if (req.id !== activeAppId) {
    activeAppId = req.id;
    const prevProfile = $("appProfile").value;
    const prevLabel = $("appLabel").value;
    $("appProfile").innerHTML = '<option value="">— Select existing profile —</option>' +
      req.profiles.map(p => `<option value="${escapeHtml(p.id)}">${escapeHtml(p.name)}</option>`).join("");
    if (prevProfile) $("appProfile").value = prevProfile;
    if (!prevLabel) $("appLabel").value = "";
    $("appConfirmBox").checked = false; $("appErr").textContent = "";
    modal.classList.add("open");
  }
  const pill = $("appCountdown");
  pill.textContent = req.valid ? `Request valid for ${Math.ceil(req.seconds_left)}s · Keep same person in view` : "Request expired or invalid";
  pill.classList.toggle("err", !req.valid);
  $("btnAppSubmit").disabled = !req.valid;
  $("btnAppRetry").style.display = req.valid ? "none" : "inline-flex";
  if (!req.valid && req.reason) $("appErr").textContent = req.reason;
}

// Button handlers (Live & Register)
const startCamera = () => api("/api/camera/start", {session: $("sessionInput").value, camera: lastState?.camera ?? 0}).catch(e => toast(e.message));
const stopCamera = () => { stopAutoCapture(); return api("/api/camera/stop", {}).catch(e => toast(e.message)); };
$("btnStart").onclick = startCamera;
$("btnStop").onclick = stopCamera;
$("btnRegStart").onclick = startCamera;
$("btnRegStop").onclick = stopCamera;
$("btnRetrySetup").onclick = () => api("/api/setup/retry", {}).catch(e => toast(e.message));
$("regFullName").addEventListener("input", () => updateRegisterReadiness(lastState));
$("btnTakeShot").onclick = () => captureGuidedShot(false);
$("btnClearShots").onclick = clearGuidedShots;
$("btnAutoCapture").onclick = () => {
  if (autoCapTimer) {
    stopAutoCapture();
  } else {
    const btn = $("btnAutoCapture");
    btn.classList.add("btn-auto-on");
    btn.textContent = "⏸ Pause auto";
    captureGuidedShot(true);
    autoCapTimer = setInterval(() => {
      if (currentPage !== "register" || !lastState?.running || guidedShots.length >= GUIDED_MAX) {
        stopAutoCapture();
        return;
      }
      if (lastState?.can_capture_shot && !regBusy) captureGuidedShot(true);
    }, 950);
  }
  updateRegisterReadiness(lastState);
};
window.addEventListener("keydown", (ev) => {
  if (currentPage !== "register" || ev.code !== "Space") return;
  if (ev.target && ev.target.closest("input, textarea, select, button")) return;
  ev.preventDefault();
  captureGuidedShot(false);
});
$("registerForm").onsubmit = (ev) => {
  ev.preventDefault();
  if (regMode === "look") {
    commitGuidedRegistration({appearance_person_id: $("regLookProfile").value, appearance_label: $("regLookLabel").value.trim()});
  } else {
    commitGuidedRegistration({force: false});
  }
};

// "Add appearance" uses the same guided camera shots, saved as a new look of an existing profile.
function lookFormReady() {
  return !!$("regLookProfile").value && !!$("regLookLabel").value.trim() && $("regLookConfirm").checked;
}
async function loadLookProfiles(selectId) {
  try {
    const data = await api("/api/profiles");
    const sel = $("regLookProfile"), keep = selectId || sel.value;
    sel.innerHTML = '<option value="">Select a person…</option>' + (data.profiles || []).map(p =>
      `<option value="${escapeHtml(p.id)}" ${p.appearances.length >= 4 ? "disabled" : ""}>${escapeHtml(p.name)} (${p.appearances.length}/4 looks)</option>`).join("");
    sel.value = [...sel.options].some(o => o.value === keep && !o.disabled) ? keep : "";
  } catch (e) { toast(e.message); }
  updateRegisterReadiness(lastState);
}
function setRegMode(mode, selectId) {
  regMode = mode;
  $("regModeNew").classList.toggle("active", mode === "new");
  $("regModeLook").classList.toggle("active", mode === "look");
  $("regNewFields").style.display = mode === "new" ? "" : "none";
  $("regLookFields").style.display = mode === "look" ? "" : "none";
  $("regFullName").required = mode === "new";
  $("regTitle").textContent = mode === "new" ? "Register a person" : "Add an appearance";
  $("regFormErr").textContent = "";
  hideDuplicateBox();
  if (mode === "look") loadLookProfiles(selectId);
  updateRegisterReadiness(lastState);
}
$("regModeNew").onclick = () => setRegMode("new");
$("regModeLook").onclick = () => setRegMode("look");
["regLookProfile", "regLookConfirm"].forEach(id => $(id).addEventListener("change", () => updateRegisterReadiness(lastState)));
$("regLookLabel").addEventListener("input", () => updateRegisterReadiness(lastState));
$("btnDupAddApp").onclick = () => {
  if (!regDuplicate) return;
  const label = $("dupAppLabel").value.trim() || "Additional look";
  commitGuidedRegistration({appearance_person_id: regDuplicate.person_id, appearance_label: label});
};
$("btnDupForce").onclick = () => commitGuidedRegistration({force: true});
$("btnDupCancel").onclick = hideDuplicateBox;
$("btnReview").onclick = () => api("/api/review/request", {}).catch(e => toast(e.message));
$("btnRegPhotoImport").onclick = () => openImportModal(null);

$("btnReviewCancel").onclick = () => {
  const id = activeReviewId; activeReviewId = null; $("modalReview").classList.remove("open");
  if (id) api("/api/review/cancel", {review_id: id}).catch(() => {});
};
$("btnReviewRetry").onclick = async () => {
  const id = activeReviewId;
  if (id) await api("/api/review/cancel", {review_id: id}).catch(() => {});
  await api("/api/review/request", {}).catch(e => toast(e.message));
};
$("btnReviewSubmit").onclick = async () => {
  $("reviewErr").textContent = "";
  try {
    await api("/api/review/confirm", {
      review_id: activeReviewId, person_id: $("reviewProfile").value,
      reviewer: $("reviewOperator").value, reason: $("reviewReason").value,
      confirmed: $("reviewConfirmBox").checked
    });
    activeReviewId = null; $("modalReview").classList.remove("open");
  } catch (e) { $("reviewErr").textContent = e.message; }
};

$("btnAppCancel").onclick = () => {
  const id = activeAppId; activeAppId = null; $("modalAppearance").classList.remove("open");
  if (id) api("/api/appearance/cancel", {request_id: id}).catch(() => {});
};
$("btnAppRetry").onclick = async () => {
  const id = activeAppId;
  if (id) await api("/api/appearance/cancel", {request_id: id}).catch(() => {});
  await api("/api/appearance/request", {}).catch(e => toast(e.message));
};
$("btnAppSubmit").onclick = async () => {
  $("appErr").textContent = "";
  try {
    await api("/api/appearance/begin", {
      request_id: activeAppId, person_id: $("appProfile").value,
      label: $("appLabel").value, confirmed: $("appConfirmBox").checked
    });
    activeAppId = null; $("modalAppearance").classList.remove("open");
    toast("Capturing 3 live appearance samples…");
  } catch (e) { $("appErr").textContent = e.message; }
};

// Profiles page
async function loadProfiles() {
  try {
    const data = await api("/api/profiles");
    profilesCache = data.profiles || [];
    $("btnNewPhotoImport").disabled = !data.editable || !data.photo_import_available;
    $("profilesNotice").textContent = data.editable
      ? "One profile per person. Add separate appearances (e.g. with and without glasses) or import 3 reference photos."
      : "Camera is currently running — stop the camera before editing profiles or importing photos.";
    renderProfiles(data.editable, data.photo_import_available);
  } catch (e) { toast(e.message); }
}

function renderProfiles(editable = true, canImport = true) {
  const q = $("profileSearch").value.trim().toLowerCase();
  const list = profilesCache.filter(p => !q || p.name.toLowerCase().includes(q));
  $("profilesTotal").textContent = list.length + (list.length === 1 ? " person" : " people");
  const body = $("profilesBody");
  if (!list.length) {
    body.innerHTML = '<tr><td colspan="3" class="empty-row">No matching profiles</td></tr>';
  } else {
    body.innerHTML = list.map(p => `
      <tr class="${p.id === selectedProfileId ? 'selected' : ''}" onclick="selectProfile('${escapeHtml(p.id)}')">
        <td><strong>${escapeHtml(p.name)}</strong></td>
        <td>${p.appearances.length}</td>
        <td>${p.sample_count}</td>
      </tr>`).join("");
  }
  const sel = profilesCache.find(p => p.id === selectedProfileId);
  $("selProfileTitle").textContent = sel ? sel.name : "No profile selected";
  $("selProfileSub").textContent = sel ? `Created ${sel.created_local} · ${sel.sample_count} stored embeddings` : "Select a person from the list to inspect or update their profile.";
  $("renameInput").value = sel ? sel.name : "";
  $("renameInput").disabled = !sel || !editable;
  $("btnRename").disabled = !sel || !editable;
  $("btnImportLook").disabled = !sel || sel.appearances.length >= 4;
  $("btnDeleteProfile").disabled = !sel || !editable;
  const ab = $("appearancesBody");
  if (!sel) {
    ab.innerHTML = '<tr><td colspan="3" class="empty-row">—</td></tr>';
  } else {
    ab.innerHTML = sel.appearances.map(a => `
      <tr>
        <td>${escapeHtml(a.label)}</td>
        <td>${a.sample_count}</td>
        <td style="text-align:right;">${sel.appearances.length > 1 && editable
          ? `<button class="btn btn-danger btn-sm" onclick="removeLook('${escapeHtml(sel.id)}','${escapeHtml(a.id)}','${escapeHtml(a.label)}')">Remove</button>` : ""}</td>
      </tr>`).join("");
  }
}

function selectProfile(id) { selectedProfileId = id; renderProfiles(!lastState?.running, lastState?.photo_import_available); }
$("profileSearch").addEventListener("input", () => renderProfiles(!lastState?.running, lastState?.photo_import_available));
$("btnRefreshProfiles").onclick = loadProfiles;
$("btnRename").onclick = async () => {
  if (!selectedProfileId) return;
  try {
    const res = await api("/api/profiles/rename", {person_id: selectedProfileId, name: $("renameInput").value});
    toast(res.message || "Profile renamed");
    await loadProfiles();
  } catch (e) { toast(e.message); }
};
$("btnDeleteProfile").onclick = async () => {
  const sel = profilesCache.find(p => p.id === selectedProfileId);
  if (!sel || !confirm(`Delete "${sel.name}"? Their face samples, all appearances and all attendance records are removed. A database backup is saved automatically.`)) return;
  try {
    const res = await api("/api/profiles/delete", {person_id: sel.id});
    selectedProfileId = null;
    toast(res.message || "Profile deleted");
    await loadProfiles();
  } catch (e) { toast(e.message); }
};
async function removeLook(pid, aid, label) {
  if (!confirm(`Remove appearance "${label}"? A database backup is saved automatically.`)) return;
  try {
    const res = await api("/api/profiles/remove-appearance", {person_id: pid, appearance_id: aid});
    toast(res.message || "Appearance removed");
    await loadProfiles();
  } catch (e) { toast(e.message); }
}

// Photo import modal
function openImportModal(person) {
  importPerson = person; importToken = null; importOpId = null;
  $("importTitle").textContent = person ? `Add reference photos: ${person.name}` : "New person from reference photos";
  $("importFieldLabel").textContent = person ? "Appearance label (e.g. Without glasses)" : "Name / student ID";
  $("importTarget").value = ""; $("importFiles").value = "";
  $("importConfirmBox").checked = false; $("importErr").textContent = "";
  $("btnImportCommit").disabled = true;
  $("importBody").innerHTML = '<tr><td colspan="2" class="empty-row">Select exactly 3 photos, then click Check photos</td></tr>';
  $("modalImport").classList.add("open");
}
$("btnNewPhotoImport").onclick = () => openImportModal(null);
$("btnImportLook").onclick = () => { const id = selectedProfileId; switchPage("register"); setRegMode("look", id); };
$("btnImportClose").onclick = () => $("modalImport").classList.remove("open");
$("importConfirmBox").onchange = () => { $("btnImportCommit").disabled = !(importToken && $("importConfirmBox").checked); };

function readFileB64(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve({name: file.name, data: r.result.split(",")[1]});
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}
$("btnImportCheck").onclick = async () => {
  $("importErr").textContent = ""; importToken = null; $("btnImportCommit").disabled = true;
  const files = Array.from($("importFiles").files || []);
  if (files.length !== 3) { $("importErr").textContent = "Please choose exactly 3 photos."; return; }
  try {
    const encoded = await Promise.all(files.map(readFileB64));
    const rep = await api("/api/profiles/import-check", {
      person_id: importPerson ? importPerson.id : null,
      name: importPerson ? "" : $("importTarget").value,
      label: importPerson ? $("importTarget").value : "",
      files: encoded
    });
    importToken = rep.token; importOpId = rep.operation_id;
    $("importBody").innerHTML = (rep.items || []).map(it =>
      `<tr><td>${escapeHtml(it.file)}</td><td>${escapeHtml(it.message)}</td></tr>`).join("");
    if (!rep.token) $("importErr").textContent = rep.message || "Photo checks did not pass.";
    else toast(rep.message || "Photos passed checks. Confirm and click Import.");
    $("btnImportCommit").disabled = !(importToken && $("importConfirmBox").checked);
  } catch (e) { $("importErr").textContent = e.message; }
};
$("btnImportCommit").onclick = async () => {
  try {
    const res = await api("/api/profiles/import-commit", {operation_id: importOpId, token: importToken, confirmed: true});
    $("modalImport").classList.remove("open");
    toast(res.message || "Reference samples imported.");
    await loadProfiles();
  } catch (e) { $("importErr").textContent = e.message; }
};

// Settings handlers
const saveSettings = () => api("/api/settings", {
  mirror_preview: $("optMirror").checked,
  show_hud: $("optHud").checked,
  stable_frames: Number($("optStable").value),
  liveness_smooth_window: Number($("optSmooth").value)
}).then(() => toast("Settings updated")).catch(e => toast(e.message));
$("optMirror").onchange = saveSettings;
$("optHud").onchange = saveSettings;
$("optStable").onchange = saveSettings;
$("optSmooth").onchange = saveSettings;

// Attendance history page
async function loadHistory() {
  const params = new URLSearchParams({
    session: $("histSession").value, source: $("histSource").value, q: $("histSearch").value
  });
  try {
    const d = await api("/api/history?" + params.toString());
    const selVal = $("histSession").value;
    $("histSession").innerHTML = '<option value="">All sessions</option>' +
      (d.sessions || []).map(s => `<option value="${escapeHtml(s)}" ${s === selVal ? "selected" : ""}>${escapeHtml(s)}</option>`).join("");
    const t = d.summary || {};
    $("kpiRecords").textContent = t.records || 0;
    $("kpiAbsent").textContent = t.absent || 0;
    $("kpiSpoofs").textContent = t.spoofs || 0;
    $("kpiAuto").textContent = t.automatic || 0;
    $("kpiManual").textContent = t.manual_review || 0;
    $("attScopeNote").textContent = t.target_session ? `Absent checked against session: ${t.target_session}` : "";
    histRows = d.rows || [];
    $("btnExportCsv").disabled = !histRows.length;
    const body = $("histBody");
    if (!histRows.length) {
      body.innerHTML = '<tr><td colspan="5" class="empty-row">No matching attendance records</td></tr>';
    } else {
      body.innerHTML = histRows.map((r, i) => `
        <tr onclick="toast('${escapeHtml(r.name)} · ${escapeHtml(r.source_label)}' + (${r.reason ? `' · ' + ${JSON.stringify(r.reason)}` : `''`}))">
          <td>${escapeHtml(r.session)}</td>
          <td><strong>${escapeHtml(r.name)}</strong></td>
          <td style="font-variant-numeric:tabular-nums;">${escapeHtml(r.recorded_local)}</td>
          <td><span class="tag ${r.source === 'manual_review' ? 'tag-manual' : r.source === 'automatic' ? 'tag-auto' : 'tag-legacy'}">${escapeHtml(r.source_label)}</span></td>
          <td>${escapeHtml(r.reviewer || "—")}</td>
        </tr>`).join("");
    }
    const abs = d.absent || [];
    $("absentBody").innerHTML = !abs.length
      ? '<tr><td colspan="3" class="empty-row">Everyone registered is present</td></tr>'
      : abs.map(p => `<tr><td><strong>${escapeHtml(p.name)}</strong></td><td>${escapeHtml(p.created_local)}</td><td><span class="tag tag-legacy">Absent</span></td></tr>`).join("");
    const spoofs = d.spoof_events || [];
    $("spoofBody").innerHTML = !spoofs.length
      ? '<tr><td colspan="3" class="empty-row">No blocked spoof attempts in this run</td></tr>'
      : spoofs.map(e => `<tr><td style="font-variant-numeric:tabular-nums;">${escapeHtml(e.time_local)}</td><td>${escapeHtml(e.session)}</td><td><span class="tag" style="background:var(--rose-soft);color:var(--rose-ink);">Blocked</span> ${escapeHtml(e.note)}</td></tr>`).join("");
  } catch (e) { toast(e.message); }
}
$("histSession").onchange = loadHistory;
$("histSource").onchange = loadHistory;
$("histSearch").oninput = loadHistory;
$("btnHistRefresh").onclick = loadHistory;
$("btnExportCsv").onclick = () => {
  const params = new URLSearchParams({
    session: $("histSession").value, source: $("histSource").value, q: $("histSearch").value
  });
  window.location.href = "/api/history/export.csv?" + params.toString();
};

// Poll loop (fast when camera running, light when idle)
async function pollLoop() {
  try {
    const s = await api("/api/state");
    renderState(s);
  } catch (_) {}
  setTimeout(pollLoop, lastState?.running ? 160 : 450);
}
pollLoop();
</script>
</body>
</html>
"""
