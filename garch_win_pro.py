"""
GARCH-GJR(1,1) PRO - Projeção de Volatilidade para WIN (^BVSP)
Versão PRO: GJR + VaR + CVaR + Percentil + Dashboard completo
GitHub Actions — salva index.html na raiz do repositório
"""

import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from scipy.stats import norm
import warnings, os, json
warnings.filterwarnings("ignore")

# ── CONFIGURAÇÕES ─────────────────────────────
TICKER         = "^BVSP"
ANOS_HISTORICO = 5
HORIZONTE_DIAS = 5
TRADING_DAYS   = 252
OUTPUT_HTML    = "index.html"
FUSO_BR        = ZoneInfo("America/Sao_Paulo")
CONFIANCA_VAR  = 0.95   # VaR 95%

# ── DADOS ─────────────────────────────────────
print("Baixando dados...")
data_fim    = datetime.today()
data_inicio = data_fim - timedelta(days=365 * ANOS_HISTORICO)
df = yf.download(TICKER, start=data_inicio, end=data_fim, progress=False)
if df.empty:
    print("Erro: sem dados.")
    exit()

fechamento = df["Close"].dropna()
retornos   = 100 * np.log(fechamento / fechamento.shift(1)).dropna()
ibov       = float(np.array(fechamento.iloc[-1]).flat[0])

# ── GARCH(1,1) CLÁSSICO ───────────────────────
print("Estimando GARCH(1,1)...")
m1       = arch_model(retornos, vol="Garch", p=1, q=1, dist="normal", rescale=False)
r1       = m1.fit(disp="off")
vol_garch = float(np.sqrt(r1.conditional_volatility.iloc[-1]) * np.sqrt(TRADING_DAYS))

# ── GARCH-GJR(1,1) ────────────────────────────
print("Estimando GARCH-GJR(1,1)...")
m2       = arch_model(retornos, vol="Garch", p=1, o=1, q=1, dist="normal", rescale=False)
r2       = m2.fit(disp="off")

omega    = r2.params["omega"]
alpha    = r2.params["alpha[1]"]
gamma    = r2.params["gamma[1]"]   # efeito assimétrico (choque negativo)
beta     = r2.params["beta[1]"]
persist  = alpha + gamma/2 + beta

vol_hoje = float(np.sqrt(r2.conditional_volatility.iloc[-1]) * np.sqrt(TRADING_DAYS))
vol_lp   = float(np.sqrt(omega / (1 - persist)) * np.sqrt(TRADING_DAYS))
var_hoje = float(r2.conditional_volatility.iloc[-1]) ** 2
var_lp   = omega / (1 - persist)

# ── PERCENTIL HISTÓRICO ───────────────────────
vol_hist_serie = r2.conditional_volatility * np.sqrt(TRADING_DAYS)
percentil_atual = float((vol_hist_serie < vol_hoje).mean() * 100)

def faixa_percentil(p):
    if p < 25:   return ("Quartil inferior", "#22c55e")
    elif p < 50: return ("Abaixo da mediana", "#84cc16")
    elif p < 75: return ("Acima da mediana",  "#eab308")
    elif p < 90: return ("Quartil superior",  "#f97316")
    else:        return ("Extremo histórico", "#ef4444")

fp_label, fp_color = faixa_percentil(percentil_atual)

# ── PROJEÇÃO GJR — 5 DIAS ─────────────────────
hoje  = pd.Timestamp(datetime.today().date())
datas = pd.bdate_range(start=hoje + pd.offsets.BDay(1), periods=HORIZONTE_DIAS)
proj  = []
for h in range(1, HORIZONTE_DIAS + 1):
    var_h    = var_lp + (persist ** h) * (var_hoje - var_lp)
    vol_d    = float(np.sqrt(var_h))
    vol_a    = vol_d * np.sqrt(TRADING_DAYS)
    amp_pts  = ibov * vol_d / 100
    amp_fin  = amp_pts * 0.20
    # VaR e CVaR diários (% do capital)
    z_var    = norm.ppf(1 - CONFIANCA_VAR)
    var_pct  = abs(z_var * vol_d)
    cvar_pct = abs(norm.pdf(z_var) / (1 - CONFIANCA_VAR) * vol_d)
    var_pts  = ibov * var_pct / 100
    cvar_pts = ibov * cvar_pct / 100
    proj.append({
        "h":        h,
        "data":     datas[h-1].strftime("%d/%m"),
        "dia":      datas[h-1].strftime("%a"),
        "vol_d":    round(vol_d, 4),
        "vol_a":    round(vol_a, 2),
        "amp_pts":  round(amp_pts),
        "amp_fin":  round(amp_fin),
        "var_pct":  round(var_pct, 3),
        "cvar_pct": round(cvar_pct, 3),
        "var_pts":  round(var_pts),
        "cvar_pts": round(cvar_pts),
        "var_fin":  round(var_pts * 0.20),
        "cvar_fin": round(cvar_pts * 0.20),
    })

def regime(v):
    if v < 15:   return ("Baixa",   "#22c55e", "#dcfce7")
    elif v < 22: return ("Normal",  "#eab308", "#fef9c3")
    elif v < 30: return ("Elevada", "#f97316", "#ffedd5")
    else:        return ("Stress",  "#ef4444", "#fee2e2")

prox      = proj[0]
rg_hoje   = regime(vol_hoje)
rg_prox   = regime(prox["vol_a"])
direcao   = "Arrefecendo" if vol_hoje > vol_lp else "Subindo"
dir_icon  = "↘" if direcao == "Arrefecendo" else "↗"
dir_color = "#22c55e" if direcao == "Arrefecendo" else "#f97316"
assimetria = gamma > 0.05

# ── VOL HISTÓRICA (126 pregões) ───────────────
vol_hist    = r2.conditional_volatility.iloc[-126:] * np.sqrt(TRADING_DAYS)
hist_labels = [d.strftime("%d/%m") for d in vol_hist.index]
hist_values = [round(float(v), 2) for v in vol_hist.values]

# ── GARCH(1,1) vol para comparação ────────────
vol_garch_hist    = r1.conditional_volatility.iloc[-126:] * np.sqrt(TRADING_DAYS)
garch_values      = [round(float(v), 2) for v in vol_garch_hist.values]

# ── TABELA SEMANAL ────────────────────────────
rows = ""
for p in proj:
    rg = regime(p["vol_a"])
    rows += f"""
        <tr>
          <td><span class="muted">{p['dia']}</span> {p['data']}</td>
          <td>{p['vol_d']:.3f}%</td>
          <td><span class="badge" style="background:{rg[2]};color:{rg[1]}">{p['vol_a']:.2f}%</span></td>
          <td><span style="color:{rg[1]};font-weight:600">{rg[0]}</span></td>
          <td>±{p['amp_pts']:,} pts</td>
          <td>±R$ {p['amp_fin']:,}</td>
          <td style="color:#f97316">{p['var_pct']:.3f}% / {p['var_pts']:,} pts</td>
          <td style="color:#ef4444">{p['cvar_pct']:.3f}% / {p['cvar_pts']:,} pts</td>
        </tr>"""

html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GARCH-GJR WIN PRO — Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg:      #0a0e1a;
    --surface: #111827;
    --border:  #1f2937;
    --text:    #e5e7eb;
    --muted:   #6b7280;
    --accent:  #3b82f6;
    --green:   #22c55e;
    --yellow:  #eab308;
    --orange:  #f97316;
    --red:     #ef4444;
  }}
  body {{ background: var(--bg); color: var(--text); font-family: 'IBM Plex Sans', sans-serif; min-height: 100vh; padding: 32px 24px; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}

  /* HEADER */
  .header {{ display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 32px; border-bottom: 1px solid var(--border); padding-bottom: 20px; }}
  .header-left h1 {{ font-size: 22px; font-weight: 700; letter-spacing: -0.5px; }}
  .header-left h1 span {{ color: var(--accent); }}
  .header-left h1 .pro {{ color: var(--orange); font-size: 14px; border: 1px solid var(--orange); border-radius: 4px; padding: 2px 8px; margin-left: 8px; vertical-align: middle; }}
  .header-left p {{ color: var(--muted); font-size: 13px; margin-top: 4px; font-family: 'IBM Plex Mono', monospace; }}
  .header-right {{ text-align: right; font-size: 12px; color: var(--muted); font-family: 'IBM Plex Mono', monospace; }}
  .live-badge {{ display: inline-flex; align-items: center; gap: 6px; background: #0d2b1a; border: 1px solid #22c55e33; border-radius: 20px; padding: 4px 12px; font-size: 11px; color: var(--green); }}
  .live-dot {{ width: 6px; height: 6px; border-radius: 50%; background: var(--green); animation: pulse 2s infinite; }}
  @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:0.3}} }}

  /* KPI GRID */
  .kpi-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin-bottom: 20px; }}
  .kpi {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 18px 16px; }}
  .kpi label {{ font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.8px; display: block; margin-bottom: 8px; }}
  .kpi .val {{ font-size: 24px; font-weight: 700; font-family: 'IBM Plex Mono', monospace; line-height: 1; }}
  .kpi .sub {{ font-size: 11px; color: var(--muted); margin-top: 6px; }}

  /* PRÓXIMO PREGÃO */
  .next-card {{ background: var(--surface); border: 1px solid var(--border); border-left: 4px solid {rg_prox[1]}; border-radius: 10px; padding: 22px 24px; margin-bottom: 20px; }}
  .next-card h2 {{ font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-bottom: 16px; }}
  .next-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; }}
  .next-item label {{ font-size: 10px; color: var(--muted); display: block; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}
  .next-item .val {{ font-size: 18px; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }}
  .next-item .val.small {{ font-size: 14px; }}
  .divider {{ width: 1px; background: var(--border); }}

  /* ALERTA GJR */
  .gjr-alert {{ background: #1a1208; border: 1px solid #f97316aa; border-radius: 10px; padding: 14px 20px; margin-bottom: 20px; display: flex; align-items: center; gap: 12px; font-size: 13px; }}
  .gjr-alert .icon {{ font-size: 20px; }}

  /* CARDS */
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 22px 24px; margin-bottom: 20px; }}
  .card h2 {{ font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-bottom: 16px; }}
  .card-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}

  /* PERCENTIL */
  .percentil-bar {{ background: #1a2035; border-radius: 6px; height: 10px; margin: 10px 0; position: relative; overflow: hidden; }}
  .percentil-fill {{ height: 100%; border-radius: 6px; background: linear-gradient(90deg, #22c55e, #eab308, #f97316, #ef4444); width: 100%; }}
  .percentil-marker {{ position: absolute; top: -4px; width: 3px; height: 18px; background: white; border-radius: 2px; left: {percentil_atual:.1f}%; transform: translateX(-50%); }}

  /* TABELA */
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th {{ text-align: left; padding: 8px 10px; font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.6px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
  td {{ padding: 11px 10px; border-bottom: 1px solid var(--border); font-family: 'IBM Plex Mono', monospace; font-size: 12px; }}
  tr:last-child td {{ border-bottom: none; }}
  .badge {{ padding: 3px 8px; border-radius: 20px; font-size: 11px; font-weight: 600; }}
  .muted {{ color: var(--muted); font-size: 10px; }}

  /* PARAMS */
  .params {{ display: flex; gap: 10px; flex-wrap: wrap; }}
  .param {{ background: #1a2035; border: 1px solid var(--border); border-radius: 6px; padding: 7px 12px; font-family: 'IBM Plex Mono', monospace; font-size: 11px; }}
  .param span {{ color: var(--muted); }}
  .param.highlight {{ border-color: var(--orange); }}

  .chart-wrap {{ height: 200px; position: relative; }}
  .footer {{ text-align: center; color: var(--muted); font-size: 11px; margin-top: 32px; font-family: 'IBM Plex Mono', monospace; }}

  @media (max-width: 800px) {{
    .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .next-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .card-row {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="container">

  <!-- HEADER -->
  <div class="header">
    <div class="header-left">
      <h1>GARCH<span>(GJR)</span> — WIN / IBOV <span class="pro">PRO</span></h1>
      <p>^BVSP · {len(fechamento)} pregões · {fechamento.index[0].strftime('%d/%m/%Y')} → {fechamento.index[-1].strftime('%d/%m/%Y')}</p>
    </div>
    <div class="header-right">
      <div class="live-badge"><span class="live-dot"></span> Atualizado às 19h57</div><br>
      <strong style="color:var(--text)">{datetime.now(FUSO_BR).strftime('%d/%m/%Y %H:%M')}</strong>
    </div>
  </div>

  <!-- KPIs -->
  <div class="kpi-grid">
    <div class="kpi">
      <label>IBOV Atual</label>
      <div class="val">{ibov:,.0f}</div>
      <div class="sub">pontos</div>
    </div>
    <div class="kpi">
      <label>Vol Hoje (GJR)</label>
      <div class="val" style="color:{rg_hoje[1]}">{vol_hoje:.2f}%</div>
      <div class="sub">{rg_hoje[0]} · anualizada</div>
    </div>
    <div class="kpi">
      <label>Vol Longo Prazo</label>
      <div class="val" style="color:var(--accent)">{vol_lp:.2f}%</div>
      <div class="sub">média histórica</div>
    </div>
    <div class="kpi">
      <label>Percentil Histórico</label>
      <div class="val" style="color:{fp_color}">{percentil_atual:.0f}º</div>
      <div class="sub">{fp_label}</div>
    </div>
    <div class="kpi">
      <label>Direção</label>
      <div class="val" style="color:{dir_color}">{dir_icon} {direcao}</div>
      <div class="sub">para {vol_lp:.1f}%</div>
    </div>
  </div>

  <!-- ALERTA GJR -->
  {'<div class="gjr-alert"><span class="icon">⚠️</span><div><strong style="color:var(--orange)">Assimetria detectada (GJR ativo)</strong> — O modelo identificou que quedas geram volatilidade significativamente maior que altas. Tome cuidado com posições compradas em cenário de stress.</div></div>' if assimetria else '<div class="gjr-alert" style="border-color:#22c55e88;background:#0d2b1a"><span class="icon">✅</span><div><strong style="color:var(--green)">Assimetria baixa</strong> — O mercado está se comportando de forma relativamente simétrica. Altas e quedas geram volatilidade similar.</div></div>'}

  <!-- PRÓXIMO PREGÃO -->
  <div class="next-card">
    <h2>📅 Próximo Pregão — {prox['dia']} {prox['data']}</h2>
    <div class="next-grid">
      <div class="next-item">
        <label>Vol Projetada (anual)</label>
        <div class="val" style="color:{rg_prox[1]}">{prox['vol_a']:.2f}% <span style="font-size:13px">{rg_prox[0]}</span></div>
      </div>
      <div class="next-item">
        <label>Amplitude Estimada</label>
        <div class="val">±{prox['amp_pts']:,} pts</div>
        <div class="sub">±R$ {prox['amp_fin']:,}/contrato</div>
      </div>
      <div class="next-item">
        <label>VaR 95% (perda máx.)</label>
        <div class="val small" style="color:var(--orange)">{prox['var_pct']:.3f}% · {prox['var_pts']:,} pts</div>
        <div class="sub">R$ {prox['var_fin']:,}/contrato</div>
      </div>
      <div class="next-item">
        <label>CVaR 95% (pior caso)</label>
        <div class="val small" style="color:var(--red)">{prox['cvar_pct']:.3f}% · {prox['cvar_pts']:,} pts</div>
        <div class="sub">R$ {prox['cvar_fin']:,}/contrato</div>
      </div>
      <div class="next-item">
        <label>Assimetria GJR (γ)</label>
        <div class="val small" style="color:{'var(--orange)' if assimetria else 'var(--green)'}">{gamma:.4f} {'↑ Quedas amplificam' if assimetria else '✓ Simétrico'}</div>
      </div>
    </div>
  </div>

  <!-- GRÁFICO + PERCENTIL -->
  <div class="card-row">
    <div class="card">
      <h2>📈 Volatilidade Condicional — 126 Pregões</h2>
      <div class="chart-wrap">
        <canvas id="volChart"></canvas>
      </div>
    </div>
    <div class="card">
      <h2>📊 Percentil Histórico da Volatilidade</h2>
      <div style="margin-top: 16px">
        <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--muted);font-family:'IBM Plex Mono',monospace">
          <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
        </div>
        <div class="percentil-bar">
          <div class="percentil-fill"></div>
          <div class="percentil-marker"></div>
        </div>
        <div style="text-align:center;margin-top:16px">
          <div style="font-size:42px;font-weight:700;font-family:'IBM Plex Mono',monospace;color:{fp_color}">{percentil_atual:.0f}º</div>
          <div style="color:var(--muted);font-size:13px;margin-top:4px">{fp_label}</div>
          <div style="color:var(--muted);font-size:12px;margin-top:8px">A vol atual está acima de {percentil_atual:.0f}% dos últimos 5 anos</div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:20px">
          <div style="background:#1a2035;border-radius:8px;padding:12px;text-align:center">
            <div style="font-size:10px;color:var(--muted);margin-bottom:4px">VOL MÍN. HISTÓRICA</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:16px;color:var(--green)">{float(vol_hist_serie.min()):.2f}%</div>
          </div>
          <div style="background:#1a2035;border-radius:8px;padding:12px;text-align:center">
            <div style="font-size:10px;color:var(--muted);margin-bottom:4px">VOL MÁX. HISTÓRICA</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:16px;color:var(--red)">{float(vol_hist_serie.max()):.2f}%</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- TABELA SEMANAL -->
  <div class="card">
    <h2>📆 Projeção Semanal — GJR + VaR + CVaR</h2>
    <table>
      <thead>
        <tr>
          <th>Data</th>
          <th>Vol Diária</th>
          <th>Vol Anual</th>
          <th>Regime</th>
          <th>Amplitude</th>
          <th>Valor/Contrato</th>
          <th>VaR 95%</th>
          <th>CVaR 95%</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>

  <!-- PARÂMETROS -->
  <div class="card">
    <h2>⚙️ Parâmetros GARCH-GJR(1,1)</h2>
    <div class="params">
      <div class="param"><span>ω</span> {omega:.6f}</div>
      <div class="param"><span>α (choque)</span> {alpha:.4f}</div>
      <div class="param highlight"><span>γ (assimetria)</span> {gamma:.4f}</div>
      <div class="param"><span>β (memória)</span> {beta:.4f}</div>
      <div class="param"><span>persist.</span> {persist:.4f}</div>
      <div class="param"><span>GARCH clássico</span> {vol_garch:.2f}%</div>
      <div class="param"><span>GJR (assimétrico)</span> {vol_hoje:.2f}%</div>
      <div class="param"><span>1 pt WIN =</span> R$ 0,20</div>
    </div>
    <div style="margin-top:14px;font-size:12px;color:var(--muted);line-height:1.6">
      <strong style="color:var(--text)">VaR 95%:</strong> perda máxima esperada em 95% dos dias · 
      <strong style="color:var(--text)">CVaR 95%:</strong> perda média nos piores 5% dos dias · 
      <strong style="color:var(--text)">γ:</strong> efeito assimétrico — quanto quedas amplificam a volatilidade além das altas
    </div>
  </div>

  <div class="footer">GARCH-GJR(1,1) · VaR/CVaR · Percentil Histórico · Nobel Engle 2003 · Dados: Yahoo Finance (^BVSP)</div>
</div>

<script>
const labels  = {json.dumps(hist_labels)};
const gjr     = {json.dumps(hist_values)};
const garch   = {json.dumps(garch_values)};
const lp      = {round(vol_lp, 2)};
const ctx     = document.getElementById('volChart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{
    labels,
    datasets: [
      {{
        label: 'GJR (assimétrico)',
        data: gjr,
        borderColor: '#f97316',
        backgroundColor: 'rgba(249,115,22,0.06)',
        borderWidth: 1.8,
        pointRadius: 0,
        tension: 0.3,
        fill: true,
      }},
      {{
        label: 'GARCH clássico',
        data: garch,
        borderColor: '#3b82f6',
        backgroundColor: 'transparent',
        borderWidth: 1,
        borderDash: [4,3],
        pointRadius: 0,
        tension: 0.3,
      }},
      {{
        label: 'Vol Longo Prazo',
        data: Array(labels.length).fill(lp),
        borderColor: '#4b5563',
        borderWidth: 1,
        borderDash: [8,4],
        pointRadius: 0,
      }}
    ]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ labels: {{ color: '#9ca3af', font: {{ family: 'IBM Plex Mono', size: 10 }} }} }},
      tooltip: {{ mode: 'index', intersect: false }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#4b5563', maxTicksLimit: 8, font: {{ family: 'IBM Plex Mono', size: 10 }} }}, grid: {{ color: '#1f2937' }} }},
      y: {{ ticks: {{ color: '#4b5563', font: {{ family: 'IBM Plex Mono', size: 10 }}, callback: v => v + '%' }}, grid: {{ color: '#1f2937' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"✅ Dashboard PRO salvo: {OUTPUT_HTML}")
print("🌐 Acesse: https://DayaneTeixeira.github.io/garch-win-pro")
