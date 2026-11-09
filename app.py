"""
APEX-RACELINE GO-KART SIMULATOR  |  SYRTI V72M300C320 reluctance motor
ONLY INPUT: throttle voltage 0-5 V  (0 V none, 2.5 V half, 5 V full)
Outputs (time graphs): kart acceleration, total tractive force, wheel torque (= F x r),
motor torque (= wheel torque / (GR x eta)), motor power.
Run:  py -m streamlit run app.py
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Apex-Raceline Go-kart simulator", page_icon="🏎️", layout="wide")

# ----------------------- fixed vehicle data (notebook + your inputs) -----------------------
MASS = 220.0                    # vehicle mass with driver [kg]            (PDF)
SPROCKET_DRIVING, SPROCKET_DRIVEN = 13, 39
GR = SPROCKET_DRIVEN / SPROCKET_DRIVING      # gear ratio 3 : 1                (PDF)
R, ETA = 0.1333, 0.93           # wheel radius [m], transmission efficiency     (PDF)
CRR, CD, AREA, RHO = 0.01, 0.8, 0.71, 1.2    # rolling coeff., drag coeff., frontal area [m2], air density (PDF)
CDA = CD * AREA
MU = 0.9                        # tyre grip (assumed, not in PDF)
V0, RINT, CAP = 72.0, 0.03, 1.44             # pack volts, ohm, kWh (assumed; DC bus 72 V from datasheet)
DURATION = 30.0                 # fixed simulated time [s]
T_PEAK, P_PEAK = 46.493, 9.5    # peak motor torque [Nm] (PDF) / datasheet peak power [kW]
P_RATED = 5.3                   # rated (continuous) motor power [kW]
# ---- reference values printed in the PDF ----
PDF_AXLE_T = T_PEAK * GR * ETA              # 129.72 Nm
PDF_F = PDF_AXLE_T / R                      # 973.11 N
PDF_AXLE_P = P_RATED * ETA                  # 4.929 kW
PDF_RR = CRR * MASS * 9.81                  # 21.58 N
PDF_AIR60 = 0.5 * RHO * CD * AREA * (60 / 3.6) ** 2   # 94.67 N
PDF_T60_PEAK, PDF_T60_CONT = 5.68, 8.28     # theoretical 0-60 km/h estimates [s]

# ----------------------- motor data (peak_power_test_dataset.xlsx, built in) -----------------------
RPM = np.array([0, 503.1, 1005.75, 1504.8, 2004.75, 2505.15, 3004.65, 3504.15, 4003.2, 4500.9, 5001.3, 5500.8, 6000.75, 6501.15, 7002, 7501.5, 8001.45])
TQ = np.array([46.493, 46.493, 45.954, 43.97, 35.904, 29.786, 25.238, 21.844, 18.991, 16.675, 14.821, 12.644, 10.178, 8.102, 6.739, 5.792, 5.286])
PIN = np.array([4.174, 6.692, 8.743, 8.863, 8.841, 8.86, 8.897, 8.837, 8.783, 8.735, 8.279, 7.391, 6.551, 6.026, 5.739, 5.696])
POUT = np.array([2.449, 4.84, 6.929, 7.538, 7.815, 7.941, 8.016, 7.961, 7.86, 7.762, 7.283, 6.396, 5.516, 4.941, 4.551, 4.429])
LOSS = np.r_[1.7, PIN - POUT]   # kW motor loss at peak torque
CE = np.array([80, 86.047, 90.545, 92.976, 94.956, 96.115, 96.826, 97.14, 97.405, 97.533, 97.633, 97.714, 97.719, 97.786, 97.841, 97.847, 97.857])
G, RAD = 9.81, 2 * math.pi / 60


def peak_t(n):
    return 0.0 if n > 8001 else float(np.interp(n, RPM, TQ))


def cont_t(n):
    """S2-30 min rated torque: 12.5 Nm, then 5.3 kW (used for thermal derating after the 30 s peak)."""
    return 0.0 if n > 8001 else min(12.5, 5300 * 9.549 / max(n, 1.0))


def pwr_t(n):
    """PDF 'continuous power' case: peak torque curve limited to 5.3 kW."""
    return 0.0 if n > 8001 else min(peak_t(n), P_RATED * 1000 * 9.549 / max(n, 1.0))


@st.cache_data
def simulate(volt: float, case: str = "peak", dt=0.01, dec=10) -> pd.DataFrame:
    m = MASS
    Fcrr, Fmax = CRR * m * G, MU * 0.6 * m * G
    f = volt / 5.0
    v = x = E = th = 0.0
    Vb = V0
    rows = []
    for i in range(int(DURATION / dt) + 1):
        n = v / R * GR / RAD
        neq = n * 72.0 / Vb
        pk, ct = peak_t(neq), cont_t(neq)
        s = min(max((1 - th) / 0.05, 0.0), 1.0)          # 30 s peak -> continuous derating
        Tav = ct + (pk - ct) * s if case == "peak" else pwr_t(neq)
        T = min(f * T_PEAK, Tav)                            # throttle % of 46.493 Nm peak, limited by motor
        Ft = min(T * GR * ETA / R, Fmax)
        Fres = (Fcrr if v > 0.01 else min(Fcrr, Ft)) + 0.5 * RHO * CDA * v * v
        a = (Ft - Fres) / m
        Pm = T * n * RAD / 1000
        if T > 0:
            loss = np.interp(n, RPM, LOSS) * (0.25 + 0.75 * (T / max(peak_t(n), 1.0)) ** 2)
            Pdc = (Pm + loss) / (np.interp(n, RPM, CE) / 100)
        else:
            loss = Pdc = 0.0
        Vb = (V0 + math.sqrt(max(V0 ** 2 - 4 * RINT * Pdc * 1000, 0))) / 2
        if i % dec == 0:
            rows.append(dict(t=i * dt, thr=volt, kmh=v * 3.6, acc=a, Ft=Ft, Fres=Fres, Tw=Ft * R, T=T, Tav=Tav,
                             n=n, Pm=Pm, Pdc=Pdc, I=Pdc * 1000 / Vb))
        th += dt * (loss / 0.58 - th) / 74.0
        v = max(0.0, v + a * dt)
    return pd.DataFrame(rows)


# ----------------------- style -----------------------
st.markdown("""
<style>
.hero{background:linear-gradient(135deg,#111 0%,#2b2b2b 62%,#f2b705 170%);border-radius:14px;padding:18px 26px;color:#fff;border-bottom:6px solid #f2b705}
.hero h1{margin:0;font-size:2.2rem;letter-spacing:1px;color:#fff;padding:0}
.hero p{margin:4px 0 0;color:#ddd;font-size:.95rem}
.stripe{height:10px;background:repeating-linear-gradient(90deg,#fff 0 20px,#111 20px 40px);border-radius:4px;margin:10px 0 14px}
div[data-testid="stMetric"]{background:rgba(242,183,5,.08);border:1px solid rgba(242,183,5,.35);border-left:5px solid #f2b705;border-radius:10px;padding:10px 14px}
div.stButton>button,div.stDownloadButton>button{border-radius:10px;font-weight:700;border:1.5px solid #f2b705;transition:all .15s}
div.stButton>button:hover,div.stDownloadButton>button:hover{background:#f2b705;color:#111;border-color:#f2b705;transform:translateY(-2px)}
.pill{display:inline-block;padding:5px 14px;border-radius:999px;font-weight:700;font-size:.9rem;color:#fff;margin:4px 6px 4px 0}
.ok{background:#1f8f4e}.bad{background:#c62828}.info{background:#2f7de1}
.spec{border:1px solid rgba(128,128,128,.35);border-radius:10px;padding:10px 14px;text-align:center}
.spec b{display:block;font-size:1.4rem;color:#f2b705}
</style>
<div class="hero"><h1>🏎️ Apex-Raceline Go-kart simulator</h1>
<p>⚡ SYRTI V72M300C320 reluctance motor · 72 V · throttle voltage in, race data out</p></div>
<div class="stripe"></div>
""", unsafe_allow_html=True)

# ----------------------- throttle input: preset buttons + slider -----------------------
if "V" not in st.session_state:
    st.session_state["V"] = 5.0


def set_v(x):
    st.session_state["V"] = x


PRESETS = [("🅿️ Idle · 0 V", 0.0), ("🐢 Cruise · 1.25 V", 1.25), ("🚗 Half · 2.5 V", 2.5),
           ("🚀 Sport · 3.75 V", 3.75), ("🏁 Full · 5 V", 5.0)]
for col, (label, val) in zip(st.columns(len(PRESETS)), PRESETS):
    col.button(label, on_click=set_v, args=(val,), width="stretch")
V = st.slider("🎚️ Throttle voltage (V)", 0.0, 5.0, key="V", step=0.05,
              help="0 V = no throttle, 2.5 V = half, 5 V = full")
st.caption(f"Throttle = {V / 5 * 100:.0f} %  ·  vehicle {MASS:.0f} kg with driver  ·  gear ratio {GR:.0f}:1 ({SPROCKET_DRIVEN}/{SPROCKET_DRIVING} teeth)  ·  wheel radius {R} m  ·  η {ETA}")
S = simulate(round(V, 2))
hit = S.loc[S.kmh >= 60, "t"]

# ----------------------- gauges -----------------------
def gauge(val, title, rng, unit, color, thr=None):
    ax = dict(range=rng, tickwidth=1)
    g = dict(axis=ax, bar=dict(color=color), bgcolor="rgba(128,128,128,0.15)")
    if thr:
        g["threshold"] = dict(line=dict(color="#c62828", width=4), value=thr)
    return go.Indicator(mode="gauge+number", value=val, title=dict(text=title), number=dict(suffix=unit), gauge=g)


gf = make_subplots(rows=1, cols=3, specs=[[{"type": "indicator"}] * 3])
gf.add_trace(gauge(V / 5 * 100, "🎚️ Throttle", [0, 100], " %", "#f2b705"), row=1, col=1)
gf.add_trace(gauge(S.kmh.max(), "🏎️ Speed reached", [0, 120], " km/h", "#2f7de1"), row=1, col=2)
gf.add_trace(gauge(S.Pm.max(), "⚡ Peak motor power", [0, 12], " kW", "#d6452f", thr=P_PEAK), row=1, col=3)
gf.update_layout(height=230, margin=dict(l=30, r=30, t=50, b=10))
st.plotly_chart(gf, width="stretch")

# ----------------------- peak value cards -----------------------
c = st.columns(6)
c[0].metric("🚀 Peak acceleration", f"{S.acc.max():.2f} m/s²")
c[1].metric("💪 Peak tractive force", f"{S.Ft.max():.0f} N")
c[2].metric("🛞 Peak wheel torque (F × r)", f"{S.Tw.max():.1f} Nm")
c[3].metric("🔧 Peak motor torque", f"{S['T'].max():.1f} Nm")
c[4].metric("⚡ Peak motor power", f"{S.Pm.max():.2f} kW")
c[5].metric("⏱️ 0–60 km/h", f"{hit.iloc[0]:.2f} s" if len(hit) else "not reached")

met = len(hit) and hit.iloc[0] <= PDF_T60_PEAK * 1.10
st.markdown(
    f'<span class="pill {"ok" if met else "bad"}">{"✔" if met else "✘"} 0–60 km/h within 10 % of PDF estimate ({PDF_T60_PEAK} s)</span>'
    f'<span class="pill {"ok" if S["T"].max() <= T_PEAK else "bad"}">{"✔" if S["T"].max() <= T_PEAK else "✘"} motor torque ≤ 46.493 Nm peak</span>'
    f'<span class="pill {"ok" if S.Pm.max() <= P_PEAK else "bad"}">{"✔" if S.Pm.max() <= P_PEAK else "✘"} motor power ≤ 9.5 kW</span>'
    f'<span class="pill info">🔋 peak battery {S.Pdc.max():.1f} kW · {S.I.max():.0f} A</span>', unsafe_allow_html=True)

tab_g, tab_d, tab_n = st.tabs(["📈 Race graphs", "📋 Data & CSV export", "📝 Notebook & motor"])

# ----------------------- graphs -----------------------
with tab_g:
    titles = ["🎚️ Throttle voltage (input) [V]", "🏎️ Vehicle speed [km/h]", "🚀 Kart acceleration [m/s²]",
              "💪 Tractive force vs resistance [N]", "🛞 Wheel torque = F × r [Nm]", "🔧 Motor torque [Nm]",
              "⚙️ Motor speed [rpm]", "⚡ Motor power [kW]", "📉 Torque–speed: motor limit + kart path"]
    fig = make_subplots(rows=3, cols=3, subplot_titles=titles, vertical_spacing=0.11)
    B, Rd = "#2f7de1", "#d6452f"

    def add(col_name, r, c_, name, color=B, dash=None):
        fig.add_trace(go.Scatter(x=S.t, y=S[col_name], name=name, line=dict(color=color, width=2, dash=dash), showlegend=False), row=r, col=c_)

    add("thr", 1, 1, "Throttle"); add("kmh", 1, 2, "Speed"); add("acc", 1, 3, "Acceleration")
    add("Ft", 2, 1, "Tractive force"); add("Fres", 2, 1, "Resistance", Rd)
    add("Tw", 2, 2, "Wheel torque")
    add("T", 2, 3, "Delivered"); add("Tav", 2, 3, "Motor limit", Rd, "dash")
    add("n", 3, 1, "rpm"); add("Pm", 3, 2, "Shaft power"); add("Pdc", 3, 2, "Battery power", Rd)
    nn = np.arange(0, 8001, 100)
    fig.add_trace(go.Scatter(x=nn, y=[peak_t(n) for n in nn], line=dict(color=Rd, dash="dash"), showlegend=False), row=3, col=3)
    fig.add_trace(go.Scatter(x=nn, y=[pwr_t(n) for n in nn], line=dict(color="#2aa876", dash="dot"), showlegend=False), row=3, col=3)
    fig.add_trace(go.Scatter(x=S.n, y=S["T"], line=dict(color=B, width=3), showlegend=False), row=3, col=3)
    fig.update_xaxes(title_text="Time (s)", title_font_size=10)
    fig.update_xaxes(title_text="Motor speed (rpm)", row=3, col=3)
    fig.update_layout(height=1000, margin=dict(l=40, r=20, t=50, b=40))
    st.plotly_chart(fig, width="stretch")
    st.caption("Blue = main signal. Red = resistance / battery power / motor limit. The kart starts from standstill and accelerates "
               "for 30 s at the chosen throttle voltage. Mass, gear ratio, wheel radius, drag and rolling resistance come from Powertrain_Calculation.pdf; tyre grip and the battery are assumed.")

# ----------------------- data + CSV export -----------------------
with tab_d:
    COLS = {"t": "Time (s)", "thr": "Throttle voltage (V)", "kmh": "Speed (km/h)", "acc": "Acceleration (m/s2)",
            "Ft": "Tractive force (N)", "Fres": "Resistance force (N)", "Tw": "Wheel torque (Nm)",
            "T": "Motor torque delivered (Nm)", "Tav": "Motor torque available (Nm)", "n": "Motor speed (rpm)",
            "Pm": "Motor shaft power (kW)", "Pdc": "Battery power (kW)", "I": "Battery current (A)"}
    readings = S.rename(columns=COLS).round(4)
    summary = pd.DataFrame({
        "Quantity": ["Throttle voltage (V)", "Peak acceleration (m/s2)", "Peak tractive force (N)", "Peak wheel torque (Nm)",
                     "Peak motor torque (Nm)", "Peak motor power (kW)", "0-60 km/h time (s)", "Top speed in run (km/h)"],
        "Value": [V, round(S.acc.max(), 3), round(S.Ft.max(), 1), round(S.Tw.max(), 2), round(S["T"].max(), 2),
                  round(S.Pm.max(), 3), round(hit.iloc[0], 2) if len(hit) else "not reached", round(S.kmh.max(), 1)]})
    tag = f"{V:.2f}V".replace(".", "p")
    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("⬇️ Download readings (CSV)", readings.to_csv(index=False).encode("utf-8"),
                       file_name=f"kart_readings_{tag}.csv", mime="text/csv", width="stretch")
    d2.download_button("⬇️ Download summary (CSV)", summary.to_csv(index=False).encode("utf-8"),
                       file_name=f"kart_summary_{tag}.csv", mime="text/csv", width="stretch")
    st.dataframe(readings, width="stretch", height=380)

# ----------------------- PDF reference + motor specs -----------------------
with tab_n:
    st.markdown("**📝 Powertrain calculation sheet (Powertrain_Calculation.pdf)**")
    st.table(pd.DataFrame({
        "Item": ["Gear ratio", "Peak axle torque", "Peak tractive force", "Axle power (5.3 kW rated)",
                 "Rolling resistance", "Air resistance at 60 km/h"],
        "Formula": [f"{SPROCKET_DRIVEN} / {SPROCKET_DRIVING}", f"{T_PEAK} × {GR:.0f} × {ETA}", f"{PDF_AXLE_T:.2f} / {R}",
                    f"{P_RATED} × {ETA}", f"{CRR} × {MASS:.0f} × 9.81", f"0.5 × {RHO} × {CD} × {AREA} × 16.67²"],
        "Result": [f"{GR:.0f} : 1", f"{PDF_AXLE_T:.2f} Nm", f"{PDF_F:.2f} N", f"{PDF_AXLE_P:.3f} kW",
                   f"{PDF_RR:.2f} N", f"{PDF_AIR60:.2f} N"]}))
    st.markdown("**⏱️ 0–60 km/h at full throttle (5 V): PDF estimate vs this simulation**")
    rows = []
    for label, case, ref in [("Peak motor torque curve", "peak", PDF_T60_PEAK), ("5.3 kW continuous power limit", "cont", PDF_T60_CONT)]:
        h = simulate(5.0, case).loc[lambda d: d.kmh >= 60, "t"]
        rows.append({"Case": label, "PDF estimate": f"{ref:.2f} s", "This simulation": f"{h.iloc[0]:.2f} s" if len(h) else "not reached in 30 s"})
    st.table(pd.DataFrame(rows))
    st.caption("The 129.72 Nm axle torque (peak case) and 4.929 kW axle power (continuous case) are different operating cases and are not "
               "simultaneous values. The PDF times are theoretical; the simulation also includes the measured torque fall-off with rpm, "
               "battery sag and the 30 s thermal limit. Tyre grip and battery data are assumed.")
    st.markdown("**⚡ SYRTI V72M300C320 datasheet**")
    specs = [("46 Nm", "Peak torque (30 s)"), ("9.5 kW", "Peak power (30 s)"), ("12.5 Nm", "Rated torque"),
             ("5.3 kW", "Rated power"), ("8000 rpm", "Peak speed"), ("72 V", "DC bus"), ("93 %", "Peak efficiency")]
    for col, (big, small) in zip(st.columns(len(specs)), specs):
        col.markdown(f'<div class="spec"><b>{big}</b>{small}</div>', unsafe_allow_html=True)
