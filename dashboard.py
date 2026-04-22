"""
dashboard.py — Streamlit interactive dashboard for the Pairs Trading system.

Run:
    streamlit run dashboard.py

Dependencies (in addition to requirements.txt):
    pip install streamlit plotly
"""

import sys, os

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError in terminal)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (CANDIDATE_PAIRS, START_DATE, END_DATE,
                    LOOKBACK, ENTRY_STD, EXIT_STD, TRAIN_RATIO,
                    SIGNIFICANCE, TIMEFRAMES, PERIODS_PER_YEAR)
from data_loader import download_universe, get_pair_data
from cointegration import screen_pairs, ols_regression, compute_spread, adf_test
from backtest import run_full_analysis


# ─── Page setup ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pairs Trading · US Market",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-box {
    background:#1e2130; border-radius:8px; padding:12px 16px;
    text-align:center; margin:4px;
}
.metric-label { font-size:0.78rem; color:#aaa; margin-bottom:4px; }
.metric-value { font-size:1.4rem; font-weight:700; }
.green  { color:#00d084; }
.red    { color:#ff4b4b; }
.yellow { color:#ffd700; }
</style>
""", unsafe_allow_html=True)


# ─── Cached loaders ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Downloading market data…")
def cached_download(start: str, end: str) -> pd.DataFrame:
    return download_universe(CANDIDATE_PAIRS, start, end)


@st.cache_data(show_spinner="Screening pairs for cointegration…")
def cached_screen(_data: pd.DataFrame, sig: float) -> pd.DataFrame:
    return screen_pairs(_data, CANDIDATE_PAIRS, sig)


@st.cache_data(show_spinner="Running backtest & forward test…")
def cached_analysis(_data: pd.DataFrame, ty: str, tx: str,
                    lb: int, entry: float, exit_s: float, tr: float) -> dict:
    return run_full_analysis(_data, ty, tx,
                             lookback=lb, entry_std=entry,
                             exit_std=exit_s, train_ratio=tr)


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def color(val: float, good_positive: bool = True) -> str:
    if val > 0:
        return "green" if good_positive else "red"
    return "red" if good_positive else "green"


def metric_html(label: str, value, unit: str = "", good_positive: bool = True) -> str:
    num = float(value) if isinstance(value, (int, float, np.number)) else None
    cls = color(num, good_positive) if num is not None and num != 0 else "yellow"
    return (f'<div class="metric-box">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value {cls}">{value}{unit}</div>'
            f'</div>')


def plot_prices(data: pd.DataFrame, ty: str, tx: str,
                split_date=None) -> go.Figure:
    py = data[ty]
    px = data[tx]
    norm_y = py / py.iloc[0] * 100
    norm_x = px / px.iloc[0] * 100

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=[f"{ty} vs {tx} — Raw Prices",
                        "Normalised (rebased to 100)"],
        vertical_spacing=0.1,
    )
    fig.add_trace(go.Scatter(x=py.index, y=py, name=ty,
                             line=dict(color='#4c9be8', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=px.index, y=px, name=tx,
                             line=dict(color='#f5a623', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=norm_y.index, y=norm_y, name=f"{ty} (norm)",
                             line=dict(color='#4c9be8', width=1.5, dash='dot'),
                             showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=norm_x.index, y=norm_x, name=f"{tx} (norm)",
                             line=dict(color='#f5a623', width=1.5, dash='dot'),
                             showlegend=False), row=2, col=1)

    if split_date:
        sd = str(split_date)[:10]   # "YYYY-MM-DD" — Plotly needs a string on Cloud
        for row in [1, 2]:
            fig.add_vline(x=sd, line_dash='dash',
                          line_color='rgba(255,255,255,0.4)',
                          annotation_text="Train | Test split",
                          annotation_position="top left", row=row, col=1)

    fig.update_layout(height=450, template='plotly_dark', margin=dict(t=40, b=20))
    return fig


def plot_spread(df: pd.DataFrame, period: str, split_date=None) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df['spread'], name='Spread',
                             line=dict(color='white', width=1.2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['ma'], name='Rolling Mean',
                             line=dict(color='yellow', width=1, dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['upper_entry'],
                             name='Upper Entry', fill=None,
                             line=dict(color='rgba(255,80,80,0.7)', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['lower_entry'],
                             name='Lower Entry',
                             fill='tonexty',
                             fillcolor='rgba(255,80,80,0.06)',
                             line=dict(color='rgba(255,80,80,0.7)', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['upper_exit'],
                             name='Upper Exit',
                             line=dict(color='rgba(0,200,130,0.6)', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['lower_exit'],
                             name='Lower Exit',
                             line=dict(color='rgba(0,200,130,0.6)', width=1, dash='dot')))

    # Shade position periods — convert Timestamps to strings for Plotly Cloud
    pos = df['positions']
    long_mask  = pos > 0
    short_mask = pos < 0

    for mask, clr, lbl in [(long_mask,  'rgba(0,200,130,0.15)', 'Long'),
                            (short_mask, 'rgba(255,80,80,0.15)',  'Short')]:
        in_trade = False
        start_i  = None
        for i, (ts, flag) in enumerate(mask.items()):
            if flag and not in_trade:
                in_trade = True
                start_i  = ts
            elif not flag and in_trade:
                in_trade = False
                if start_i != ts:   # guard against zero-width rect
                    fig.add_vrect(x0=str(start_i)[:10], x1=str(ts)[:10],
                                  fillcolor=clr, line_width=0,
                                  annotation_text=lbl if i < 10 else '')
        if in_trade and start_i is not None and start_i != mask.index[-1]:
            fig.add_vrect(x0=str(start_i)[:10], x1=str(mask.index[-1])[:10],
                          fillcolor=clr, line_width=0)

    if split_date:
        fig.add_vline(x=str(split_date)[:10], line_dash='dash',
                      line_color='rgba(255,255,255,0.4)')

    fig.update_layout(title=f"Spread + Bollinger Bands ({period})",
                      height=400, template='plotly_dark',
                      margin=dict(t=40, b=20))
    return fig


def plot_equity(df_train: pd.DataFrame, df_test: pd.DataFrame,
                tf: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_train.index, y=df_train['cumulative_returns'],
        name='Backtest', line=dict(color='#4c9be8', width=2)))

    # Stitch test curve from where train ended
    last_val = float(df_train['cumulative_returns'].dropna().iloc[-1]) \
               if not df_train['cumulative_returns'].dropna().empty else 1.0
    test_cum = df_test['cumulative_returns'].dropna()
    if not test_cum.empty:
        stitched = test_cum / test_cum.iloc[0] * last_val
        fig.add_trace(go.Scatter(
            x=stitched.index, y=stitched,
            name='Forward Test', line=dict(color='#f5a623', width=2, dash='dash')))

    fig.add_hline(y=1.0, line_dash='dot',
                  line_color='rgba(255,255,255,0.3)')

    fig.update_layout(title=f"Equity Curve — {tf}",
                      yaxis_title="Cumulative Return",
                      height=380, template='plotly_dark',
                      margin=dict(t=40, b=20))
    return fig


def metrics_table(m_train: dict, m_test: dict) -> pd.DataFrame:
    labels = {
        'total_return'  : ('Total Return',    '%'),
        'annual_return' : ('Annual Return',   '%'),
        'sharpe_ratio'  : ('Sharpe Ratio',    ''),
        'sortino_ratio' : ('Sortino Ratio',   ''),
        'max_drawdown'  : ('Max Drawdown',    '%'),
        'calmar_ratio'  : ('Calmar Ratio',    ''),
        'win_rate'      : ('Win Rate',        '%'),
        'profit_factor' : ('Profit Factor',   ''),
        'num_trades'    : ('# Trades',        ''),
        'volatility'    : ('Volatility (ann)','%'),
    }
    rows = []
    for k, (label, unit) in labels.items():
        tr = m_train.get(k, '-')
        te = m_test.get(k, '-')
        fmt = lambda v, u: f"{v}{u}" if isinstance(v, (int, float)) else str(v)
        rows.append({'Metric': label,
                     'Backtest': fmt(tr, unit),
                     'Forward Test': fmt(te, unit)})
    return pd.DataFrame(rows)


# ─── Main App ────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Sidebar: global controls ───────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙ Configuration")
        start_date  = st.text_input("Start Date", START_DATE)
        end_date    = st.text_input("End Date",   END_DATE)
        significance = st.selectbox("ADF Significance", [0.01, 0.05, 0.10],
                                    index=2)   # default 10%

        st.markdown("---")
        st.subheader("Strategy Parameters")
        lookback  = st.slider("Lookback (periods)", 5, 120, LOOKBACK)
        entry_std = st.slider("Entry σ",  1.0, 4.0, float(ENTRY_STD), 0.1)
        exit_std  = st.slider("Exit σ",   0.1, 2.0, float(EXIT_STD),  0.1)
        train_ratio = st.slider("Train Ratio", 0.50, 0.90, float(TRAIN_RATIO), 0.05)

        st.markdown("---")
        st.subheader("Pair Selection")
        show_only_coint = st.checkbox("Show only cointegrated pairs", value=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    data = cached_download(start_date, end_date)

    if data.empty:
        st.error("No data downloaded. Check tickers and date range.")
        return

    # ── Screen pairs ──────────────────────────────────────────────────────────
    screened = cached_screen(data, significance)

    if screened.empty:
        st.error("No valid pairs found in data.")
        return

    # ── Pair picker ───────────────────────────────────────────────────────────
    display = screened[screened['cointegrated']] if show_only_coint else screened

    if display.empty:
        st.warning("No cointegrated pairs at this significance level. "
                   "Try unchecking 'Show only cointegrated pairs'.")
        display = screened

    pair_labels = display.apply(
        lambda r: (f"{'✅' if r['cointegrated'] else '❌'}  "
                   f"{r['pair']}   (p={r['p_value']:.4f}, β={r['beta']:.3f})"),
        axis=1
    ).tolist()

    with st.sidebar:
        idx = st.radio("Select a pair", range(len(pair_labels)),
                       format_func=lambda i: pair_labels[i])

    selected   = display.iloc[idx]
    ticker_y   = selected['ticker_y']
    ticker_x   = selected['ticker_x']

    # ── Page title ─────────────────────────────────────────────────────────────
    st.title(f"Pairs Trading — {ticker_y} / {ticker_x}")
    coint_badge = "✅ Cointegrated" if selected['cointegrated'] else "❌ Not Cointegrated"
    st.caption(f"{start_date} → {end_date}  |  {coint_badge}  |  "
               f"β = {selected['beta']}  α = {selected['alpha']}  "
               f"R² = {selected['r_squared']}")

    # ── Run full analysis ─────────────────────────────────────────────────────
    results = cached_analysis(data, ticker_y, ticker_x,
                              lookback, entry_std, exit_std, train_ratio)

    daily_res  = results.get('Daily')
    split_date = daily_res['split_date'] if daily_res else None

    # ──────────────────────────────────────────────────────────────────────────
    tab_ov, tab_coint, tab_bt, tab_ft, tab_scr = st.tabs([
        "📈 Overview",
        "🔬 Cointegration",
        "📊 Backtest",
        "🔮 Forward Test",
        "🗂 Pair Screener",
    ])

    # ── Tab 1: Overview ───────────────────────────────────────────────────────
    with tab_ov:
        if daily_res:
            m_tr = daily_res['train']['metrics']
            m_te = daily_res['test']['metrics']
            cols = st.columns(6)
            for col, (lbl, val, unit, gp) in zip(cols, [
                ("Train Return",   m_tr['total_return'],  '%', True),
                ("Test Return",    m_te['total_return'],  '%', True),
                ("Train Sharpe",   m_tr['sharpe_ratio'],  '',  True),
                ("Test Sharpe",    m_te['sharpe_ratio'],  '',  True),
                ("Train Max DD",   m_tr['max_drawdown'],  '%', False),
                ("Test Max DD",    m_te['max_drawdown'],  '%', False),
            ]):
                col.markdown(metric_html(lbl, val, unit, gp), unsafe_allow_html=True)

        st.plotly_chart(plot_prices(data, ticker_y, ticker_x, split_date),
                        use_container_width=True)

        if daily_res:
            df_tr = daily_res['train']['df']
            df_te = daily_res['test']['df']
            full_df = pd.concat([df_tr, df_te])
            st.plotly_chart(plot_spread(full_df, 'Daily', split_date),
                            use_container_width=True)

    # ── Tab 2: Cointegration ──────────────────────────────────────────────────
    with tab_coint:
        st.subheader("ADF Test — Spread Stationarity")
        st.markdown("""
The **Augmented Dickey-Fuller (ADF)** test checks whether the spread is *stationary*
(mean-reverting) — the statistical foundation of pairs trading.

- **H₀**: Spread has a unit root → random walk → NOT tradable
- **H₁**: Spread is stationary → mean-reverting → pairs trade is valid
- Reject H₀ when **p-value < significance level**
        """)

        for tf_label, res in results.items():
            if res is None:
                st.write(f"**{tf_label}**: insufficient data")
                continue

            with st.expander(f"**{tf_label}** — Train ADF & Test ADF", expanded=(tf_label == 'Daily')):
                c1, c2 = st.columns(2)

                for col, period, adf in [(c1, 'Train', res['train']['adf']),
                                          (c2, 'Test',  res['test']['adf'])]:
                    with col:
                        st.markdown(f"#### {period} Period")
                        verdict_col = "green" if adf['is_stationary'] else "red"
                        st.markdown(
                            f"<span style='color:{verdict_col};font-size:1.1rem;font-weight:700'>"
                            f"{adf['verdict']}</span>", unsafe_allow_html=True)

                        st.markdown(f"""
| Parameter | Value |
|---|---|
| ADF Statistic | `{adf['adf_statistic']}` |
| p-value | `{adf['p_value']}` |
| Lags Used | `{adf['lags_used']}` |
| Observations | `{adf['n_observations']}` |
                        """)

                        st.markdown("**Critical Values:**")
                        crit_df = pd.DataFrame(
                            list(adf['critical_values'].items()),
                            columns=['Level', 'Value']
                        )
                        crit_df['ADF passes?'] = crit_df['Value'].apply(
                            lambda cv: '✅' if adf['adf_statistic'] < cv else '❌')
                        st.dataframe(crit_df, hide_index=True)

                st.markdown("**Spread chart:**")
                spread_df = pd.concat([res['train']['df'], res['test']['df']])
                st.plotly_chart(plot_spread(spread_df, tf_label, res['split_date']),
                                use_container_width=True)

    # ── Tab 3: Backtest ───────────────────────────────────────────────────────
    with tab_bt:
        st.subheader(f"Backtest Results ({int(train_ratio*100)}% of data)")
        st.caption(f"Entry: ±{entry_std}σ  |  Exit: ±{exit_std}σ  |  Lookback: {lookback}")

        tf_sel = st.selectbox("Timeframe", list(TIMEFRAMES.keys()), key="bt_tf")
        res    = results.get(tf_sel)

        if res is None:
            st.warning(f"Insufficient data for {tf_sel} backtest.")
        else:
            m_tr = res['train']['metrics']
            m_te = res['test']['metrics']

            # Equity curve
            st.plotly_chart(plot_equity(res['train']['df'],
                                        res['test']['df'], tf_sel),
                            use_container_width=True)

            # Spread chart (train period only)
            st.plotly_chart(plot_spread(res['train']['df'],
                                        f"{tf_sel} — Backtest", None),
                            use_container_width=True)

            # Metrics table
            st.subheader("Backtest Metrics")
            tbl = metrics_table(m_tr, m_te)
            # Colour code in native st.dataframe with background
            def highlight(row):
                styles = ['', '']
                try:
                    bt_val = float(str(row['Backtest']).replace('%', ''))
                    ft_val = float(str(row['Forward Test']).replace('%', ''))
                    styles[1] = (
                        'background-color: rgba(0,200,130,0.15)'
                        if bt_val > 0 else
                        'background-color: rgba(255,80,80,0.15)'
                    )
                    styles[2 - 1] = styles[1]   # same colour for FT column
                except Exception:
                    pass
                return styles + ['']

            st.dataframe(tbl, hide_index=True, use_container_width=True)

            # OLS summary
            with st.expander("OLS Regression Summary"):
                st.text(res['model'].summary())

    # ── Tab 4: Forward Test ───────────────────────────────────────────────────
    with tab_ft:
        st.subheader(f"Forward Test Results ({int((1-train_ratio)*100)}% of data)")
        st.caption("OLS parameters estimated on training set — applied out-of-sample.")

        tf_sel_ft = st.selectbox("Timeframe", list(TIMEFRAMES.keys()), key="ft_tf")
        res_ft    = results.get(tf_sel_ft)

        if res_ft is None:
            st.warning(f"Insufficient data for {tf_sel_ft} forward test.")
        else:
            m_te = res_ft['test']['metrics']

            # ADF badge
            adf_te = res_ft['test']['adf']
            badge_col = "green" if adf_te['is_stationary'] else "red"
            st.markdown(
                f"ADF p-value (test spread): "
                f"<span style='color:{badge_col}'><b>{adf_te['p_value']}</b>  "
                f"{adf_te['verdict']}</span>",
                unsafe_allow_html=True)

            # Equity curve (test period only)
            dummy_train = res_ft['train']['df'].copy()
            dummy_train['cumulative_returns'] = 1.0  # flat baseline for stitching
            st.plotly_chart(plot_equity(res_ft['train']['df'],
                                        res_ft['test']['df'], tf_sel_ft),
                            use_container_width=True)

            # Spread chart (test period)
            st.plotly_chart(plot_spread(res_ft['test']['df'],
                                        f"{tf_sel_ft} — Forward Test", None),
                            use_container_width=True)

            # Metrics
            st.subheader("Forward Test Metrics")
            metrics_rows = [
                ("Total Return",    m_te['total_return'],   '%'),
                ("Annual Return",   m_te['annual_return'],  '%'),
                ("Sharpe Ratio",    m_te['sharpe_ratio'],   ''),
                ("Sortino Ratio",   m_te['sortino_ratio'],  ''),
                ("Max Drawdown",    m_te['max_drawdown'],   '%'),
                ("Calmar Ratio",    m_te['calmar_ratio'],   ''),
                ("Win Rate",        m_te['win_rate'],       '%'),
                ("Profit Factor",   m_te['profit_factor'],  ''),
                ("# Trades",        m_te['num_trades'],     ''),
                ("Volatility",      m_te['volatility'],     '%'),
            ]

            col_a, col_b = st.columns(2)
            for i, (lbl, val, unit) in enumerate(metrics_rows):
                good_positive = lbl not in ('Max Drawdown',)
                target_col = col_a if i % 2 == 0 else col_b
                target_col.markdown(
                    metric_html(lbl, val, unit, good_positive),
                    unsafe_allow_html=True)

            # Timeframe comparison table
            st.subheader("Forward Test by Timeframe")
            comp_rows = []
            for tf, r in results.items():
                if r is None:
                    comp_rows.append({'Timeframe': tf,
                                      'Total Return': 'N/A',
                                      'Sharpe': 'N/A',
                                      'Max DD': 'N/A',
                                      'ADF p-val': 'N/A',
                                      'Cointegrated?': 'N/A'})
                    continue
                m = r['test']['metrics']
                a = r['test']['adf']
                comp_rows.append({
                    'Timeframe'     : tf,
                    'Total Return'  : f"{m['total_return']}%",
                    'Sharpe'        : m['sharpe_ratio'],
                    'Max DD'        : f"{m['max_drawdown']}%",
                    'Win Rate'      : f"{m['win_rate']}%",
                    '# Trades'      : m['num_trades'],
                    'ADF p-val'     : a['p_value'],
                    'Cointegrated?' : a['verdict'],
                })
            st.dataframe(pd.DataFrame(comp_rows), hide_index=True,
                         use_container_width=True)

    # ── Tab 5: Screener ───────────────────────────────────────────────────────
    with tab_scr:
        st.subheader("US Market Pair Screener")
        st.caption("Cointegration tested on the full date range using ADF on OLS residuals.")

        col_f1, col_f2 = st.columns(2)
        min_r2   = col_f1.slider("Min R²",   0.0, 1.0, 0.5, 0.05)
        max_pval = col_f2.slider("Max p-value", 0.01, 0.20, float(significance), 0.01)

        filtered = screened.copy()
        filtered = filtered[filtered['r_squared'] >= min_r2]
        filtered = filtered[filtered['p_value']   <= max_pval]

        def row_style(row):
            bg = ('background-color: rgba(0,200,130,0.10)'
                  if row['cointegrated'] else
                  'background-color: rgba(255,80,80,0.08)')
            return [bg] * len(row)

        st.dataframe(
            filtered[['pair', 'p_value', 'adf_statistic', 'cointegrated',
                       'alpha', 'beta', 'r_squared', 'correlation', 'n_obs']]
            .style.apply(row_style, axis=1),
            hide_index=True,
            use_container_width=True,
        )

        st.markdown(f"**{len(filtered)}** pairs shown  |  "
                    f"**{filtered['cointegrated'].sum()}** cointegrated")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    main()
