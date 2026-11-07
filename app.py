from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ==========================================
# 1. 시스템 설정 및 UI 스타일
# ==========================================
st.set_page_config(
    page_title="주식 의사결정 매니저 Pro 5차", page_icon="🎯", layout="wide"
)

st.markdown(
    """
    <style>
    .main-title { font-size: 26px; font-weight: 700; color: #f8fafc; margin-bottom: 5px; }
    .sub-title { font-size: 14px; color: #94a3b8; margin-bottom: 20px; }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 16px; border-radius: 10px; border: 1px solid #334155; text-align: center;
    }
    .badge-strategy {
        background-color: #0284c7; color: white; padding: 3px 8px;
        border-radius: 4px; font-size: 11px; font-weight: 600;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="main-title">🎯 주식 의사결정 매니저 Pro (5차 최종 고도화 에디션)</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-title">갭 진입 통제 · 청산 비용 반영 순자산 에퀴티 · 0~100점'
    " 표준화 스코어링</p>",
    unsafe_allow_html=True,
)


# ==========================================
# 2. 보조지표 및 데이터 안정성 함수
# ==========================================
def calculate_wilder_rsi(series, period=14):
  delta = series.diff()
  gain = delta.clip(lower=0)
  loss = -1 * delta.clip(upper=0)
  avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
  avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
  rs = avg_gain / (avg_loss + 1e-10)
  return 100 - (100 / (1 + rs))


def calculate_atr(df, period=14):
  high_low = df["High"] - df["Low"]
  high_cp = (df["High"] - df["Close"].shift(1)).abs()
  low_cp = (df["Low"] - df["Close"].shift(1)).abs()
  tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
  return tr.ewm(alpha=1 / period, adjust=False).mean()


def safe_download(ticker, period="12mo"):
  try:
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df.empty or len(df) < 60:
      return None
    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)

    # 데이터 정제: 결측치 검사, 중복 날짜 제거, 오름차순 정렬
    df = df.loc[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    df = df.dropna(subset=["Close", "Open", "High", "Low", "Volume"])
    return df
  except Exception:
    return None


# ==========================================
# 3. KOSPI & KOSDAQ 독립적 시장 국면 진단
# ==========================================
@st.cache_data(ttl=1800)
def analyze_market_regimes():
  regimes = {}
  last_dates = {}
  try:
    for name, tk in [("KOSPI", "^KS11"), ("KOSDAQ", "^KQ11")]:
      df = safe_download(tk, period="6mo")
      if df is None:
        regimes[name] = "ERROR"
        continue
      close = df["Close"]
      val = float(close.iloc[-1])
      ma20 = float(close.rolling(20).mean().iloc[-1])
      ma60 = float(close.rolling(60).mean().iloc[-1])
      last_dates[name] = df.index[-1].strftime("%Y-%m-%d")

      if val >= ma20 and ma20 >= ma60:
        regimes[name] = "상승장 (Bullish)"
      elif val >= ma20:
        regimes[name] = "중립장 (Neutral)"
      else:
        regimes[name] = "하락장 (Bearish)"
  except Exception:
    regimes = {"KOSPI": "ERROR", "KOSDAQ": "ERROR"}
  return regimes, last_dates


# ==========================================
# 4. 종목 풀 및 표준화된(0~100점) 스크리닝 엔진
# ==========================================
WATCH_POOL = {
    # 코스피
    "삼성전자": ("005930.KS", "KOSPI"),
    "SK하이닉스": ("000660.KS", "KOSPI"),
    "LG에너지솔루션": ("373220.KS", "KOSPI"),
    "현대차": ("005380.KS", "KOSPI"),
    "기아": ("000270.KS", "KOSPI"),
    "셀트리온": ("068270.KS", "KOSPI"),
    "POSCO홀딩스": ("005490.KS", "KOSPI"),
    "KB금융": ("105560.KS", "KOSPI"),
    "NAVER": ("035420.KS", "KOSPI"),
    "HD현대중공업": ("329180.KS", "KOSPI"),
    "한화에어로스페이스": ("012450.KS", "KOSPI"),
    "현대로템": ("064350.KS", "KOSPI"),
    # 코스닥
    "알테오젠": ("196170.KQ", "KOSDAQ"),
    "에코프로비엠": ("247540.KQ", "KOSDAQ"),
    "에코프로": ("086520.KQ", "KOSDAQ"),
    "HLB": ("028300.KQ", "KOSDAQ"),
    "레인보우로보틱스": ("277810.KQ", "KOSDAQ"),
    "리노공업": ("058470.KQ", "KOSDAQ"),
    "클래시스": ("214150.KQ", "KOSDAQ"),
}


@st.cache_data(ttl=1800)
def run_multi_strategy_screening(market_regimes):
  results = []
  price_histories = {}

  for name, (ticker, market) in WATCH_POOL.items():
    m_state = market_regimes.get(market, "ERROR")
    is_market_bearish = "하락장" in m_state or "ERROR" in m_state

    df = safe_download(ticker, period="8mo")
    if df is None or len(df) < 100:
      continue

    if df["Volume"].iloc[-1] == 0 or df["Close"].isna().iloc[-1]:
      continue

    price_histories[name] = df.tail(60)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_p = df["Open"]
    volume = df["Volume"]

    c_price = float(close.iloc[-1])
    p_price = float(close.iloc[-2])
    change_rate = ((c_price - p_price) / p_price) * 100

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma60 = close.rolling(60).mean()

    c_ma20 = float(ma20.iloc[-1])
    c_ma50 = float(ma50.iloc[-1])
    c_ma60 = float(ma60.iloc[-1])

    val_series = (close * volume) / 1e8
    c_val = float(val_series.iloc[-1])
    val_med20 = float(val_series.rolling(20).median().iloc[-1])
    val_spike = (c_val / val_med20) if val_med20 > 0 else 1.0

    rsi = calculate_wilder_rsi(close, 14)
    c_rsi = float(rsi.iloc[-1])

    atr = calculate_atr(df, 14)
    c_atr = float(atr.iloc[-1])

    if is_market_bearish:
      results.append({
          "종목명": name,
          "티커": ticker,
          "시장": market,
          "현재가": round(c_price),
          "등락률(%)": round(change_rate, 2),
          "거래대금(억)": round(c_val, 1),
          "거래대금폭증": round(val_spike, 2),
          "RSI": round(c_rsi, 1),
          "ATR": round(c_atr),
          "포착전략": "시장하락통제 (매수차단)",
          "기회점수": 0.0,
          "위험도": "HIGH",
          "손절가": 0,
          "손절폭(%)": 0.0,
          "1차목표가": 0,
          "손익비": 0.0,
          "기관외인수급": "[데이터 한계: 미구현]",
      })
      continue

    # 표준화된 0~100점 스코어 계산
    # 1. 돌파형 점수 (돌파 강도, 거래대금, 종가 위치, 추세)
    high_20 = float(high.iloc[-21:-1].max())
    close_near_high = (c_price - float(low.iloc[-1])) / (
        float(high.iloc[-1]) - float(low.iloc[-1]) + 1e-10
    ) >= 0.7
    is_breakout = (
        (c_price >= high_20) and (val_spike >= 1.5) and close_near_high
    )
    score_A = (
        min(
            100,
            (
                (min(val_spike, 4.0) / 4.0) * 40
                + (30 if close_near_high else 10)
                + (30 if c_price > c_ma50 else 0)
            ),
        )
        if is_breakout
        else 0.0
    )

    # 2. 눌림목형 점수 (중기 추세, MA 지지, 거래량 감소, RSI 반등)
    uptrend = (c_price > c_ma50) and (c_ma20 >= c_ma60)
    near_ma = abs(c_price - c_ma20) / c_ma20 <= 0.03
    vol_dry = float(volume.iloc[-1]) < float(volume.rolling(5).mean().iloc[-1])
    rsi_turning = c_rsi > float(rsi.iloc[-2]) and float(rsi.iloc[-2]) < 50
    is_pullback = uptrend and near_ma and vol_dry and rsi_turning
    score_B = (
        min(
            100,
            (
                (40 if uptrend else 10)
                + (30 if near_ma else 10)
                + (30 if vol_dry and rsi_turning else 0)
            ),
        )
        if is_pullback
        else 0.0
    )

    # 3. 수급폭발형 점수 (거래대금 증가율, 상승률, 유동성, RSI 과열 여부)
    pct_changes = close.pct_change()
    c_pct = float(pct_changes.iloc[-1]) if not pct_changes.empty else 0.0
    is_surge = (
        (val_spike >= 2.0)
        and (c_pct >= 0.03)
        and (c_val >= 200)
        and (c_rsi < 75)
    )
    score_C = (
        min(
            100,
            (
                (min(val_spike, 5.0) / 5.0) * 40
                + (min(c_pct * 10, 1.0) * 30)
                + (30 if c_rsi < 70 else 10)
            ),
        )
        if is_surge
        else 0.0
    )

    strategies = []
    if score_A > 0:
      strategies.append("돌파형")
    if score_B > 0:
      strategies.append("눌림목")
    if score_C > 0:
      strategies.append("수급폭발")

    if not strategies:
      primary_strat = "일반관심"
      opportunity_score = round(
          min(50.0, (c_val / 500) * 30 + (20 if c_price > c_ma50 else 0)), 1
      )  # 일반관심 종목은 최대 50점으로 제한하여 신호 종목과 구분
    else:
      primary_strat = "/".join(strategies)
      opportunity_score = round(max(score_A, score_B, score_C), 1)

    atr_stop = c_price - (1.8 * c_atr)
    ma_support = c_ma50 * 0.99
    stop_loss = (
        round(max(atr_stop, ma_support))
        if c_price > c_ma50
        else round(atr_stop)
    )
    stop_pct = round(((stop_loss - c_price) / c_price) * 100, 1)

    target_1st = round(c_price + (2.0 * c_atr))
    risk_amt = c_price - stop_loss
    reward_amt = target_1st - c_price
    rr_ratio = round(reward_amt / risk_amt, 2) if risk_amt > 0 else 0.0

    risk_level = (
        "HIGH"
        if c_rsi >= 75 or c_price < c_ma50
        else ("MEDIUM" if val_spike > 4.0 else "LOW")
    )

    results.append({
        "종목명": name,
        "티커": ticker,
        "시장": market,
        "현재가": round(c_price),
        "등락률(%)": round(change_rate, 2),
        "거래대금(억)": round(c_val, 1),
        "거래대금폭증": round(val_spike, 2),
        "RSI": round(c_rsi, 1),
        "ATR": round(c_atr),
        "포착전략": primary_strat,
        "기회점수": opportunity_score,
        "위험도": risk_level,
        "손절가": stop_loss,
        "손절폭(%)": stop_pct,
        "1차목표가": target_1st,
        "손익비": rr_ratio,
        "기관외인수급": "[데이터 한계: 미구현]",
    })

  if not results:
    return pd.DataFrame(), {}

  res_df = pd.DataFrame(results).sort_values(by="기회점수", ascending=False)
  res_df["순위"] = range(1, len(res_df) + 1)
  return res_df.reset_index(drop=True), price_histories


# ==========================================
# 5. 5차 최종 정밀 백테스트 엔진
# ==========================================
def run_precise_backtest(
    ticker,
    market_type="KOSPI",
    strategy_type="돌파형",
    initial_capital=10000000,
    use_market_filter=True,
):
  df = safe_download(ticker, period="12mo")
  if df is None or len(df) < 100:
    return None

  # 백테스트용 시장 지수 데이터 확보 (필터 연동용)
  market_tk = "^KS11" if market_type == "KOSPI" else "^KQ11"
  m_df = safe_download(market_tk, period="12mo")

  close = df["Close"]
  high = df["High"]
  low = df["Low"]
  open_p = df["Open"]
  volume = df["Volume"]

  ma50 = close.rolling(50).mean()
  val_med20 = ((close * volume) / 1e8).rolling(20).median()
  rsi = calculate_wilder_rsi(close, 14)
  atr = calculate_atr(df, 14)
  pct_changes = close.pct_change()

  capital = initial_capital
  cash = initial_capital
  shares = 0
  in_position = False
  entry_price = 0
  stop_price = 0
  target_price = 0
  holding_days = 0

  equity_curve = []
  trades = []
  pending_entry = False
  pending_stop = 0
  pending_target = 0

  # 진단 지표 카운터
  gap_cancel_count = 0
  chase_cancel_count = 0
  risk_cancel_count = 0
  market_block_count = 0

  slippage = 0.002
  buy_fee = 0.00015
  sell_fee_tax = 0.00215  # 매도 수수료 + 증권거래세 총합

  for i in range(60, len(df)):
    c_date = df.index[i]
    c_open = float(open_p.iloc[i])
    c_high = float(high.iloc[i])
    c_low = float(low.iloc[i])
    c_close = float(close.iloc[i])

    # 시장 필터 체크 (해당일 기준 지수 20일선 위 여부)
    is_market_ok = True
    if use_market_filter and m_df is not None:
      sub_m = m_df[m_df.index <= c_date]
      if len(sub_m) >= 20:
        m_val = float(sub_m["Close"].iloc[-1])
        m_ma20 = float(sub_m["Close"].rolling(20).mean().iloc[-1])
        if m_val < m_ma20:
          is_market_ok = False

    # 1. 미체결 진입 대기 신호 처리 (시가 갭 조건 검사)
    if pending_entry and not in_position:
      if not is_market_ok:
        market_block_count += 1
        pending_entry = False
      elif c_open <= pending_stop:
        gap_cancel_count += 1  # 갭 하락으로 손절가 이탈 진입 취소
        pending_entry = False
      elif c_open >= pending_target:
        chase_cancel_count += 1  # 갭 상승으로 목표가 도달 추격매수 취소
        pending_entry = False
      else:
        # 정상 진입 가능한 경우: 실제 진입가 기준 재산정
        entry_price = c_open * (1 + slippage)
        c_atr = float(atr.iloc[i])
        actual_stop = entry_price - (1.8 * c_atr)
        actual_target = entry_price + (2.5 * c_atr)

        stop_pct = (entry_price - actual_stop) / entry_price
        if stop_pct > 0.10:  # 손절폭이 10%를 초과하면 과도한 위험으로 진입 취소
          risk_cancel_count += 1
          pending_entry = False
        else:
          risk_per_share = entry_price - actual_stop
          if risk_per_share > 0:
            max_risk_amt = capital * 0.01
            max_weight_amt = capital * 0.20
            shares_by_risk = int(max_risk_amt // risk_per_share)
            shares_by_weight = int(max_weight_amt // entry_price)
            shares = min(shares_by_risk, shares_by_weight)
            if shares > 0:
              cost = shares * entry_price * (1 + buy_fee)
              if cost <= cash:
                cash -= cost
                stop_price = actual_stop
                target_price = actual_target
                in_position = True
                holding_days = 0
                entry_date = c_date
          pending_entry = False

    # 2. 보유 포지션 청산 조건 검사
    exit_triggered = False
    exit_p = 0
    exit_reason = ""

    if in_position:
      holding_days += 1
      if c_open <= stop_price:
        exit_p = c_open * (1 - slippage)
        exit_reason = "손절(Gap)"
        exit_triggered = True
      elif c_open >= target_price:
        exit_p = c_open * (1 - slippage)
        exit_reason = "익절(Gap)"
        exit_triggered = True
      else:
        hit_stop = c_low <= stop_price
        hit_target = c_high >= target_price

        if hit_stop and hit_target:
          exit_p = stop_price * (1 - slippage)
          exit_reason = "손절(동시도달우선)"
          exit_triggered = True
        elif hit_stop:
          exit_p = stop_price * (1 - slippage)
          exit_reason = "손절"
          exit_triggered = True
        elif hit_target:
          exit_p = target_price * (1 - slippage)
          exit_reason = "익절"
          exit_triggered = True
        elif holding_days >= 20:
          exit_p = c_close * (1 - slippage)
          exit_reason = "시간청산"
          exit_triggered = True

      if exit_triggered:
        proceeds = shares * exit_p * (1 - sell_fee_tax)
        cash += proceeds
        capital = cash
        pnl = proceeds - (shares * entry_price * (1 + buy_fee))
        net_ret = (exit_p - entry_price) / entry_price - 0.0023

        trades.append({
            "entry_date": entry_date,
            "exit_date": c_date,
            "return_pct": round(net_ret * 100, 2),
            "pnl_dollar": round(pnl),
            "result": "WIN" if pnl > 0 else "LOSS",
            "reason": exit_reason,
            "holding_days": holding_days,
        })
        in_position = False
        shares = 0
        holding_days = 0

    # 3. 매일 종가 기준 일별 자산 평가 (총 평가 자산 및 예상 청산 후 순자산 구분)
    gross_equity = cash + (shares * c_close if in_position else 0)
    net_liquidation_equity = cash + (
        shares * c_close * (1 - sell_fee_tax) if in_position else 0
    )
    equity_curve.append({
        "date": c_date,
        "gross_equity": gross_equity,
        "net_equity": net_liquidation_equity,
    })

    # 4. 신규 매수 신호 탐지
    if not in_position and not pending_entry and i < len(df) - 1:
      if use_market_filter and not is_market_ok:
        continue

      c_val = float((close.iloc[i] * volume.iloc[i]) / 1e8)
      v_med = float(val_med20.iloc[i])
      c_rsi = float(rsi.iloc[i])
      c_atr = float(atr.iloc[i])
      c_ma50 = float(ma50.iloc[i])
      c_pct = float(pct_changes.iloc[i]) if not pct_changes.empty else 0.0

      signal = False
      if strategy_type == "돌파형":
        high_20 = float(high.iloc[i - 20 : i].max())
        signal = c_close >= high_20 and (c_val / (v_med + 1e-10)) >= 1.5
      elif strategy_type == "눌림목형":
        signal = (
            c_close > c_ma50
            and abs(c_close - float(close.rolling(20).mean().iloc[i])) / c_close
            <= 0.03
            and c_rsi < 55
        )
      elif strategy_type == "수급폭발형":
        signal = (c_val / (v_med + 1e-10)) >= 2.0 and c_pct >= 0.03

      if signal:
        pending_entry = True
        pending_stop = c_close - (1.8 * c_atr)
        pending_target = c_close + (2.5 * c_atr)

  if not trades and len(equity_curve) == 0:
    return None

  eq_df = pd.DataFrame(equity_curve)
  final_equity = float(eq_df["net_equity"].iloc[-1])
  cum_return = round(
      ((final_equity - initial_capital) / initial_capital) * 100, 2
  )

  total_days = (eq_df["date"].iloc[-1] - eq_df["date"].iloc[0]).days
  years = max(total_days / 365.25, 0.25)
  cagr = round(
      (((final_equity / initial_capital) ** (1 / years)) - 1) * 100, 2
  )

  # MDD는 청산 후 순자산(net_equity) 기준으로 정밀 산출
  eq_df["peak"] = eq_df["net_equity"].cummax()
  eq_df["drawdown"] = (eq_df["net_equity"] - eq_df["peak"]) / eq_df["peak"] * 100
  mdd = round(eq_df["drawdown"].min(), 2)

  tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
  if not tdf.empty:
    wins = len(tdf[tdf["result"] == "WIN"])
    total = len(tdf)
    win_rate = round((wins / total) * 100, 1)

    total_profit_dlr = (
        tdf[tdf["pnl_dollar"] > 0]["pnl_dollar"].sum() if wins > 0 else 0
    )
    total_loss_dlr = (
        abs(tdf[tdf["pnl_dollar"] < 0]["pnl_dollar"].sum())
        if (total - wins) > 0
        else 1
    )
    profit_factor = (
        round(total_profit_dlr / total_loss_dlr, 2)
        if total_loss_dlr > 0
        else 9.99
    )

    avg_win = round(tdf[tdf["result"] == "WIN"]["pnl_dollar"].mean(), 1) if wins > 0 else 0
    avg_loss = round(tdf[tdf["result"] == "LOSS"]["pnl_dollar"].mean(), 1) if (total - wins) > 0 else 0
    avg_hold = round(tdf["holding_days"].mean(), 1)
  else:
    win_rate = 0.0
    profit_factor = 0.0
    total = 0
    avg_win = 0
    avg_loss = 0
    avg_hold = 0

  bench_return = round(
      ((float(close.iloc[-1]) - float(close.iloc[60])) / float(close.iloc[60]))
      * 100,
      2,
  )

  # 미청산 포지션 잔여 여부 확인
  open_positions_count = 1 if in_position else 0
  open_unrealized_pnl = (
      round(shares * float(close.iloc[-1]) * (1 - sell_fee_tax) - (shares * entry_price * (1 + buy_fee)))
      if in_position
      else 0
  )

  return {
      "strategy": strategy_type,
      "total_trades": total,
      "win_rate": win_rate,
      "cum_return": cum_return,
      "cagr": cagr,
      "profit_factor": profit_factor,
      "mdd": mdd,
      "final_capital": round(final_equity),
      "avg_win_dollar": avg_win,
      "avg_loss_dollar": avg_loss,
      "avg_holding_days": avg_hold,
      "benchmark_return": bench_return,
      "gap_cancel": gap_cancel_count,
      "chase_cancel": chase_cancel_count,
      "risk_cancel": risk_cancel_count,
      "market_block": market_block_count,
      "open_positions": open_positions_count,
      "open_unrealized_pnl": open_unrealized_pnl,
      "history": tdf,
      "equity_curve": eq_df,
  }


# ==========================================
# 6. Streamlit 사용자 인터페이스 (UI)
# ==========================================
market_regimes, last_dates = analyze_market_regimes()

st.markdown("### 🌐 KOSPI & KOSDAQ 독립적 시장 국면 진단")
mc1, mc2 = st.columns(2)
with mc1:
  kpi_state = market_regimes.get("KOSPI", "ERROR")
  kpi_date = last_dates.get("KOSPI", "N/A")
  st.metric(
      "KOSPI 시장",
      kpi_state,
      delta=f"최신 데이터: {kpi_date} (지연 없음)"
      if kpi_state != "ERROR"
      else "데이터 지연/오류",
  )
with mc2:
  ksq_state = market_regimes.get("KOSDAQ", "ERROR")
  ksq_date = last_dates.get("KOSDAQ", "N/A")
  st.metric(
      "KOSDAQ 시장",
      ksq_state,
      delta=f"최신 데이터: {ksq_date} (지연 없음)"
      if ksq_state != "ERROR"
      else "데이터 지연/오류",
  )

st.markdown("---")

tab1, tab2, tab3 = st.tabs(
    ["🔍 다중 전략 스크리닝", "🧮 보수적 자금 관리기", "📜 5차 정밀 백테스트"]
)

# ----------------------------------------------------
# TAB 1: 스크리닝
# ----------------------------------------------------
with tab1:
  if st.button(
      "🚀 실시간 전략 스크리닝 실행", type="primary", use_container_width=True
  ):
    with st.spinner(
        "시장 국면 반영 및 0~100점 표준 스코어링 계산 중..."
    ):
      df_res, price_hist = run_multi_strategy_screening(market_regimes)
      if df_res.empty:
        st.warning("조건을 충족하는 종목이 없습니다.")
      else:
        st.session_state["df_res"] = df_res
        st.session_state["price_hist"] = price_hist

  if "df_res" in st.session_state:
    df_res = st.session_state["df_res"]
    price_hist = st.session_state["price_hist"]

    st.markdown("#### 🏆 상위 발굴 유망주 (0~100점 표준화)")
    cols = st.columns(3)
    for i in range(min(3, len(df_res))):
      row = df_res.iloc[i]
      name = row["종목명"]
      with cols[i]:
        st.markdown(
            f"""
                <div class="metric-card">
                    <span class="badge-strategy">{row['포착전략']}</span>
                    <h3 style="color:#ffffff; margin:8px 0px;">{name} ({int(row['순위'])}위)</h3>
                    <p style="color:#cbd5e1; margin:0px;">기회점수: <b style="color:#38bdf8;">{row['기회점수']}점</b> | 시장: <b>{row['시장']}</b></p>
                    <hr style="border-color:#334155; margin:10px 0px;">
                    <p style="color:#cbd5e1; font-size:13px; margin:2px;">현재가: <b>{format(row['현재가'], ',')}원</b></p>
                    <p style="color:#f87171; font-size:13px; margin:2px;">손절가: <b>{format(row['손절가'], ',')}원</b> ({row['손절폭(%)']}%)</p>
                    <p style="color:#4ade80; font-size:13px; margin:2px;">1차목표: <b>{format(row['1차목표가'], ',')}원</b></p>
                </div>
                """,
            unsafe_allow_html=True,
        )

        with st.expander(f"📈 {name} 추세 차트"):
          if name in price_hist:
            fig = px.line(
                price_hist[name], x=price_hist[name].index, y="Close"
            )
            fig.update_layout(
                height=180, margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📋 전체 스크리닝 결과 리스트")
    show_cols = [
        "순위",
        "종목명",
        "시장",
        "포착전략",
        "기회점수",
        "위험도",
        "현재가",
        "등락률(%)",
        "거래대금(억)",
        "손절가",
        "손절폭(%)",
        "1차목표가",
        "손익비",
    ]
    st.dataframe(df_res[show_cols], use_container_width=True, hide_index=True)


# ----------------------------------------------------
# TAB 2: 보수적 자금 관리기
# ----------------------------------------------------
with tab2:
  st.markdown("### 🧮 보수적 포지션 사이징 계산기")
  st.caption(
      "손실 허용 한도와 최대 투자 비중 한도 중 **더 엄격한 조건**을"
      " 자동으로 적용합니다."
  )

  c1, c2 = st.columns(2)
  with c1:
    account_eq = st.number_input(
        "계좌 평가금액 (원)", min_value=1000000, value=10000000, step=1000000
    )
    max_risk_pct = st.slider(
        "1회 거래 허용 최대 손실률 (%)",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1,
    )
  with c2:
    max_weight_pct = st.slider(
        "종목당 최대 투자 비중 한도 (%)", min_value=5.0, max_value=50.0, value=20.0, step=5.0
    )
    entry_p = st.number_input(
        "매수 예정가 (원)", min_value=100, value=70000, step=500
    )
    stop_p = st.number_input("손절가 (원)", min_value=90, value=67500, step=500)

  if entry_p <= stop_p:
    st.error("손절가는 매수가보다 낮아야 합니다.")
  else:
    allowed_risk_amt = account_eq * (max_risk_pct / 100)
    risk_per_share = entry_p - stop_p
    shares_by_risk = int(allowed_risk_amt // risk_per_share)

    max_alloc_amt = account_eq * (max_weight_pct / 100)
    shares_by_weight = int(max_alloc_amt // entry_p)

    final_shares = min(shares_by_risk, shares_by_weight)
    total_invest = final_shares * entry_p
    actual_weight = round((total_invest / account_eq) * 100, 1)

    st.markdown("---")
    r1, r2, r3 = st.columns(3)
    with r1:
      st.metric(
          "최종 산출 매수 수량",
          f"{format(final_shares, ',')} 주",
          delta=(
              "비중 한도 적용됨"
              if shares_by_weight < shares_by_risk
              else "위험 한도 적용됨"
          ),
      )
    with r2:
      st.metric("총 투자 금액", f"{format(total_invest, ',')} 원")
    with r3:
      st.metric(
          "계좌 내 실제 비중",
          f"{actual_weight} %",
          delta="안전" if actual_weight <= max_weight_pct else "주의",
      )


# ----------------------------------------------------
# TAB 3: 5차 정밀 백테스트 검증
# ----------------------------------------------------
with tab3:
  st.markdown("### 📜 5차 정밀 전략 백테스트 검증")
  st.caption(
      "삼성전자(005930.KS) 종목을 대상으로 갭 진입 취소, 순자산(청산 비용"
      " 반영) 에퀴티, 시장 국면 필터 적용 성과를 검증합니다."
  )

  tc1, tc2 = st.columns(2)
  with tc1:
    strat_choice = st.selectbox(
        "백테스트할 전략 선택", ["돌파형", "눌림목형", "수급폭발형"]
    )
  with tc2:
    use_filter = st.checkbox("시장 국면(KOSPI 20일선) 필터 적용", value=True)

  if st.button("📊 5차 정밀 백테스트 실행"):
    with st.spinner(f"{strat_choice} 갭 체결 및 순자산 에퀴티 시뮬레이션 중..."):
      bt = run_precise_backtest(
          "005930.KS",
          market_type="KOSPI",
          strategy_type=strat_choice,
          initial_capital=10000000,
          use_market_filter=use_filter,
      )
      if not bt:
        st.warning("백테스트 데이터가 부족하거나 체결된 거래가 없습니다.")
      else:
        if bt["total_trades"] < 5:
          st.warning(
              "⚠️ **[주의]** 총 거래 횟수가 5회 미만입니다. 표본이 부족하여 성과가"
              " 과도하게 왜곡되었을 수 있으므로 주의해서 해석하세요."
          )

        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
          st.metric("총 거래 횟수", f"{bt['total_trades']} 회")
          st.metric("전략 승률", f"{bt['win_rate']} %")
        with bc2:
          st.metric("누적 수익률 (순자산)", f"{bt['cum_return']} %")
          st.metric("CAGR (연환산)", f"{bt['cagr']} %")
        with bc3:
          st.metric("Profit Factor", f"{bt['profit_factor']}")
          st.metric("최대 낙폭 (MDD)", f"{bt['mdd']} %")
        with bc4:
          st.metric(
              "벤치마크 수익률", f"{bt['benchmark_return']} % (Buy&Hold)"
          )
          st.metric("평균 보유 기간", f"{bt['avg_holding_days']} 일")

        st.markdown(
            f"**최종 청산 후 순자산 평가액:** {format(bt['final_capital'], ',')} 원"
            f" | **미청산 포지션 수:** {bt['open_positions']}개 (평가손익:"
            f" {format(bt['open_unrealized_pnl'], ',')}원)"
        )

        st.markdown(
            "**진단 로그:** 갭하락 진입취소:"
            f" <b>{bt['gap_cancel']}회</b> | 갭상승(추격) 진입취소:"
            f" <b>{bt['chase_cancel']}회</b> | 과도한 리스크 진입취소:"
            f" <b>{bt['risk_cancel']}회</b> | 시장하락 차단:"
            f" <b>{bt['market_block']}회</b>",
            unsafe_allow_html=True,
        )

        st.markdown("#### 📈 일별 순자산 및 총 자산 에퀴티 곡선")
        fig_eq = px.line(
            bt["equity_curve"],
            x="date",
            y=["gross_equity", "net_equity"],
            labels={
                "value": "평가 자산 (원)",
                "date": "날짜",
                "variable": "구분",
            },
            title=(
                "포트폴리오 에퀴티 곡선 (Gross 총 평가액 vs Net 청산 후"
                " 순자산)"
            ),
        )
        fig_eq.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_eq, use_container_width=True)

        st.markdown("#### 📜 상세 거래 내역")
        if not bt["history"].empty:
          st.dataframe(bt["history"], use_container_width=True)
        else:
          st.info("기록된 거래 내역이 없습니다.")
