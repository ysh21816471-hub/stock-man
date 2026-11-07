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
    page_title="주식 의사결정 매니저 Pro 3차", page_icon="🚀", layout="wide"
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
    '<p class="main-title">🚀 주식 의사결정 매니저 Pro (3차 고도화 에디션)</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-title">엄격한 백테스트 엔진 · 보수적 자금 관리 · 독립 시장 국면'
    " 분석 시스템</p>",
    unsafe_allow_html=True,
)


# ==========================================
# 2. 보조지표 및 데이터 정밀 계산 함수
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


def safe_download(ticker, period="8mo"):
  try:
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df.empty or len(df) < 60:
      return None
    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)
    return df
  except Exception:
    return None


# ==========================================
# 3. 독립적 시장 국면 분석 (KOSPI & KOSDAQ)
# ==========================================
@st.cache_data(ttl=1800)
def analyze_market_regime():
  regime = {
      "KOSPI": {"state": "ERROR", "val": 0, "ma20": 0, "ma60": 0},
      "KOSDAQ": {"state": "ERROR", "val": 0, "ma20": 0, "ma60": 0},
      "trading_approved": False,
  }
  try:
    for name, tk in [("KOSPI", "^KS11"), ("KOSDAQ", "^KQ11")]:
      df = safe_download(tk, period="6mo")
      if df is None:
        continue
      close = df["Close"]
      val = float(close.iloc[-1])
      ma20 = float(close.rolling(20).mean().iloc[-1])
      ma60 = float(close.rolling(60).mean().iloc[-1])

      if val >= ma20 and ma20 >= ma60:
        state = "상승장 (Bullish)"
      elif val >= ma20:
        state = "중립장 (Neutral)"
      else:
        state = "하락장 (Bearish)"

      regime[name] = {
          "state": state,
          "val": round(val, 2),
          "ma20": round(ma20, 2),
          "ma60": round(ma60, 2),
      }

    # 적어도 한 시장이 중립 이상이고 데이터 수집 성공 시 승인
    if regime["KOSPI"]["state"] != "ERROR" and regime["KOSDAQ"]["state"] != "ERROR":
      if (
          "하락장" not in regime["KOSPI"]["state"]
          or "하락장" not in regime["KOSDAQ"]["state"]
      ):
        regime["trading_approved"] = True
  except Exception:
    regime["trading_approved"] = False

  return regime


# ==========================================
# 4. 종목 풀 및 다중 전략 스크리닝 엔진
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
def run_multi_strategy_screening():
  results = []
  price_histories = {}

  for name, (ticker, market) in WATCH_POOL.items():
    df = safe_download(ticker, period="8mo")
    if df is None or len(df) < 100:
      continue

    # 거래정지 및 0 거래량 필터
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
    val_mean5 = float(val_series.rolling(5).mean().iloc[-1])
    val_med20 = float(val_series.rolling(20).median().iloc[-1])
    val_spike = (c_val / val_med20) if val_med20 > 0 else 1.0

    rsi = calculate_wilder_rsi(close, 14)
    c_rsi = float(rsi.iloc[-1])

    atr = calculate_atr(df, 14)
    c_atr = float(atr.iloc[-1])

    # ----------------------------------------------------
    # 전략 A: 돌파형 주도주 (Breakout)
    # ----------------------------------------------------
    high_20 = float(high.iloc[-21:-1].max())
    high_60 = float(high.iloc[-61:-1].max())
    close_near_high = (c_price - float(low.iloc[-1])) / (
        float(high.iloc[-1]) - float(low.iloc[-1]) + 1e-10
    ) >= 0.7
    vcp_cond = c_atr < float(atr.rolling(20).mean().iloc[-1])  # 변동성 축소

    is_breakout = (
        (c_price >= high_20) and (val_spike >= 1.5) and close_near_high
    )
    score_A = 75 if is_breakout else 0
    if is_breakout and vcp_cond:
      score_A += 15

    # ----------------------------------------------------
    # 전략 B: 눌림목 매수형 (Pullback) - 엄격한 필터 적용
    # ----------------------------------------------------
    uptrend = (c_price > c_ma50) and (c_ma20 >= c_ma60)
    near_ma = abs(c_price - c_ma20) / c_ma20 <= 0.03
    vol_dry = float(volume.iloc[-1]) < float(volume.rolling(5).mean().iloc[-1])
    rsi_turning = (
        c_rsi > float(rsi.iloc[-2]) and float(rsi.iloc[-2]) < 50
    )  # 50 아래에서 골든크로스 형태 반등
    bullish_candle = c_price > float(open_p.iloc[-1])

    is_pullback = uptrend and near_ma and vol_dry and rsi_turning and bullish_candle
    score_B = 85 if is_pullback else 0

    # ----------------------------------------------------
    # 전략 C: 수급 폭발형 (Volume Surge)
    # ----------------------------------------------------
    is_surge = (
        (val_spike >= 2.0)
        and (change_rate >= 3.0)
        and (c_val >= 200)
        and (c_rsi < 75)
    )  # 단기 과열(75이상) 필터
    score_C = 70 + min(30, val_spike * 5) if is_surge else 0

    # 종합 스코어 및 전략 태그
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
          (c_val / 200) * 10 + (10 if c_price > c_ma50 else 0), 1
      )
    else:
      primary_strat = "/".join(strategies)
      opportunity_score = round(max(score_A, score_B, score_C), 1)

    # 손절 및 목표가 산정 (ATR + MA50 지지선 결합)
    atr_stop = c_price - (1.8 * c_atr)
    ma_support = c_ma50 * 0.99
    stop_loss = (
        round(max(atr_stop, ma_support))
        if c_price > c_ma50
        else round(atr_stop)
    )

    stop_pct = ((stop_loss - c_price) / c_price) * 100
    if stop_pct > -1.5:
      stop_loss = round(c_price * 0.985)
      stop_pct = -1.5
    elif stop_pct < -7.0:  # 손절폭이 너무 넓으면 제한
      stop_loss = round(c_price * 0.93)
      stop_pct = -7.0

    target_1st = round(c_price + (2.0 * c_atr))
    target_2nd = round(c_price + (3.5 * c_atr))

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
        "5일평균대금(억)": round(val_mean5, 1),
        "20일중앙대금(억)": round(val_med20, 1),
        "거래대금폭증": round(val_spike, 2),
        "RSI": round(c_rsi, 1),
        "ATR": round(c_atr),
        "포착전략": primary_strat,
        "기회점수": opportunity_score,
        "위험도": risk_level,
        "손절가": stop_loss,
        "손절폭(%)": round(stop_pct, 1),
        "1차목표가": target_1st,
        "2차목표가": target_2nd,
        "손익비": rr_ratio,
        "기관외인수급": "[데이터 한계: 미구현]",
    })

  if not results:
    return pd.DataFrame(), {}

  res_df = pd.DataFrame(results).sort_values(by="기회점수", ascending=False)
  res_df["순위"] = range(1, len(res_df) + 1)
  return res_df.reset_index(drop=True), price_histories


# ==========================================
# 5. 백테스트 엔진 (복리, 수수료, 갭, 시간청산 반영)
# ==========================================
def run_strategy_backtest(ticker, strategy_type="돌파형", initial_capital=10000000):
  df = safe_download(ticker, period="12mo")
  if df is None or len(df) < 100:
    return None

  close = df["Close"]
  high = df["High"]
  low = df["Low"]
  open_p = df["Open"]
  volume = df["Volume"]

  ma50 = close.rolling(50).mean()
  val_med20 = ((close * volume) / 1e8).rolling(20).median()
  rsi = calculate_wilder_rsi(close, 14)
  atr = calculate_atr(df, 14)

  capital = initial_capital
  trades = []
  equity_curve = [initial_capital]

  in_position = False
  entry_price = 0
  stop_price = 0
  target_price = 0
  holding_days = 0
  max_hold = 20  # 최대 보유 기간(시간 청산)

  for i in range(60, len(df) - 1):
    c_price = float(close.iloc[i])
    c_high = float(high.iloc[i])
    c_low = float(low.iloc[i])
    next_open = float(open_p.iloc[i + 1])
    c_val = float((close.iloc[i] * volume.iloc[i]) / 1e8)
    v_med20 = float(val_med20.iloc[i])
    c_rsi = float(rsi.iloc[i])
    c_atr = float(atr.iloc[i])
    c_ma50 = float(ma50.iloc[i])

    if in_position:
      holding_days += 1
      # 갭 하락 손절 또는 갭 상승 익절 처리 (시초가 반영)
      exit_price = 0
      exit_reason = ""

      if next_open <= stop_price:  # 손절가 이탈 갭 하락
        exit_price = next_open * 0.998  # 슬리피지 반영
        exit_reason = "손절(Gap)"
      elif next_open >= target_price:  # 목표가 도달 갭 상승
        exit_price = next_open * 0.998
        exit_reason = "익절(Gap)"
      elif c_low <= stop_price:
        exit_price = stop_price * 0.998
        exit_reason = "손절"
      elif c_high >= target_price:
        exit_price = target_price * 0.998
        exit_reason = "익절"
      elif holding_days >= max_hold:
        exit_price = c_price * 0.998
        exit_reason = "시간청산"

      if exit_reason:
        # 보수적 동시 충돌 처리: 시초가 기준 손절/익절 동시 발생 시 손절 우선
        if c_low <= stop_price and c_high >= target_price and "Gap" not in exit_reason:
          exit_price = stop_price * 0.998
          exit_reason = "손절(동시충돌우선)"

        # 수수료 및 세금 반영 (매수 0.015%, 매도+세금 0.215%)
        net_return = (exit_price - entry_price) / entry_price - 0.0023
        pnl = capital * 0.1 * net_return  # 10% 비중 베팅 가정
        capital += pnl
        equity_curve.append(capital)

        trades.append({
            "return_pct": round(net_return * 100, 2),
            "result": "WIN" if net_return > 0 else "LOSS",
            "reason": exit_reason,
        })
        in_position = False
        holding_days = 0
      continue

    # 진입 조건 검사
    signal = False
    if strategy_type == "돌파형":
      high_20 = float(high.iloc[i - 20 : i].max())
      signal = c_price >= high_20 and (c_val / (v_med20 + 1e-10)) >= 1.5
    elif strategy_type == "눌림목형":
      signal = (
          c_price > c_ma50
          and abs(c_price - float(close.rolling(20).mean().iloc[i])) / c_price <= 0.03
          and c_rsi < 55
      )
    elif strategy_type == "수급폭발형":
      signal = (c_val / (v_med20 + 1e-10)) >= 2.0 and float(df["Change"].iloc[i]) >= 0.03

    if signal and not in_position:
      entry_price = next_open * 1.002  # 슬리피지 반영 매수
      stop_price = entry_price - (1.8 * c_atr)
      target_price = entry_price + (2.5 * c_atr)
      in_position = True
      holding_days = 0

  if not trades:
    return None

  tdf = pd.DataFrame(trades)
  wins = len(tdf[tdf["result"] == "WIN"])
  total = len(tdf)
  win_rate = round((wins / total) * 100, 1)

  cum_return = round(((capital - initial_capital) / initial_capital) * 100, 2)
  avg_win = tdf[tdf["result"] == "WIN"]["return_pct"].mean() if wins > 0 else 0
  avg_loss = abs(tdf[tdf["result"] == "LOSS"]["return_pct"].mean()) if (total - wins) > 0 else 1
  profit_factor = round((avg_win * wins) / (avg_loss * (total - wins)), 2) if (total - wins) > 0 else 9.99

  eq_series = pd.Series(equity_curve)
  peak = eq_series.cummax()
  mdd = round(((eq_series - peak) / peak).min() * 100, 2)

  return {
      "strategy": strategy_type,
      "total_trades": total,
      "win_rate": win_rate,
      "cum_return": cum_return,
      "profit_factor": profit_factor,
      "mdd": mdd,
      "final_capital": round(capital),
      "history": tdf,
  }


# ==========================================
# 6. Streamlit 사용자 인터페이스 (UI)
# ==========================================
regime = analyze_market_regime()

# 상단 시장 환경 매크로 배너
st.markdown("### 🌐 독립적 시장 국면 진단")
mc1, mc2, mc3 = st.columns(3)

with mc1:
  kpi = regime["KOSPI"]
  st.metric(
      "KOSPI 시장",
      f"{kpi['val']} pt" if kpi["val"] else "데이터 오류",
      delta=kpi["state"],
  )
with mc2:
  ksq = regime["KOSDAQ"]
  st.metric(
      "KOSDAQ 시장",
      f"{ksq['val']} pt" if ksq["val"] else "데이터 오류",
      delta=ksq["state"],
  )
with mc3:
  if regime["trading_approved"]:
    st.success("✅ **[매매 승인]** 시장 환경 양호 (스크리닝 정상 가동)")
  else:
    st.error("⚠️ **[매매 통제]** 주요 시장 하락세 또는 데이터 오류 감지")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(
    ["🔍 다중 전략 스크리닝", "🧮 보수적 자금 관리기", "📜 독립 전략 백테스트"]
)

# ----------------------------------------------------
# TAB 1: 스크리닝
# ----------------------------------------------------
with tab1:
  if st.button("🚀 실시간 전략 스크리닝 실행", type="primary", use_container_width=True):
    with st.spinner("시장 데이터 수집 및 3대 전략 정밀 채점 중..."):
      df_res, price_hist = run_multi_strategy_screening()
      if df_res.empty:
        st.warning("조건을 충족하는 종목이 없습니다.")
      else:
        st.session_state["df_res"] = df_res
        st.session_state["price_hist"] = price_hist

  if "df_res" in st.session_state:
    df_res = st.session_state["df_res"]
    price_hist = st.session_state["price_hist"]

    st.markdown("#### 🏆 상위 발굴 유망주")
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
                    <p style="color:#cbd5e1; margin:0px;">기회점수: <b style="color:#38bdf8;">{row['기회점수']}점</b> | 소속: <b>{row['시장']}</b></p>
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
        "기관외인수급",
    ]
    st.dataframe(df_res[show_cols], use_container_width=True, hide_index=True)


# ----------------------------------------------------
# TAB 2: 보수적 자금 관리 계산기
# ----------------------------------------------------
with tab2:
  st.markdown("### 🧮 보수적 포지션 사이징 계산기")
  st.caption(
      "손실 허용 한도와 최대 투자 비중 한도 중 **더 엄격한(작은) 조건**을"
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
    entry_p = st.number_input("매수 예정가 (원)", min_value=100, value=50000, step=500)
    stop_p = st.number_input("손절가 (원)", min_value=90, value=48000, step=500)

  if entry_p <= stop_p:
    st.error("손절가는 매수가보다 낮아야 합니다.")
  else:
    # 1. 리스크 기준 수량
    allowed_risk_amt = account_eq * (max_risk_pct / 100)
    risk_per_share = entry_p - stop_p
    shares_by_risk = int(allowed_risk_amt // risk_per_share)

    # 2. 비중 한도 기준 수량
    max_alloc_amt = account_eq * (max_weight_pct / 100)
    shares_by_weight = int(max_alloc_amt // entry_p)

    # 더 엄격한(작은) 수량 채택
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
# TAB 3: 독립 전략 백테스트
# ----------------------------------------------------
with tab3:
  st.markdown("### 📜 전략별 독립 백테스트 검증")
  st.caption(
      "삼성전자(005930.KS) 종목을 대상으로 각 전략별 과거 1년 성과(복리,"
      " 수수료, 갭 반영)를 검증합니다."
  )

  strat_choice = st.selectbox(
      "백테스트할 전략 선택", ["돌파형", "눌림목형", "수급폭발형"]
  )

  if st.button("📊 백테스트 실행"):
    with st.spinner(f"{strat_choice} 전략 시뮬레이션 가동 중..."):
      bt = run_strategy_backtest(
          "005930.KS", strategy_type=strat_choice, initial_capital=10000000
      )
      if not bt:
        st.warning("백테스트 데이터가 부족하거나 체결된 거래가 없습니다.")
      else:
        bc1, bc2, bc3, bc4, bc5 = st.columns(5)
        with bc1:
          st.metric("총 거래 횟수", f"{bt['total_trades']} 회")
        with bc2:
          st.metric("전략 승률", f"{bt['win_rate']} %")
        with bc3:
          st.metric("누적 수익률 (복리)", f"{bt['cum_return']} %")
        with bc4:
          st.metric("Profit Factor", f"{bt['profit_factor']}")
        with bc5:
          st.metric("최대 낙폭 (MDD)", f"{bt['mdd']} %")

        st.markdown(f"**최종 자산 평가액:** {format(bt['final_capital'], ',')} 원")
        st.markdown("#### 📜 상세 거래 내역")
        st.dataframe(bt["history"], use_container_width=True)
