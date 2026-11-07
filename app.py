from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ==========================================
# 1. 시스템 설정 및 CSS 레이아웃
# ==========================================
st.set_page_config(
    page_title="주식 의사결정 매니저 Pro", page_icon="⚖️", layout="wide"
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
    .status-ok { color: #4ade80; font-weight: bold; }
    .status-warn { color: #f87171; font-weight: bold; }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="main-title">⚖️ 주식 의사결정 매니저 Pro (2차 고도화 에디션)</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-title">다중 전략 스크리닝 · 정밀 리스크 산출 · 계좌 자금'
    " 관리 · 백테스트 검증 시스템</p>",
    unsafe_allow_html=True,
)


# ==========================================
# 2. 보조지표 정밀 계산 함수 모듈
# ==========================================
def calculate_wilder_rsi(series, period=14):
  """와일더(J. Welles Wilder) 지수이동평균 방식 표준 RSI 계산"""
  delta = series.diff()
  gain = delta.clip(lower=0)
  loss = -1 * delta.clip(upper=0)

  avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
  avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

  rs = avg_gain / (avg_loss + 1e-10)
  rsi = 100 - (100 / (1 + rs))
  return rsi


def calculate_atr(df, period=14):
  """ATR (Average True Range) 변동성 지표 산출"""
  high_low = df["High"] - df["Low"]
  high_cp = (df["High"] - df["Close"].shift(1)).abs()
  low_cp = (df["Low"] - df["Close"].shift(1)).abs()
  tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
  atr = tr.ewm(alpha=1 / period, adjust=False).mean()
  return atr


# ==========================================
# 3. 시장 매크로 독립 분석 (KOSPI & KOSDAQ)
# ==========================================
@st.cache_data(ttl=1800)
def analyze_market_macro():
  """코스피와 코스닥 지수를 독립적으로 분리하여 상태 진단"""
  macro_status = {
      "KOSPI": {"ok": False, "val": 0, "ma20": 0, "ma60": 0, "msg": ""},
      "KOSDAQ": {"ok": False, "val": 0, "ma20": 0, "ma60": 0, "msg": ""},
      "overall_trading_allowed": False,
      "error": None,
  }
  try:
    tickers = {"KOSPI": "^KS11", "KOSDAQ": "^KQ11"}
    for key, tk in tickers.items():
      df = yf.download(tk, period="6mo", progress=False)
      if df.empty or len(df) < 60:
        macro_status[key]["msg"] = "데이터 수집 실패"
        continue

      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

      close = df["Close"]
      c_val = float(close.iloc[-1])
      ma20 = float(close.rolling(20).mean().iloc[-1])
      ma60 = float(close.rolling(60).mean().iloc[-1])

      is_ok = (c_val >= ma20) and (ma20 >= ma60)
      macro_status[key] = {
          "ok": is_ok,
          "val": round(c_val, 2),
          "ma20": round(ma20, 2),
          "ma60": round(ma60, 2),
          "msg": (
              "정배열 상승장"
              if is_ok
              else ("20일선 하회 (경고)" if c_val < ma20 else "60일선 조정")
          ),
      }

    # 두 시장 중 하나라도 20일선 위에 있으면 제한적 매매 허용
    macro_status["overall_trading_allowed"] = (
        macro_status["KOSPI"]["ok"] or macro_status["KOSDAQ"]["ok"]
    )

  except Exception as e:
    macro_status["error"] = f"매크로 데이터 수집 중 에러: {str(e)}"
    macro_status["overall_trading_allowed"] = False

  return macro_status


# ==========================================
# 4. 다중 전략 종목 스크리닝 엔진
# ==========================================
WATCH_POOL = {
    # 코스피 대형주 및 주도주
    "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS",
    "LG에너지솔루션": "373220.KS",
    "삼성바이오로직스": "207940.KS",
    "현대차": "005380.KS",
    "기아": "000270.KS",
    "셀트리온": "068270.KS",
    "KB금융": "105560.KS",
    "NAVER": "035420.KS",
    "HD현대중공업": "329180.KS",
    "한화오션": "042660.KS",
    "두산에너빌리티": "034020.KS",
    "한화에어로스페이스": "012450.KS",
    "현대로템": "064350.KS",
    "삼성전기": "009150.KS",
    # 코스닥 성장주 및 주도주
    "알테오젠": "196170.KQ",
    "에코프로비엠": "247540.KQ",
    "에코프로": "086520.KQ",
    "HLB": "028300.KQ",
    "레인보우로보틱스": "277810.KQ",
    "리노공업": "058470.KQ",
    "삼천당제약": "000250.KQ",
    "클래시스": "214150.KQ",
    "파두": "440110.KQ",
    "원익IPS": "240810.KQ",
}


@st.cache_data(ttl=1800)
def run_multi_strategy_screening():
  screened_results = []
  raw_price_history = {}

  for name, ticker in WATCH_POOL.items():
    try:
      df = yf.download(ticker, period="8mo", progress=False)
      if df.empty or len(df) < 120:
        continue

      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

      # 결측치 및 거래정지 체크
      if df["Volume"].iloc[-1] == 0 or df["Close"].isna().iloc[-1]:
        continue

      raw_price_history[name] = df.tail(60)

      close = df["Close"]
      high = df["High"]
      low = df["Low"]
      volume = df["Volume"]

      c_price = float(close.iloc[-1])
      p_price = float(close.iloc[-2])
      change_rate = ((c_price - p_price) / p_price) * 100

      # 이동평균선
      ma20 = close.rolling(20).mean()
      ma50 = close.rolling(50).mean()
      ma60 = close.rolling(60).mean()

      c_ma20 = float(ma20.iloc[-1])
      c_ma50 = float(ma50.iloc[-1])
      c_ma60 = float(ma60.iloc[-1])

      # 거래대금 (억 원)
      val_series = (close * volume) / 1e8
      c_val = float(val_series.iloc[-1])
      val_ma5 = float(val_series.rolling(5).mean().iloc[-1])
      val_ma20 = float(val_series.rolling(20).mean().iloc[-1])
      val_spike = (c_val / val_ma20) if val_ma20 > 0 else 1.0

      # 지표 계산
      rsi_series = calculate_wilder_rsi(close, 14)
      c_rsi = float(rsi_series.iloc[-1])

      atr_series = calculate_atr(df, 14)
      c_atr = float(atr_series.iloc[-1])

      # ----------------------------------------------------
      # 전략 A: 돌파형 주도주 (Breakout Strategy)
      # ----------------------------------------------------
      high_20 = float(high.iloc[-21:-1].max())
      high_60 = float(high.iloc[-61:-1].max())
      is_breakout = (c_price >= high_20) and (c_val >= val_ma20 * 1.5)
      score_A = 0
      if is_breakout:
        score_A = 70 + (30 if c_price >= high_60 else 15)

      # ----------------------------------------------------
      # 전략 B: 눌림목 매수형 (Pullback Strategy)
      # ----------------------------------------------------
      is_uptrend = (c_price > c_ma50) and (c_ma20 >= c_ma60)
      near_ma20 = abs(c_price - c_ma20) / c_ma20 <= 0.03
      vol_dropped = (
          float(volume.iloc[-1]) < float(volume.iloc[-2])
          and float(volume.iloc[-1]) < float(volume.rolling(5).mean().iloc[-1])
      )
      rsi_rebound = 40 <= c_rsi <= 60

      is_pullback = is_uptrend and near_ma20 and rsi_rebound
      score_B = 0
      if is_pullback:
        score_B = 65 + (20 if vol_dropped else 10) + (15 if c_rsi >= 50 else 5)

      # ----------------------------------------------------
      # 전략 C: 수급 폭발형 (Volume Surge Strategy)
      # ----------------------------------------------------
      is_volume_surge = (
          (val_spike >= 2.0)
          and (change_rate >= 2.5)
          and (c_val >= 200)  # 최소 거래대금 200억 이상
      )
      score_C = 0
      if is_volume_surge:
        score_C = 60 + min(40, (val_spike * 10) + (change_rate * 2))

      # ----------------------------------------------------
      # 주 전략 판정 및 종합 기회 점수 계산
      # ----------------------------------------------------
      matched_strategies = []
      if score_A > 0:
        matched_strategies.append("돌파형")
      if score_B > 0:
        matched_strategies.append("눌림목")
      if score_C > 0:
        matched_strategies.append("수급폭발")

      if not matched_strategies:
        # 세 전략 모두 미충족 시 기본 모멘텀 후보로만 평가
        primary_strat = "일반관심"
        opportunity_score = round(
            (c_val / 100) * 10 + (15 if c_price > c_ma50 else 0), 1
        )
      else:
        primary_strat = "/".join(matched_strategies)
        opportunity_score = round(max(score_A, score_B, score_C), 1)

      # ----------------------------------------------------
      # 위험도 평가 (Risk Score & Classification)
      # ----------------------------------------------------
      risk_factors = 0
      if c_rsi >= 75:
        risk_factors += 1  # 단기 과열
      if c_price < c_ma50:
        risk_factors += 1  # 추세 하향
      if val_spike > 5.0 and change_rate > 15.0:
        risk_factors += 1  # 단기 차익실현 노출

      risk_level = (
          "HIGH"
          if risk_factors >= 2
          else ("MEDIUM" if risk_factors == 1 else "LOW")
      )

      # ----------------------------------------------------
      # 동적 손절가 및 1, 2차 목표가 정밀 산출
      # ----------------------------------------------------
      atr_stop = c_price - (1.8 * c_atr)
      ma_support = c_ma50 * 0.99
      # 지지선 적용: MA50 위에 주가가 있다면 ATR 손절가와 지지선 중 더 안전한(높은) 가격 채택
      if c_price > c_ma50:
        stop_loss = round(max(atr_stop, ma_support))
      else:
        stop_loss = round(atr_stop)

      # 최소 손절 폭 보정 (-1.5% ~ -7.0% 제한)
      stop_pct = ((stop_loss - c_price) / c_price) * 100
      if stop_pct > -1.5:
        stop_loss = round(c_price * 0.985)
        stop_pct = -1.5

      target_1st = round(c_price + (2.0 * c_atr))
      target_2nd = round(c_price + (3.5 * c_atr))

      target_1st_pct = round(((target_1st - c_price) / c_price) * 100, 1)
      target_2nd_pct = round(((target_2nd - c_price) / c_price) * 100, 1)

      # 손익비 (Risk-Reward Ratio)
      risk_amount = c_price - stop_loss
      reward_amount = target_1st - c_price
      rr_ratio = (
          round(reward_amount / risk_amount, 2) if risk_amount > 0 else 0.0
      )

      screened_results.append({
          "종목명": name,
          "티커": ticker,
          "현재가": round(c_price),
          "등락률(%)": round(change_rate, 2),
          "거래대금(억)": round(c_val, 1),
          "거래대금폭증배율": round(val_spike, 2),
          "RSI": round(c_rsi, 1),
          "ATR": round(c_atr),
          "포착전략": primary_strat,
          "기회점수": opportunity_score,
          "위험도": risk_level,
          "손절가": stop_loss,
          "손절폭(%)": round(stop_pct, 1),
          "1차목표가": target_1st,
          "1차목표폭(%)": target_1st_pct,
          "2차목표가": target_2nd,
          "2차목표폭(%)": target_2nd_pct,
          "손익비": rr_ratio,
      })

    except Exception:
      continue

  if not screened_results:
    return pd.DataFrame(), {}

  res_df = pd.DataFrame(screened_results)
  res_df = res_df.sort_values(by="기회점수", ascending=False).reset_index(
      drop=True
  )
  res_df["순위"] = res_df.index + 1
  return res_df, raw_price_history


# ==========================================
# 5. 백테스팅 검증 엔진 (Historical Tester)
# ==========================================
@st.cache_data(ttl=3600)
def run_strategy_backtest(ticker, months=12):
  """선택 종목에 대한 과거 1년 일봉 기준 벡터화 백테스트 실행"""
  try:
    df = yf.download(ticker, period=f"{months}mo", progress=False)
    if df.empty or len(df) < 100:
      return None

    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    ma50 = close.rolling(50).mean()
    val_series = (close * volume) / 1e8
    val_ma20 = val_series.rolling(20).mean()
    rsi_series = calculate_wilder_rsi(close, 14)
    atr_series = calculate_atr(df, 14)

    trades = []
    in_position = False
    entry_price = 0
    stop_price = 0
    target_price = 0
    entry_date = None

    for i in range(60, len(df) - 1):
      c_price = float(close.iloc[i])
      c_high = float(high.iloc[i])
      c_low = float(low.iloc[i])
      c_val = float(val_series.iloc[i])
      v_ma20 = float(val_ma20.iloc[i])
      c_rsi = float(rsi_series.iloc[i])
      c_atr = float(atr_series.iloc[i])
      c_ma50 = float(ma50.iloc[i])

      # 매수 신호 탐지 (돌파 OR 수급폭발)
      high_20 = float(high.iloc[i - 20 : i].max())
      signal_buy = (
          (c_price >= high_20 and c_val >= v_ma20 * 1.5)  # 돌파
          or (c_val >= v_ma20 * 2.0 and c_price > c_ma50)  # 수급폭발
      )

      if not in_position and signal_buy:
        # 익일 시가 매수 가정
        entry_price = float(df["Open"].iloc[i + 1])
        entry_date = df.index[i + 1]
        stop_price = entry_price - (1.8 * c_atr)
        target_price = entry_price + (2.5 * c_atr)
        in_position = True
        continue

      if in_position:
        # 손절/익절 체결 여부 판단
        if c_low <= stop_price:
          ret = ((stop_price - entry_price) / entry_price) * 100
          trades.append({
              "entry_date": entry_date,
              "exit_date": df.index[i],
              "return": ret,
              "result": "LOSS",
          })
          in_position = False
        elif c_high >= target_price:
          ret = ((target_price - entry_price) / entry_price) * 100
          trades.append({
              "entry_date": entry_date,
              "exit_date": df.index[i],
              "return": ret,
              "result": "WIN",
          })
          in_position = False

    if not trades:
      return None

    tdf = pd.DataFrame(trades)
    total_trades = len(tdf)
    wins = len(tdf[tdf["result"] == "WIN"])
    win_rate = round((wins / total_trades) * 100, 1)

    total_return = tdf["return"].sum()
    avg_win = (
        tdf[tdf["result"] == "WIN"]["return"].mean()
        if wins > 0
        else 0
    )
    avg_loss = (
        abs(tdf[tdf["result"] == "LOSS"]["return"].mean())
        if (total_trades - wins) > 0
        else 1
    )
    profit_factor = (
        round((avg_win * wins) / (avg_loss * (total_trades - wins)), 2)
        if (total_trades - wins) > 0
        else 9.99
    )

    # MDD 계산
    tdf["cum_return"] = (1 + tdf["return"] / 100).cumprod()
    tdf["peak"] = tdf["cum_return"].cummax()
    tdf["drawdown"] = (tdf["cum_return"] - tdf["peak"]) / tdf["peak"] * 100
    mdd = round(tdf["drawdown"].min(), 2)

    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "cum_return": round(total_return, 2),
        "profit_factor": profit_factor,
        "mdd": mdd,
        "trade_history": tdf,
    }
  except Exception:
    return None


# ==========================================
# 6. Streamlit 메인 UI 레이아웃
# ==========================================
macro_info = analyze_market_macro()

# 상단 시장 상태 매크로 배너
st.markdown("### 🌐 시장 환경 및 위험 지수 진단")
m_col1, m_col2, m_col3 = st.columns(3)

with m_col1:
  kpi = macro_info["KOSPI"]
  st.metric(
      label="KOSPI 지수",
      value=f"{kpi['val']} pt",
      delta=f"20MA: {kpi['ma20']} | {kpi['msg']}",
  )

with m_col2:
  ksq = macro_info["KOSDAQ"]
  st.metric(
      label="KOSDAQ 지수",
      value=f"{ksq['val']} pt",
      delta=f"20MA: {ksq['ma20']} | {ksq['msg']}",
  )

with m_col3:
  if macro_info["overall_trading_allowed"]:
    st.success("✅ **[매매 승인]** 정상적인 스크리닝 및 진입 가능")
  else:
    st.error("⚠️ **[경고]** 두 시장 모두 20일선 하회. 현금 비중 극대화 권장")

st.markdown("---")

# 탭 메뉴 구성
tab1, tab2, tab3 = st.tabs(
    ["🔍 다중 전략 스크리닝", "🧮 정밀 계좌 자금관리기", "📜 전략 백테스트 검증"]
)

# ----------------------------------------------------
# TAB 1: 다중 전략 종목 스크리닝
# ----------------------------------------------------
with tab1:
  if st.button(
      "🚀 실시간 전략 스크리닝 실행", type="primary", use_container_width=True
  ):
    with st.spinner("다중 전략(돌파/눌림목/수급폭발) 및 위험도를 정밀 분석 중..."):
      df_res, price_hist = run_multi_strategy_screening()

      if df_res.empty:
        st.warning("현재 스크리닝 요건을 통과한 종목이 없습니다.")
      else:
        st.session_state["screened_df"] = df_res
        st.session_state["price_hist"] = price_hist

  if "screened_df" in st.session_state:
    df_res = st.session_state["screened_df"]
    price_hist = st.session_state["price_hist"]

    st.markdown("#### 🏆 상위 발굴 종목 리스트")

    # 상위 TOP 3 종목 카드
    top3_cols = st.columns(3)
    for i in range(min(3, len(df_res))):
      row = df_res.iloc[i]
      s_name = row["종목명"]
      with top3_cols[i]:
        st.markdown(
            f"""
                <div class="metric-card">
                    <span class="badge-strategy">{row['포착전략']}</span>
                    <h3 style="color:#ffffff; margin:8px 0px;">{s_name} ({int(row['순위'])}위)</h3>
                    <p style="color:#cbd5e1; margin:0px;">기회점수: <b style="color:#38bdf8; font-size:18px;">{row['기회점수']}점</b> | 위험도: <b>{row['위험도']}</b></p>
                    <hr style="border-color:#334155; margin:10px 0px;">
                    <p style="color:#cbd5e1; font-size:13px; margin:2px;">현재가: <b>{format(row['현재가'], ',')}원</b> ({row['등락률(%)']}%)</p>
                    <p style="color:#f87171; font-size:13px; margin:2px;">손절가: <b>{format(row['손절가'], ',')}원</b> ({row['손절폭(%)']}%)</p>
                    <p style="color:#4ade80; font-size:13px; margin:2px;">1차목표가: <b>{format(row['1차목표가'], ',')}원</b> (+{row['1차목표폭(%)']}%)</p>
                </div>
                """,
            unsafe_allow_html=True,
        )

        with st.expander(f"📈 {s_name} 차트 & 상세정보"):
          if s_name in price_hist:
            fig = px.line(
                price_hist[s_name],
                x=price_hist[s_name].index,
                y="Close",
                title=f"{s_name} 최근 60일 추세",
            )
            fig.update_layout(
                height=200, margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
          st.write(f"- RSI(14): **{row['RSI']}**")
          st.write(f"- ATR(14): **{format(row['ATR'], ',')}원**")
          st.write(f"- 손익비(R/R): **{row['손익비']}**")

    st.markdown("<br>", unsafe_allow_html=True)

    # 전체 데이터 표 출력
    st.markdown("#### 📋 전체 스크리닝 결과")
    show_cols = [
        "순위",
        "종목명",
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
# TAB 2: 계좌 리스크 및 정밀 매수 수량 계산기
# ----------------------------------------------------
with tab2:
  st.markdown("### 🧮 자금 관리 (Position Sizing) 계산기")
  st.caption(
      "계좌 전체 자산 대비 허용 위험(%)을 기반으로 종목별 정밀 매수 수량을"
      " 계산합니다."
  )

  c_col1, c_col2 = st.columns(2)
  with c_col1:
    total_capital = st.number_input(
        "총 계좌 자산 (원)",
        min_value=1000000,
        max_value=10000000000,
        value=10000000,
        step=1000000,
    )
    max_account_risk_pct = st.slider(
        "1회 매매 시 계좌 최대 허용 손실률 (%)",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1,
    )

  with c_col2:
    target_entry_price = st.number_input(
        "매수 예정가 (원)", min_value=100, value=70000, step=500
    )
    target_stop_price = st.number_input(
        "손절가 (원)", min_value=90, value=67500, step=500
    )

  if target_entry_price <= target_stop_price:
    st.error("손절가는 매수가보다 낮아야 합니다.")
  else:
    # 정밀 수량 계산
    max_risk_amount = total_capital * (max_account_risk_pct / 100)
    risk_per_share = target_entry_price - target_stop_price
    buy_shares = int(max_risk_amount // risk_per_share)
    total_buy_amount = buy_shares * target_entry_price
    portfolio_weight = round((total_buy_amount / total_capital) * 100, 1)

    st.markdown("---")
    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
      st.metric(
          label="최대 허용 손실금", value=f"{format(int(max_risk_amount), ',')} 원"
      )
    with res_col2:
      st.metric(
          label="추천 매수 수량",
          value=f"{format(buy_shares, ',')} 주",
          delta=f"총 매수금액: {format(int(total_buy_amount), ',')}원",
      )
    with res_col3:
      st.metric(
          label="계좌 내 투자 비중",
          value=f"{portfolio_weight} %",
          delta="안전 수준" if portfolio_weight <= 30 else "비중 과다 (주의)",
      )


# ----------------------------------------------------
# TAB 3: 과거 전략 백테스트 검증
# ----------------------------------------------------
with tab3:
  st.markdown("### 📜 개별 종목 과거 전략 검증 (백테스팅)")
  st.caption("최근 1년 일봉 데이터를 바탕으로 알고리즘 매매 성과를 검증합니다.")

  selected_stock_name = st.selectbox(
      "검증할 종목 선택", list(WATCH_POOL.keys())
  )
  selected_ticker = WATCH_POOL[selected_stock_name]

  if st.button("📊 백테스트 실행"):
    with st.spinner(f"{selected_stock_name} 과거 성과 검증 중..."):
      bt_res = run_strategy_backtest(selected_ticker, months=12)

      if not bt_res:
        st.warning("백테스트를 위한 과거 매매 신호가 충분하지 않습니다.")
      else:
        b_col1, b_col2, b_col3, b_col4 = st.columns(4)
        with b_col1:
          st.metric(label="총 거래 횟수", value=f"{bt_res['total_trades']} 회")
        with b_col2:
          st.metric(label="전략 승률", value=f"{bt_res['win_rate']} %")
        with b_col3:
          st.metric(
              label="Profit Factor (손익비)",
              value=f"{bt_res['profit_factor']}",
          )
        with b_col4:
          st.metric(
              label="최대 낙폭 (MDD)",
              value=f"{bt_res['mdd']} %",
              delta_color="normal",
          )

        st.markdown("#### 📜 상세 거래 기록")
        st.dataframe(bt_res["trade_history"], use_container_width=True)
