from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# 1. 페이지 레이아웃 및 디자인 설정
st.set_page_config(
    page_title="나만의 주식 매니저", page_icon="📈", layout="wide"
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 15px;
        color: #94a3b8;
        margin-bottom: 20px;
    }
    .card-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        text-align: center;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="main-title">🚀 나만의 주식 의사결정 매니저 (고도화 에디션)</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-title">수급, 추세(MA50), 과열지표(RSI), 매크로 리스크 및'
    " 리스크-리워드를 종합 분석합니다.</p>",
    unsafe_allow_html=True,
)


# 2. 시장 매크로(코스피 지수) 상태 체크 함수
@st.cache_data(ttl=3600)
def check_market_macro():
  try:
    df_market = yf.download("^KS11", period="2mo", progress=False)
    if isinstance(df_market.columns, pd.MultiIndex):
      df_market.columns = df_market.columns.get_level_values(0)
    current_idx = float(df_market["Close"].iloc[-1])
    ma20_idx = float(df_market["Close"].rolling(20).mean().iloc[-1])
    is_bullish = current_idx >= ma20_idx
    return is_bullish, current_idx
  except:
    return True, 0.0


# 3. 주도주 수집 및 고도화 분석 함수
@st.cache_data(ttl=3600)
def analyze_stocks():
  WATCH_LIST = {
      "삼성전자": "005930.KS",
      "SK하이닉스": "000660.KS",
      "LG에너지솔루션": "373220.KS",
      "삼성바이오로직스": "207940.KS",
      "현대차": "005380.KS",
      "기아": "000270.KS",
      "셀트리온": "068270.KS",
      "POSCO홀딩스": "005490.KS",
      "KB금융": "105560.KS",
      "신한지주": "055550.KS",
      "NAVER": "035420.KS",
      "카카오": "035720.KS",
      "LG화학": "051910.KS",
      "HD현대중공업": "329180.KS",
      "한화오션": "042660.KS",
      "두산에너빌리티": "034020.KS",
      "HMM": "011200.KS",
      "알테오젠": "196170.KQ",
      "에코프로비엠": "247540.KQ",
      "에코프로": "086520.KQ",
      "HLB": "028300.KQ",
      "레인보우로보틱스": "277810.KQ",
      "JYP Ent.": "035900.KQ",
      "셀트리온제약": "068760.KQ",
      "리노공업": "058470.KQ",
  }

  results = []
  for name, ticker in WATCH_LIST.items():
    try:
      # 지표 계산을 위해 3개월(약 70거래일) 데이터 확보
      df = yf.download(ticker, period="3mo", progress=False)
      if df.empty or len(df) < 50:
        continue

      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

      current_price = float(df["Close"].iloc[-1])
      prev_price = float(df["Close"].iloc[-2])
      change_rate = ((current_price - prev_price) / prev_price) * 100

      # 유동성 및 수급 폭증 배율
      recent_vol = float(df["Volume"].iloc[-1])
      recent_val = current_price * recent_vol / 100000000
      avg_val = (df["Close"] * df["Volume"] / 100000000).mean()
      vol_spike = (recent_val / avg_val) if avg_val > 0 else 1.0

      # 5일 모멘텀 수익률
      ret_5d = (
          (
              (current_price - float(df["Close"].iloc[-5]))
              / float(df["Close"].iloc[-5])
          )
          * 100
          if len(df) >= 5
          else 0.0
      )

      # [고도화 1] 추세 정배열 검증 (50일 이평선 기준)
      ma50 = float(df["Close"].rolling(window=50).mean().iloc[-1])
      is_uptrend = current_price > ma50

      # [고도화 2] RSI (14일) 과열 구간 계산
      delta = df["Close"].diff()
      gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
      loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
      rs = gain / loss
      rsi_series = 100 - (100 / (1 + rs))
      current_rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0

      # [고도화 4] 리스크-리워드 (추천 손절가 -2%, 목표 익절가 +5%)
      stop_loss = round(current_price * 0.98)
      take_profit = round(current_price * 1.05)

      results.append({
          "종목명": name,
          "티커": ticker,
          "현재가": round(current_price),
          "등락률(%)": round(change_rate, 2),
          "당일 거래대금(억)": round(recent_val, 1),
          "거래대금 폭증배율": round(vol_spike, 2),
          "5일 수익률(%)": round(ret_5d, 2),
          "추세정배열": "상승세 (O)" if is_uptrend else "조정/하락세 (X)",
          "RSI지표": round(current_rsi, 1),
          "추천손절가": stop_loss,
          "추천목표가": take_profit,
      })
    except:
      continue

  if not results:
    return pd.DataFrame()

  res_df = pd.DataFrame(results)

  # [고도화 3] 점수 가중치 개편 (100점 만점 구조)
  res_df["유동성점수"] = res_df["당일 거래대금(억)"].rank(pct=True) * 30
  res_df["수급폭증점수"] = res_df["거래대금 폭증배율"].rank(pct=True) * 25
  res_df["모멘텀점수"] = res_df["5일 수익률(%)"].rank(pct=True) * 20

  # 추세 가점 (정배열이면 +15점)
  res_df["추세점수"] = res_df["추세정배열"].apply(lambda x: 15 if "O" in x else 0)

  # RSI 최적 구간 가점 (과열권인 80 이상이거나 침체권인 30 미만이면 감점, 50~70 사이면 만점 10점)
  def calc_rsi_score(rsi):
    if 50 <= rsi <= 70:
      return 10
    elif 30 <= rsi < 50 or 70 < rsi <= 80:
      return 6
    else:
      return 2  # 과열(80초과) 또는 침체(30미만)

  res_df["RSI점수"] = res_df["RSI지표"].apply(calc_rsi_score)

  # 최종 종합 점수 산출
  res_df["종합점수"] = (
      res_df["유동성점수"]
      + res_df["수급폭증점수"]
      + res_df["모멘텀점수"]
      + res_df["추세점수"]
      + res_df["RSI점수"]
  ).round(1)

  res_df = res_df.sort_values(by="종합점수", ascending=False).reset_index(
      drop=True
  )
  res_df["순위"] = res_df.index + 1

  return res_df


# 4. 메인 UI 구성 및 매크로 경고 표시
market_is_bullish, kospi_val = check_market_macro()

if not market_is_bullish:
  st.warning(
      "⚠️ **[시장 매크로 경고]** 현재 코스피 지수가 20일 이동평균선 아래에"
      " 위치해 있습니다. 시장 전체가 하락세이므로 매매 비중을 줄이고 리스크를"
      " 관리하세요!"
  )
else:
  st.success(
      "✅ **[시장 매크로 양호]** 코스피 지수가 20일 이동평균선 위에 있어"
      " 안정적인 매매 환경입니다."
  )

if st.button(
    "📊 고도화 AI 주도주 스크리닝 시작", type="primary", use_container_width=True
):
  with st.spinner(
      "수급, 50일 이평선 추세, RSI 과열도 및 리스토어 수준을 종합 계산 중입니다..."
  ):
    df_result = analyze_stocks()

    if df_result.empty:
      st.error(
          "데이터 분석 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."
      )
    else:
      st.success(f"✨ 분석 완료! (총 {len(df_result)}개 우량 주도주 정밀 스코어링)")

      # TOP 3 카드
      st.markdown("### 🏆 오늘의 TOP 3 유망 주도주")
      top3_cols = st.columns(3)

      for i in range(3):
        if i < len(df_result):
          row = df_result.iloc[i]
          with top3_cols[i]:
            st.markdown(
                f"""
                <div class="card-box">
                    <h3 style="color: #38bdf8; margin: 0px;">🔥 {int(row['순위'])}위</h3>
                    <h2 style="color: #ffffff; margin: 5px 0px;">{row['종목명']}</h2>
                    <p style="color: #94a3b8; font-size: 14px; margin: 0px;">종합 점수: <b style="color: #4ade80; font-size: 18px;">{row['종합점수']}점</b></p>
                    <hr style="border-color: #334155; margin: 10px 0px;">
                    <p style="color: #cbd5e1; font-size: 13px; margin: 2px;">현재가: {format(row['현재가'], ',')}원</p>
                    <p style="color: {'#f87171' if row['등락률(%)'] < 0 else '#4ade80'}; font-size: 13px; margin: 2px;">등락률: {row['등락률(%)']}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander(f"💡 {row['종목명']} 고도화 선정 이유"):
              st.write(
                  f"**1. 유동성 및 수급:** 당일 거래대금"
                  f" **{row['당일 거래대금(억)']}억** (폭증배율:"
                  f" {row['거래대금 폭증배율']}배)"
              )
              st.write(
                  f"**2. 추세 정배열:** {row['추세정배열']} (50일선 기준)"
              )
              st.write(
                  f"**3. 과열 지표(RSI):** {row['RSI지표']}점 (과열 구간 회피"
                  " 여부 확인)"
              )
              st.markdown("---")
              st.info(
                  f"🛡️ **리스크 가이드:** 손절가 **{format(row['추천손절가'], ',')}원**"
                  f" (-2%) / 목표가 **{format(row['추천목표가'], ',')}원**"
                  " (+5%)"
              )

      st.markdown("<br>", unsafe_allow_html=True)

      # Plotly 인터랙티브 차트
      st.markdown("### 📊 TOP 10 종목 종합 점수 비교")
      top10_df = df_result.head(10).sort_values(by="종합점수", ascending=True)

      fig = px.bar(
          top10_df,
          x="종합점수",
          y="종목명",
          orientation="h",
          text="종합점수",
          color="종합점수",
          color_continuous_scale="Viridis",
      )

      fig.update_layout(
          plot_bgcolor="rgba(0,0,0,0)",
          paper_bgcolor="rgba(0,0,0,0)",
          font=dict(color="#ffffff"),
          xaxis=dict(showgrid=True, gridcolor="#334155", title="종합 점수"),
          yaxis=dict(title=""),
          coloraxis_showscale=False,
          height=380,
          margin=dict(l=20, r=20, t=20, b=20),
      )
      fig.update_traces(
          texttemplate="%{text}점",
          textposition="outside",
          hovertemplate="<b>%{y}</b><br>종합점수: %{x}점<extra></extra>",
      )

      st.plotly_chart(fig, use_container_width=True)

      # 전체 추천 순위 리스트 표 (손절가/목표가 포함)
      st.markdown("### 📋 전체 추천 순위 리스트 및 리스크 가이드")
      display_df = df_result[[
          "순위",
          "종목명",
          "현재가",
          "등락률(%)",
          "추세정배열",
          "RSI지표",
          "추천손절가",
          "추천목표가",
          "종합점수",
      ]].copy()

      st.dataframe(
          display_df, use_container_width=True, hide_index=True
      )
else:
  st.info(
      "👆 상단의 **[고도화 AI 주도주 스크리닝 시작]** 버튼을 누르면 시장 매크로"
      " 점검부터 종목 정밀 랭킹까지 분석이 시작됩니다."
  )
