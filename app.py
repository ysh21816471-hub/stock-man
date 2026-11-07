from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    '<p class="main-title">🚀 나만의 주식 의사결정 매니저 (ATR 변동성 에디션)</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-title">종목별 실제 변동성(ATR 14일)과 지지선(MA50)을'
    " 바탕으로 정밀한 동적 손절·익절가를 산출합니다.</p>",
    unsafe_allow_html=True,
)


# 2. 코스피 지수 매크로 체크
@st.cache_data(ttl=3600)
def check_market_macro():
  try:
    df_market = yf.download("^KS11", period="2mo", progress=False)
    if isinstance(df_market.columns, pd.MultiIndex):
      df_market.columns = df_market.columns.get_level_values(0)
    current_idx = float(df_market["Close"].iloc[-1])
    ma20_idx = float(df_market["Close"].rolling(20).mean().iloc[-1])
    return current_idx >= ma20_idx, current_idx
  except:
    return True, 0.0


# 3. 정밀 변동성(ATR) 기반 데이터 수집 및 분석 함수
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
  raw_price_data = {}

  for name, ticker in WATCH_LIST.items():
    try:
      df = yf.download(ticker, period="3mo", progress=False)
      if df.empty or len(df) < 50:
        continue

      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

      raw_price_data[name] = df.tail(22)  # 최근 1개월 데이터

      current_price = float(df["Close"].iloc[-1])
      prev_price = float(df["Close"].iloc[-2])
      change_rate = ((current_price - prev_price) / prev_price) * 100

      recent_vol = float(df["Volume"].iloc[-1])
      recent_val = current_price * recent_vol / 100000000
      avg_val = (df["Close"] * df["Volume"] / 100000000).mean()
      vol_spike = (recent_val / avg_val) if avg_val > 0 else 1.0

      ret_5d = (
          (
              (current_price - float(df["Close"].iloc[-5]))
              / float(df["Close"].iloc[-5])
          )
          * 100
          if len(df) >= 5
          else 0.0
      )

      # 50일 이동평균선
      ma50 = float(df["Close"].rolling(window=50).mean().iloc[-1])
      is_uptrend = current_price > ma50

      # RSI (14일)
      delta = df["Close"].diff()
      gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
      loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
      rs = gain / loss
      rsi_series = 100 - (100 / (1 + rs))
      current_rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0

      # [핵심 로직] ATR(14일) 변동성 기반 손절/익절 정밀 산출
      high_low = df["High"] - df["Low"]
      high_cp = (df["High"] - df["Close"].shift(1)).abs()
      low_cp = (df["Low"] - df["Close"].shift(1)).abs()
      tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
      atr14 = float(tr.rolling(14).mean().iloc[-1])

      # 기본 ATR 멀티플라이어 (손절 -1.5 ATR / 익절 +2.5 ATR)
      atr_stop = current_price - (1.5 * atr14)
      atr_target = current_price + (2.5 * atr14)

      # 50일선 이평선이 손절선 근처에 있으면 지지선(-0.5%)으로 정밀 보정
      if current_price > ma50 and (current_price - ma50) <= (2.0 * atr14):
        stop_loss = round(min(atr_stop, ma50 * 0.995))
      else:
        stop_loss = round(atr_stop)

      # RSI 가속 시 익절 목표 상향
      if current_rsi >= 65:
        take_profit = round(current_price + (3.0 * atr14))
      else:
        take_profit = round(atr_target)

      # 손절/익절 비율(%) 산출
      stop_loss_pct = round(
          ((stop_loss - current_price) / current_price) * 100, 1
      )
      take_profit_pct = round(
          ((take_profit - current_price) / current_price) * 100, 1
      )

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
          "ATR(14일)": round(atr14),
          "정밀손절가": stop_loss,
          "손절폭(%)": stop_loss_pct,
          "정밀목표가": take_profit,
          "목표폭(%)": take_profit_pct,
      })
    except:
      continue

  if not results:
    return pd.DataFrame(), {}

  res_df = pd.DataFrame(results)

  res_df["유동성점수"] = res_df["당일 거래대금(억)"].rank(pct=True) * 30
  res_df["수급폭증점수"] = res_df["거래대금 폭증배율"].rank(pct=True) * 25
  res_df["모멘텀점수"] = res_df["5일 수익률(%)"].rank(pct=True) * 20
  res_df["추세점수"] = res_df["추세정배열"].apply(lambda x: 15 if "O" in x else 0)

  def calc_rsi_score(rsi):
    if 50 <= rsi <= 70:
      return 10
    elif 30 <= rsi < 50 or 70 < rsi <= 80:
      return 6
    else:
      return 2

  res_df["RSI점수"] = res_df["RSI지표"].apply(calc_rsi_score)
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

  return res_df, raw_price_data


# 4. 메인 UI 구성
market_is_bullish, kospi_val = check_market_macro()

if not market_is_bullish:
  st.warning(
      "⚠️ **[시장 매크로 경고]** 현재 코스피 지수가 20일 이동평균선 아래에"
      " 위치해 있습니다. 매매 비중을 줄이고 손절 라인을 엄격히 준수하세요!"
  )
else:
  st.success(
      "✅ **[시장 매크로 양호]** 코스피 지수가 20일 이동평균선 위에 있어"
      " 안정적인 매매 환경입니다."
  )

if st.button(
    "📊 정밀 ATR 주도주 스크리닝 시작", type="primary", use_container_width=True
):
  with st.spinner(
      "종목별 변동폭(ATR 14일)과 이평선 지지 라인을 기반으로 가격을 계산하는"
      " 중입니다..."
  ):
    df_result, price_dict = analyze_stocks()

    if df_result.empty:
      st.error(
          "데이터 분석 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."
      )
    else:
      st.success(f"✨ 분석 완료! (총 {len(df_result)}개 우량 주도주 정밀 스코어링)")

      # TOP 3 카드 + 주가 차트 및 ATR 정밀 가이드
      st.markdown("### 🏆 오늘의 TOP 3 유망 주도주")
      top3_cols = st.columns(3)

      for i in range(3):
        if i < len(df_result):
          row = df_result.iloc[i]
          stock_name = row["종목명"]
          with top3_cols[i]:
            st.markdown(
                f"""
                <div class="card-box">
                    <h3 style="color: #38bdf8; margin: 0px;">🔥 {int(row['순위'])}위</h3>
                    <h2 style="color: #ffffff; margin: 5px 0px;">{stock_name}</h2>
                    <p style="color: #94a3b8; font-size: 14px; margin: 0px;">종합 점수: <b style="color: #4ade80; font-size: 18px;">{row['종합점수']}점</b></p>
                    <hr style="border-color: #334155; margin: 10px 0px;">
                    <p style="color: #cbd5e1; font-size: 13px; margin: 2px;">현재가: {format(row['현재가'], ',')}원</p>
                    <p style="color: {'#f87171' if row['등락률(%)'] < 0 else '#4ade80'}; font-size: 13px; margin: 2px;">등락률: {row['등락률(%)']}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 차트 보기
            with st.expander(f"📈 {stock_name} 최근 1개월 주가 차트 보기"):
              if stock_name in price_dict:
                df_stock = price_dict[stock_name]
                fig_mini = px.line(
                    df_stock,
                    x=df_stock.index,
                    y="Close",
                    labels={"Close": "종가(원)", "Date": "날짜"},
                )
                fig_mini.update_traces(
                    line_color="#38bdf8",
                    line_width=2.5,
                    hovertemplate="%{x|%m/%d}<br>종가: %{y:,.0f}원<extra></extra>",
                )
                fig_mini.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#cbd5e1", size=10),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="#334155"),
                    height=200,
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_mini, use_container_width=True)

            # 정밀 분석 이유 및 변동성 기반 가격 표시
            with st.expander(f"💡 {stock_name} 분석 및 정밀 가격 라인"):
              st.write(
                  f"**1. 당일 거래대금:** **{row['당일 거래대금(억)']}억** (평소 대비"
                  f" {row['거래대금 폭증배율']}배)"
              )
              st.write(
                  f"**2. 일일 평균변동폭(ATR):** 하루 평균 약"
                  f" **{format(row['ATR(14일)'], ',')}원** 변동"
              )
              st.write(
                  f"**3. 추세 및 과열도:** {row['추세정배열']} / RSI"
                  f" {row['RSI지표']}점"
              )
              st.markdown("---")
              st.warning(
                  f"🛑 **정밀 손절가:** **{format(row['정밀손절가'], ',')}원**"
                  f" ({row['손절폭(%)']}%)"
              )
              st.success(
                  f"🎯 **정밀 목표가:** **{format(row['정밀목표가'], ',')}원**"
                  f" (+{row['목표폭(%)']}%)"
              )

      st.markdown("<br>", unsafe_allow_html=True)

      # TOP 10 차트
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

      # 전체 리스트 표
      st.markdown("### 📋 전체 추천 순위 리스트 및 변동성 기반 가격 가이드")
      display_df = df_result[[
          "순위",
          "종목명",
          "현재가",
          "등락률(%)",
          "ATR(14일)",
          "정밀손절가",
          "손절폭(%)",
          "정밀목표가",
          "목표폭(%)",
          "종합점수",
      ]].copy()

      st.dataframe(
          display_df, use_container_width=True, hide_index=True
      )
else:
  st.info(
      "👆 상단의 **[정밀 ATR 주도주 스크리닝 시작]** 버튼을 누르면 정밀"
      " 가격 라인 분석이 시작됩니다."
  )
