from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# 1. 페이지 레이아웃 설정
st.set_page_config(
    page_title="나만의 주식 의사결정 매니저", page_icon="📈", layout="wide"
)

st.title("🚀 나만의 주식 의사결정 매니저 (안정형 야후파이낸스 버전)")
st.markdown(
    "한국거래소(KRX) 서버 차단 문제를 완벽히 해결하기 위해, 글로벌 금융"
    " 데이터인 **야후파이낸스(yfinance)**를 기반으로 주요 우량주의 거래대금"
    " 유입과 모멘텀을 실시간 분석합니다."
)

# 2. 국내 주요 주도주(KOSPI & KOSDAQ 대표 종목) 티커 매핑
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


# 3. 데이터 수집 및 분석 함수
@st.cache_data(ttl=3600)
def analyze_stocks():
  results = []
  for name, ticker in WATCH_LIST.items():
    try:
      # 최근 1개월 데이터 다운로드
      df = yf.download(ticker, period="1mo", progress=False)
      if df.empty or len(df) < 10:
        continue

      # MultiIndex 컬럼 대응 (최신 yfinance 버그 방지)
      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

      current_price = float(df["Close"].iloc[-1])
      prev_price = float(df["Close"].iloc[-2])
      change_rate = ((current_price - prev_price) / prev_price) * 100

      # 당일 거래대금 (종가 * 거래량, 억원 단위)
      recent_vol = float(df["Volume"].iloc[-1])
      recent_val = current_price * recent_vol / 100000000

      # 20일 평균 거래대금
      avg_val = (df["Close"] * df["Volume"] / 100000000).mean()

      # 거래대금 폭증 배율 (오늘 거래대금 / 20일 평균)
      vol_spike = (recent_val / avg_val) if avg_val > 0 else 1.0

      # 최근 5일 수익률(모멘텀)
      ret_5d = (
          (
              (current_price - float(df["Close"].iloc[-5]))
              / float(df["Close"].iloc[-5])
          )
          * 100
          if len(df) >= 5
          else 0.0
      )

      results.append({
          "종목명": name,
          "티커": ticker,
          "현재가": round(current_price),
          "등락률(%)": round(change_rate, 2),
          "당일 거래대금(억)": round(recent_val, 1),
          "거래대금 폭증배율": round(vol_spike, 2),
          "5일 수익률(%)": round(ret_5d, 2),
      })
    except Exception as e:
      continue

  if not results:
    return pd.DataFrame()

  res_df = pd.DataFrame(results)

  # [스마트 스코어링 산출]
  res_df["유동성점수"] = res_df["당일 거래대금(억)"].rank(pct=True) * 40
  res_df["수급폭증점수"] = res_df["거래대금 폭증배율"].rank(pct=True) * 30
  res_df["모멘텀점수"] = res_df["5일 수익률(%)"].rank(pct=True) * 30
  res_df["종합점수"] = (
      res_df["유동성점수"] + res_df["수급폭증점수"] + res_df["모멘텀점수"]
  ).round(1)

  res_df = res_df.sort_values(by="종합점수", ascending=False).reset_index(
      drop=True
  )
  return res_df


# 4. 웹 UI 구성
if st.button("🔍 AI 주식 매니저 분석 시작하기", type="primary"):
  with st.spinner(
      "야후파이낸스를 통해 주요 주도주 수급 및 모멘텀을 분석 중입니다..."
  ):
    df_result = analyze_stocks()

    if df_result.empty:
      st.error(
          "데이터를 불러오는 중 문제가 발생했습니다. 잠시 후 다시 시도해"
          " 주세요."
      )
    else:
      st.success(
          f"✨ 분석 완료! (총 {len(df_result)}개 우량 주도주 스크리닝 성공)"
      )

      st.subheader("📋 오늘의 주도주 추천 리스트 (종합 점수 순)")
      st.dataframe(df_result, use_container_width=True)

      st.subheader("📈 상위 종목 종합 점수 비교 (TOP 10)")
      chart_data = df_result.set_index("종목명")["종합점수"].head(10)
      st.bar_chart(chart_data)
else:
  st.info(
      "👈 왼쪽 사이드바의 **[AI 주식 매니저 분석 시작하기]** 버튼을 누르면"
      " 분석이 시작됩니다."
  )
