from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# 1. 페이지 레이아웃 및 모바일 반응형 설정
st.set_page_config(
    page_title="나만의 주식 매니저", page_icon="📈", layout="wide"
)

# 커스텀 CSS로 모바일/PC 가독성 및 디자인 세련미 강화
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
        background-color: #1e293b;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #334155;
        text-align: center;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="main-title">🚀 나만의 주식 의사결정 매니저</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-title">실시간 수급·모멘텀 데이터를 종합 분석하여 상승 여력이'
    " 높은 주도주를 발굴합니다.</p>",
    unsafe_allow_html=True,
)


# 2. 국내 주요 주도주 데이터 분석 함수
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
      df = yf.download(ticker, period="1mo", progress=False)
      if df.empty or len(df) < 10:
        continue

      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

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

      results.append({
          "종목명": name,
          "티커": ticker,
          "현재가": round(current_price),
          "등락률(%)": round(change_rate, 2),
          "당일 거래대금(억)": round(recent_val, 1),
          "거래대금 폭증배율": round(vol_spike, 2),
          "5일 수익률(%)": round(ret_5d, 2),
      })
    except:
      continue

  if not results:
    return pd.DataFrame()

  res_df = pd.DataFrame(results)

  # 점수 산출
  res_df["유동성점수"] = res_df["당일 거래대금(억)"].rank(pct=True) * 40
  res_df["수급폭증점수"] = res_df["거래대금 폭증배율"].rank(pct=True) * 30
  res_df["모멘텀점수"] = res_df["5일 수익률(%)"].rank(pct=True) * 30
  res_df["종합점수"] = (
      res_df["유동성점수"] + res_df["수급폭증점수"] + res_df["모멘텀점수"]
  ).round(1)

  # 종합 점수 기준 내림차순 정렬 (1위가 맨 위로 오도록 확실하게 고정)
  res_df = res_df.sort_values(by="종합점수", ascending=False).reset_index(
      drop=True
  )
  res_df["순위"] = res_df.index + 1  # 1위부터 순위 부여

  return res_df


# 3. 메인 인터페이스 및 분석 실행 버튼
if st.button("📊 실시간 주도주 스크리닝 시작", type="primary", use_container_width=True):
  with st.spinner(
      "시장 데이터를 수집하고 최적의 종목 순위를 매기는 중입니다..."
  ):
    df_result = analyze_stocks()

    if df_result.empty:
      st.error(
          "데이터 분석 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."
      )
    else:
      st.success(
          f"✨ 분석 완료! (총 {len(df_result)}개 종목 추천 순위 산정 완료)"
      )

      # --- [UI 개선 1] 상위 TOP 3 하이라이트 카드 (모바일/PC 모두 예쁘게 배치) ---
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
                    <p style="color: #94a3b8; font-size: 14px; margin: 0px;">종합 점수: <b style="color: #4ade80;">{row['종합점수']}점</b></p>
                    <hr style="border-color: #334155; margin: 10px 0px;">
                    <p style="color: #cbd5e1; font-size: 13px; margin: 2px;">현재가: {format(row['현재가'], ',')}원</p>
                    <p style="color: {'#f87171' if row['등락률(%)'] < 0 else '#4ade80'}; font-size: 13px; margin: 2px;">등락률: {row['등락률(%)']}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

      st.markdown("<br>", unsafe_allow_html=True)

      # --- [UI 개선 2] 전체 추천 순위 표 (반응형 및 깔끔한 컬럼 정렬) ---
      st.markdown("### 📋 전체 추천 순위 리스트")
      display_df = df_result[[
          "순위",
          "종목명",
          "현재가",
          "등락률(%)",
          "당일 거래대금(억)",
          "거래대금 폭증배율",
          "5일 수익률(%)",
          "종합점수",
      ]].copy()

      # Streamlit 반응형 테이블 출력 (모바일 가로 스크롤 및 PC 최적화 지원)
      st.dataframe(
          display_df, use_container_width=True, hide_index=True
      )

      # --- [UI 개선 3] TOP 10 점수 비교 바 차트 ---
      st.markdown("### 📊 상위 종목 종합 점수 비교")
      chart_data = df_result.set_index("종목명")["종합점수"].head(10)
      st.bar_chart(chart_data)
else:
  st.info(
      "👆 상단의 **[실시간 주도주 스크리닝 시작]** 버튼을 누르면 AI 매니저가"
      " 순위를 매겨 분석 결과를 보여줍니다."
  )
