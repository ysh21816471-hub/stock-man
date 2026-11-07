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

# 모바일/PC 반응형 커스텀 스타일
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
    '<p class="main-title">🚀 나만의 주식 의사결정 매니저</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-title">실시간 수급·모멘텀 데이터를 종합 분석하여 상승 여력이'
    " 높은 주도주를 발굴합니다.</p>",
    unsafe_allow_html=True,
)


# 2. 주도주 수집 및 분석 함수
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

  # 내림차순 정렬 및 순위 추가
  res_df = res_df.sort_values(by="종합점수", ascending=False).reset_index(
      drop=True
  )
  res_df["순위"] = res_df.index + 1

  return res_df


# 3. 메인 화면 구성 및 실행
if st.button(
    "📊 실시간 주도주 스크리닝 시작", type="primary", use_container_width=True
):
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

      # --- [개선 1] TOP 3 카드 + 선정 이유 모달(Expander) ---
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

            # 클릭 시 이유를 보여주는 접이식 버튼
            with st.expander(f"💡 {row['종목명']} 선정 이유 보기"):
              st.write(f"**1. 유동성 수급:** {row['당일 거래대금(억)']}억 유입")
              st.write(
                  f"**2. 수급 폭증률:** 평소 20일 평균 대비"
                  f" **{row['거래대금 폭증배율']}배** 폭증"
              )
              st.write(
                  f"**3. 단기 모멘텀:** 최근 5일간 **{row['5일 수익률(%)']}%**"
                  " 추세 유지"
              )

              # 종합 평가 문구 자동 생성
              if row["거래대금 폭증배율"] >= 1.5:
                st.info(
                    "🔥 **알고리즘 총평:** 평소보다 거래량이 급증하며 강한"
                    " 자금이 유입된 전형적인 주도주 형태입니다."
                )
              elif row["5일 수익률(%)"] > 5:
                st.info(
                    "📈 **알고리즘 총평:** 5일 단기 추세가 우상향하며 탄탄한"
                    " 상승 모멘텀을 형성하고 있습니다."
                )
              else:
                st.info(
                    "🛡️ **알고리즘 총평:** 거래대금 수급이 상위권에 속해 있어"
                    " 하방을 지지하는 힘이 강합니다."
                )

      st.markdown("<br>", unsafe_allow_html=True)

      # --- [개선 2] Plotly 기반의 고급 세련된 차트 UI ---
      st.markdown("### 📊 TOP 10 종목 종합 점수 비교")
      top10_df = df_result.head(10).sort_values(by="종합점수", ascending=True)

      fig = px.bar(
          top10_df,
          x="종합점수",
          y="종목명",
          orientation="h",
          text="종합점수",
          color="종합점수",
          color_continuous_scale="Viridis",  # 세련된 그라데이션 색상
      )

      fig.update_layout(
          plot_bgcolor="rgba(0,0,0,0)",
          paper_bgcolor="rgba(0,0,0,0)",
          font=dict(color="#ffffff"),
          xaxis=dict(showgrid=True, gridcolor="#334155", title="종합 점수"),
          yaxis=dict(title=""),
          coloraxis_showscale=False,  # 범례 숨기기
          height=380,
          margin=dict(l=20, r=20, t=20, b=20),
      )
      fig.update_traces(
          texttemplate="%{text}점",
          textposition="outside",
          hovertemplate="<b>%{y}</b><br>종합점수: %{x}점<extra></extra>",
      )

      st.plotly_chart(fig, use_container_width=True)

      # --- 전체 추천 순위 리스트 표 ---
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

      st.dataframe(
          display_df, use_container_width=True, hide_index=True
      )
else:
  st.info(
      "👆 상단의 **[실시간 주도주 스크리닝 시작]** 버튼을 누르면 AI 매니저가"
      " 순위를 매겨 분석 결과를 보여줍니다."
  )
