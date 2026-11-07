from datetime import datetime, timedelta
import pandas as pd
from pykrx import stock
import streamlit as st

# 1. 페이지 레이아웃 설정
st.set_page_config(
    page_title="나만의 주식 의사결정 매니저", page_icon="📈", layout="wide"
)

st.title("🚀 나만의 주식 의사결정 매니저 (스마트 자동 스크리닝)")
st.markdown(
    "복잡한 필터 조절 없이, **거래대금(유동성)**과 **재무 건전성(흑자)"
    " 지표**를 프로그램이 알아서 분석하여 가장 상승 여력이 높은 종목을 자동으로"
    " 발굴합니다."
)


# 2. 스마트 자동 스크리닝 함수 (0건 방지 및 유연한 채점 로직)
@st.cache_data(ttl=3600)
def run_smart_screening():
  today = datetime.today()
  target_date = ""
  df_cap = pd.DataFrame()

  # 최근 영업일 찾기 (최대 10일 전까지 탐색)
  for i in range(10):
    check_date = (today - timedelta(days=i)).strftime("%Y%m%d")
    try:
      df_cap = stock.get_market_cap_by_ticker(check_date)
      if not df_cap.empty:
        target_date = check_date
        break
    except:
      continue

  if df_cap.empty:
    return None, "데이터를 불러올 수 없습니다."

  # 거래대금 기준 상위 150개 종목을 넉넉하게 확보
  df_cap = df_cap.sort_values(by="거래대금", ascending=False).head(150)

  # 종목명 매핑
  names = [stock.get_market_ticker_name(code) for code in df_cap.index]
  df_cap["종목명"] = names
  df_cap = df_cap.reset_index()

  if "티커" in df_cap.columns:
    df_cap.rename(columns={"티커": "종목코드"}, inplace=True)
  elif "index" in df_cap.columns:
    df_cap.rename(columns={"index": "종목코드"}, inplace=True)
  else:
    df_cap["종목코드"] = df_cap.index

  # 펀더멘털(PER, PBR 등) 데이터 병합 시도
  try:
    df_fund = stock.get_market_fundamental_by_ticker(target_date)
    df_merged = pd.merge(
        df_cap, df_fund, left_on="종목코드", right_index=True, how="left"
    )
  except:
    df_merged = df_cap.copy()
    df_merged["PER"] = 0
    df_merged["PBR"] = 0

  # 데이터 정제 (결측치 처리)
  df_merged["PER"] = pd.to_numeric(
      df_merged["PER"], errors="coerce"
  ).fillna(0)
  df_merged["PBR"] = pd.to_numeric(
      df_merged["PBR"], errors="coerce"
  ).fillna(0)

  # 최소한의 유동성(거래대금 10억 이상)만 가볍게 잡고 모두 포용
  df_filtered = df_merged[df_merged["거래대금"] >= 1000000000].copy()

  if df_filtered.empty:
    df_filtered = df_merged.copy()

  # [스마트 스코어링 엔진] (강제 탈락 방식 대신 점수 차등화 방식 적용)
  # 1. 유동성 점수 (거래대금이 많을수록 고득점, 50점 만점)
  df_filtered["유동성점수"] = df_filtered["거래대금"].rank(pct=True) * 50

  # 2. 재무/가치 점수 (PER이 합리적일수록 우대, 50점 만점)
  def calc_per_score(per):
    if per > 0 and per < 50:
      return 50 * (1 - (per / 50))  # PER이 낮을수록 높은 가점
    elif per >= 50:
      return 15  # 고PER 종목
    else:
      return 25  # 데이터 누락 또는 적자 기업 (기본 방어 점수)

  df_filtered["재무점수"] = df_filtered["PER"].apply(calc_per_score)

  # 3. 최종 종합 점수 산출
  df_filtered["종합점수"] = (
      df_filtered["유동성점수"] + df_filtered["재무점수"]
  ).round(1)

  # 점수 높은 순으로 정렬 후 상위 30개 추출
  df_filtered = df_filtered.sort_values(
      by="종합점수", ascending=False
  ).head(30)

  return df_filtered, target_date


# 3. 웹 UI 구성
st.sidebar.info(
    "💡 **스마트 매니저 모드**\n\n수동으로 필터를 조절할 필요 없이, 프로그램이"
    " 알아서 시장의 거래대금과 재무 상태를 종합 채점하여 유망 종목을"
    " 발굴합니다."
)

if st.button("🔍 AI 주식 매니저 스크리닝 시작하기", type="primary"):
  with st.spinner(
      "한국거래소 데이터를 실시간 분석하여 유망 종목을 추리는 중입니다..."
  ):
    df_result, fetched_date = run_smart_screening()

    if df_result is None or df_result.empty:
      st.error(
          "데이터 분석 중 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해"
          " 주세요."
      )
    else:
      formatted_date = (
          f"{fetched_date[:4]}-{fetched_date[4:6]}-{fetched_date[6:]}"
      )
      st.success(
          f"✨ 분석 완료! (기준일: {formatted_date} / 상위 유망 종목"
          f" {len(df_result)}개 발굴)"
      )

      # 보기 좋은 형태로 데이터 정제
      display_df = df_result[
          ["종목명", "종가", "등락률", "거래대금", "PER", "PBR", "종합점수"]
      ].copy()
      display_df["거래대금(억)"] = (
          display_df["거래대금"] / 100000000
      ).round(1)
      display_df = display_df.drop(columns=["거래대금"])
      display_df.reset_index(drop=True, inplace=True)

      # 표 출력
      st.subheader("📋 오늘의 추천 상위 종목 리스트")
      st.dataframe(display_df, use_container_width=True)

      # 차트 시각화
      st.subheader("📈 상위 10개 종목 종합 점수 비교")
      chart_data = df_result.set_index("종목명")["종합점수"].head(10)
      st.bar_chart(chart_data)
else:
  st.info(
      "👈 왼쪽 사이드바의 **[AI 주식 매니저 스크리닝 시작하기]** 버튼을"
      " 누르면 자동으로 분석이 시작됩니다."
  )
