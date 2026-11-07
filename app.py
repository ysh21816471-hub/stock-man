from datetime import datetime, timedelta
import pandas as pd
from pykrx import stock
import streamlit as strlit
import streamlit as st

# 1. 페이지 레이아웃 설정
st.set_page_config(
    page_title="나만의 주식 의사결정 매니저", page_icon="📈", layout="wide"
)

st.title("🚀 나만의 주식 의사결정 매니저 (실시간 스크리닝)")
st.markdown(
    "한국거래소(KRX) 데이터를 바탕으로 **거래대금(유동성)**과 **재무 지표(PER)**를"
    " 종합하여 상승 여지가 높은 종목을 발굴합니다."
)


# 2. 데이터 수집 및 분석 함수 (캐싱 처리로 속도 최적화)
@st.cache_data(ttl=3600)
def load_and_screen_data(min_val_billion, max_per_limit):
  # 주말이나 공휴일일 경우 최근 영업일 찾기
  today = datetime.today()
  target_date = ""
  df_cap = pd.DataFrame()

  for i in range(7):
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

  # 거래대금 상위 100개 종목 추출
  df_cap = df_cap.sort_values(by="거래대금", ascending=False).head(100)

  # 펀더멘털 데이터(PER, PBR 등) 가져오기
  df_fund = stock.get_market_fundamental_by_ticker(target_date)

  # 데이터 병합
  df_merged = pd.merge(df_cap, df_fund, left_index=True, right_index=True)
  df_merged = df_merged.reset_index()
  df_merged.rename(columns={"티커": "종목코드"}, inplace=True)

  # 종목명 매핑
  names = [stock.get_market_ticker_name(code) for code in df_merged["종목코드"]]
  df_merged["종목명"] = names

  # 3. 필터링 로직 적용
  min_trading_value = min_val_billion * 100000000   조 단위/억 단위 변환
  df_filtered = df_merged[
      (df_merged["거래대금"] >= min_trading_value)
      & (df_merged["PER"] > 0)
      & (df_merged["PER"] <= max_per_limit)
  ].copy()

  if df_filtered.empty:
    return df_filtered, target_date

  # 4. 나만의 스코어링 엔진 (종합 점수 산출)
  # - 거래대금이 많을수록 가점 (50점 만점)
  # - PER이 낮을수록(저평가) 가점 (50점 만점)
  df_filtered["거래대금_점수"] = df_filtered["거래대금"].rank(pct=True) * 50
  df_filtered["PER_점수"] = (1 / df_filtered["PER"]).rank(pct=True) * 50
  df_filtered["종합점수"] = (
      df_filtered["거래대금_점수"] + df_filtered["PER_점수"]
  ).round(1)

  # 점수 기준 정렬
  df_filtered = df_filtered.sort_values(by="종합점수", ascending=False)

  return df_filtered, target_date


# 5. 사이드바 인터페이스 (사용자 설정)
st.sidebar.header("🔍 스크리닝 필터 설정")
slider_min_val = st.sidebar.slider(
    "최소 거래대금 (억 원)", min_value=100, max_value=5000, value=300, step=100
)
slider_max_per = st.sidebar.slider(
    "최대 PER (적자 및 고평가 제외)", min_value=5, max_value=100, value=35
)

# 6. 실행 버튼
if st.button("📊 실시간 스크리닝 실행하기", type="primary"):
  with st.spinner("KRX 주식 시장 데이터를 분석 중입니다... 잠시만 기다려주세요!"):
    df_result, fetched_date = load_and_screen_data(
        slider_min_val, slider_max_per
    )

    if df_result is None or df_result.empty:
      st.warning(
          "조건에 일치하는 종목이 없습니다. 필터 값을 조정해 보세요!"
      )
    else:
      formatted_date = f"{fetched_date[:4]}-{fetched_date[4:6]}-{fetched_date[6:]}"
      st.success(
          f"✨ 스크리닝 완료! (기준일: {formatted_date} / 총"
          f" {len(df_result)}개 종목 발굴)"
      )

      # 보기 좋은 형태로 컬럼 가공
      display_df = df_result[
          ["종목명", "종가", "등락률", "거래대금", "PER", "PBR", "종합점수"]
      ].copy()
      display_df["거래대금(억)"] = (
          display_df["거래대금"] / 100000000
      ).round(1)
      display_df = display_df.drop(columns=["거래대금"])
      display_df.reset_index(drop=True, inplace=True)

      # 화면 출력 (표)
      st.subheader("📋 상위 추천 종목 리스트")
      st.dataframe(display_df, use_container_width=True)

      # 시각화 차트
      st.subheader("📈 상위 종목 종합 점수 비교 (TOP 10)")
      chart_data = df_result.set_index("종목명")["종합점수"].head(10)
      st.bar_chart(chart_data)