from datetime import datetime, timedelta
import pandas as pd
from pykrx import stock
import streamlit as st

# 1. 페이지 레이아웃 설정
st.set_page_config(
    page_title="나만의 주식 의사결정 매니저", page_icon="📈", layout="wide"
)

st.title("🚀 나만의 주식 의사결정 매니저 (실시간 스크리닝)")
st.markdown(
    "한국거래소(KRX) 데이터를 바탕으로 **거래대금(유동성)**과 **재무 지표**를"
    " 분석하여 유망 종목을 발굴합니다."
)


# 2. 데이터 수집 및 분석 함수 (예외 처리 강화)
@st.cache_data(ttl=3600)
def load_and_screen_data(min_val_billion, max_per_limit, use_per_filter):
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

  # 거래대금 상위 100개 종목 추출
  df_cap = df_cap.sort_values(by="거래대금", ascending=False).head(100)

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

  # 펀더멘털 데이터 가져오기 시도 (에러 나거나 비어있어도 프로그램이 멈추지 않도록 예외 처리)
  try:
    df_fund = stock.get_market_fundamental_by_ticker(target_date)
    df_merged = pd.merge(
        df_cap, df_fund, left_on="종목코드", right_index=True, how="left"
    )
  except:
    df_merged = df_cap.copy()
    df_merged["PER"] = 0
    df_merged["PBR"] = 0

  # PER 데이터 결측치 처리
  if "PER" not in df_merged.columns:
    df_merged["PER"] = 0
  df_merged["PER"] = pd.to_numeric(df_merged["PER"], errors="coerce").fillna(
      0
  )

  # 3. 필터링 적용
  min_trading_value = min_val_billion * 100000000
  df_filtered = df_merged[df_merged["거래대금"] >= min_trading_value].copy()

  # 사용자가 PER 필터를 켰을 때만 적용
  if use_per_filter:
    df_filtered = df_filtered[
        (df_filtered["PER"] > 0) & (df_filtered["PER"] <= max_per_limit)
    ]

  if df_filtered.empty:
    return df_filtered, target_date

  # 4. 종합 스코어링 산출
  df_filtered["거래대금_점수"] = df_filtered["거래대금"].rank(pct=True) * 50
  if (df_filtered["PER"] > 0).any():
    valid_per = df_filtered["PER"].apply(lambda x: 1 / x if x > 0 else 0)
    df_filtered["PER_점수"] = valid_per.rank(pct=True) * 50
  else:
    df_filtered["PER_점수"] = 25

  df_filtered["종합점수"] = (
      df_filtered["거래대금_점수"] + df_filtered["PER_점수"]
  ).round(1)
  df_filtered = df_filtered.sort_values(by="종합점수", ascending=False)

  return df_filtered, target_date


# 5. 사이드바 인터페이스 설정
st.sidebar.header("🔍 스크리닝 필터 설정")
slider_min_val = st.sidebar.slider(
    "최소 거래대금 (억 원)", min_value=10, max_value=3000, value=100, step=50
)

# PER 필터 강제 적용 여부 체크박스 추가
use_per = st.sidebar.checkbox(
    "재무(PER) 조건 함께 필터링하기",
    value=False,
    help=(
        "체크하면 적자 기업이나 고 PER 종목을 제외합니다. 데이터 누락으로 종목이"
        " 안 나올 때 해제해 보세요."
    ),
)

slider_max_per = st.sidebar.slider(
    "최대 PER 제한", min_value=5, max_value=100, value=35
)

# 6. 실행 버튼
if st.button("📊 실시간 스크리닝 실행하기", type="primary"):
  with st.spinner("KRX 주식 시장 데이터를 분석 중입니다..."):
    df_result, fetched_date = load_and_screen_data(
        slider_min_val, slider_max_per, use_per
    )

    if df_result is None or df_result.empty:
      st.warning(
          "조건에 일치하는 종목이 없습니다. '최소 거래대금'을 낮추거나 '재무(PER)"
          " 필터'를 해제해 보세요!"
      )
    else:
      formatted_date = (
          f"{fetched_date[:4]}-{fetched_date[4:6]}-{fetched_date[6:]}"
      )
      st.success(
          f"✨ 스크리닝 완료! (기준일: {formatted_date} / 총"
          f" {len(df_result)}개 종목)"
      )

      # 출력할 컬럼 정리
      cols_to_show = ["종목명", "종가", "거래대금", "종합점수"]
      if "PER" in df_result.columns:
        cols_to_show.insert(2, "PER")
      if "PBR" in df_result.columns:
        cols_to_show.insert(3, "PBR")

      display_df = df_result[cols_to_show].copy()
      display_df["거래대금(억)"] = (
          display_df["거래대금"] / 100000000
      ).round(1)
      display_df = display_df.drop(columns=["거래대금"])
      display_df.reset_index(drop=True, inplace=True)

      st.subheader("📋 상위 추천 종목 리스트")
      st.dataframe(display_df, use_container_width=True)

      st.subheader("📈 상위 종목 종합 점수 비교 (TOP 10)")
      chart_data = df_result.set_index("종목명")["종합점수"].head(10)
      st.bar_chart(chart_data)
