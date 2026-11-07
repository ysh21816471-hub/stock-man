from datetime import datetime, timedelta
import pandas as pd
from pykrx import stock
import streamlit as st

# 1. 페이지 레이아웃 설정
st.set_page_config(
    page_title="나만의 주식 의사결정 매니저", page_icon="📈", layout="wide"
)

st.title("🚀 나만의 주식 의사결정 매니저 (디버깅 모드)")
st.markdown(
    "데이터 수집 과정에서 발생하는 에러 원인을 정확히 파악하기 위한"
    " 버전입니다."
)


# 2. 스마트 자동 스크리닝 함수 (에러 세부 포착)
@st.cache_data(ttl=3600)
def run_smart_screening():
  today = datetime.today()
  target_date = ""
  df_cap = pd.DataFrame()

  try:
    # 최근 영업일 찾기 (최대 10일 전까지 탐색)
    for i in range(10):
      check_date = (today - timedelta(days=i)).strftime("%Y%m%d")
      try:
        df_cap = stock.get_market_cap_by_ticker(check_date)
        if not df_cap.empty:
          target_date = check_date
          break
      except Exception as sub_e:
        continue

    if df_cap.empty:
      return (
          None,
          "최근 10일간 유효한 거래대금 데이터를 찾지 못했습니다. (휴일/주말"
          " 영향일 수 있음)",
      )

    # 거래대금 기준 상위 150개 종목 확보
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

    # 펀더멘털 데이터 병합 시도
    try:
      df_fund = stock.get_market_fundamental_by_ticker(target_date)
      df_merged = pd.merge(
          df_cap, df_fund, left_on="종목코드", right_index=True, how="left"
      )
    except Exception as e:
      df_merged = df_cap.copy()
      df_merged["PER"] = 0
      df_merged["PBR"] = 0

    # 데이터 정제
    df_merged["PER"] = pd.to_numeric(
        df_merged["PER"], errors="coerce"
    ).fillna(0)
    df_merged["PBR"] = pd.to_numeric(
        df_merged["PBR"], errors="coerce"
    ).fillna(0)

    df_filtered = df_merged[df_merged["거래대금"] >= 1000000000].copy()
    if df_filtered.empty:
      df_filtered = df_merged.copy()

    # 스마트 스코어링 산출
    df_filtered["유동성점수"] = df_filtered["거래대금"].rank(pct=True) * 50

    def calc_per_score(per):
      if per > 0 and per < 50:
        return 50 * (1 - (per / 50))
      elif per >= 50:
        return 15
      else:
        return 25

    df_filtered["재무점수"] = df_filtered["PER"].apply(calc_per_score)
    df_filtered["종합점수"] = (
        df_filtered["유동성점수"] + df_filtered["재무점수"]
    ).round(1)
    df_filtered = df_filtered.sort_values(
        by="종합점수", ascending=False
    ).head(30)

    return df_filtered, target_date

  except Exception as err:
    # 에러 발생 시 정확한 시스템 에러 메시지 반환
    return None, f"CRITICAL_ERROR: {str(err)}"


# 3. 웹 UI 구성
if st.button("🔍 에러 원인 확인 및 스크리닝 시작", type="primary"):
  with st.spinner("데이터를 가져오는 중입니다..."):
    df_result, fetched_date = run_smart_screening()

    # df_result가 DataFrame이 아니거나 비어있다면 fetched_date에 담긴 에러 문구 출력
    if df_result is None or df_result.empty:
      st.error(f"❌ 분석 실패 원인: {fetched_date}")
      st.info(
          "💡 **팁:** 만약 에러 내용에 `Connection` 이나 `HTTP` 관련 내용이"
          " 있다면, 한국거래소(KRX) 서버 측에서 일시적으로 요청을"
          " 차단했거나 주말이라 데이터 조회가 막힌 것일 수 있습니다."
      )
    else:
      formatted_date = (
          f"{fetched_date[:4]}-{fetched_date[4:6]}-{fetched_date[6:]}"
      )
      st.success(f"✨ 분석 성공! (기준일: {formatted_date})")

      display_df = df_result[
          ["종목명", "종가", "등락률", "거래대금", "PER", "PBR", "종합점수"]
      ].copy()
      display_df["거래대금(억)"] = (
          display_df["거래대금"] / 100000000
      ).round(1)
      display_df = display_df.drop(columns=["거래대금"])
      display_df.reset_index(drop=True, inplace=True)

      st.subheader("📋 오늘의 추천 상위 종목 리스트")
      st.dataframe(display_df, use_container_width=True)
else:
  st.info("👈 버튼을 눌러 에러 원인을 확인해 보세요.")
