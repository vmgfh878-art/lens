"""
strategy_svc.py
- 모델 예측 확률을 받아서 BUY / SELL / HOLD 시그널 생성
- Rule-based 조건 적용 (예: prob_up > 0.65 → BUY)
"""


def generate_signal(prob_up: float, threshold_buy: float = 0.65, threshold_sell: float = 0.35) -> str:
    """
    상승 확률(prob_up)을 기반으로 매매 시그널 반환
    """
    if prob_up >= threshold_buy:
        return "BUY"
    elif prob_up <= threshold_sell:
        return "SELL"
    else:
        return "HOLD"
