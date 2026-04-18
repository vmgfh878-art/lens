"""
train.py - 모델 학습 스크립트
사용법: python train.py --model patchtst --epochs 50
"""

import argparse
import torch
from ai.models.patchtst import PatchTST
from ai.models.cnn_lstm import CNNLSTM
from ai.models.tide import TiDE
from ai.loss import AsymmetricBCELoss

MODEL_REGISTRY = {
    "patchtst": PatchTST,
    "cnn_lstm": CNNLSTM,
    "tide": TiDE,
}


def train(model_name: str, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls().to(device)
    criterion = AsymmetricBCELoss(alpha=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    # TODO: DataLoader 연결 (data/ 디렉터리의 preprocessing.py 사용)
    print(f"[{model_name}] 학습 준비 완료. DataLoader 연결 필요.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), default="patchtst")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    train(args.model, args.epochs)
