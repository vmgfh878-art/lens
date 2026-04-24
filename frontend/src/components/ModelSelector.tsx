"use client";

import { ChangeEvent } from "react";

interface ModelSelectorProps {
  ticker: string;
  onTickerChange: (ticker: string) => void;
  model: string;
  onModelChange: (model: string) => void;
}

const MODELS = ["patchtst", "cnn_lstm", "tide"];

export default function ModelSelector({
  ticker,
  onTickerChange,
  model,
  onModelChange,
}: ModelSelectorProps) {
  return (
    <div className="flex gap-4 items-center p-4 bg-slate-800 rounded-lg">
      <input
        type="text"
        value={ticker}
        onChange={(e: ChangeEvent<HTMLInputElement>) => onTickerChange(e.target.value.toUpperCase())}
        placeholder="티커 입력 (예: AAPL)"
        className="px-3 py-2 rounded bg-slate-700 text-white placeholder-slate-400 w-40"
      />

      <div className="flex gap-2">
        {MODELS.map((m) => (
          <button
            key={m}
            onClick={() => onModelChange(m)}
            className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
              model === m
                ? "bg-blue-600 text-white"
                : "bg-slate-700 text-slate-300 hover:bg-slate-600"
            }`}
          >
            {m}
          </button>
        ))}
      </div>
    </div>
  );
}
