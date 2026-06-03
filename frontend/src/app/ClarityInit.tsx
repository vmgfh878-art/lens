"use client";

import Clarity from "@microsoft/clarity";
import { useEffect } from "react";

// Microsoft Clarity (클릭 히트맵 / 세션 리플레이 / UX insights).
// Project ID 는 클라이언트에 노출되는 공개값이라 fallback 으로 박아둔다.
// 환경별로 바꾸려면 NEXT_PUBLIC_CLARITY_ID 환경변수로 override.
const CLARITY_ID = process.env.NEXT_PUBLIC_CLARITY_ID ?? "x11wkud8tf";

export default function ClarityInit() {
  useEffect(() => {
    if (CLARITY_ID) {
      Clarity.init(CLARITY_ID);
    }
  }, []);
  return null;
}
