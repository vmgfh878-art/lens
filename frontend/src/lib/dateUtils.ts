import type { PriceBar } from "@/api/client";

export function isValidDate(value: unknown): value is string {
  return typeof value === "string" && value.length > 0 && !Number.isNaN(Date.parse(value));
}

export function uniqueDates(dates: string[]) {
  return Array.from(new Set(dates.filter(isValidDate))).sort((left, right) => left.localeCompare(right));
}

export function sortUniqueByDate<T extends { date: string }>(rows: T[]) {
  const deduped = new Map<string, T>();
  rows.forEach((row) => {
    if (isValidDate(row.date)) {
      deduped.set(row.date, row);
    }
  });
  return Array.from(deduped.values()).sort((left, right) => left.date.localeCompare(right.date));
}

export function sortUniqueByAsofDate<T extends { asof_date: string }>(rows: T[]) {
  const deduped = new Map<string, T>();
  rows.forEach((row) => {
    if (isValidDate(row.asof_date)) {
      deduped.set(row.asof_date, row);
    }
  });
  return Array.from(deduped.values()).sort((left, right) =>
    left.asof_date.localeCompare(right.asof_date)
  );
}

export function sortPriceRows(rows: PriceBar[]) {
  return sortUniqueByDate(rows.filter((row) => Number.isFinite(row.close)));
}

export function formatDate(value: Date) {
  return value.toISOString().slice(0, 10);
}

export function buildDefaultPriceWindow(lookbackDays: number) {
  const end = new Date();
  const start = new Date(end);
  start.setDate(start.getDate() - lookbackDays);
  return {
    start: formatDate(start),
    end: formatDate(end),
  };
}

export function buildFullPriceWindows(startYear: number) {
  const currentYear = new Date().getFullYear();
  return Array.from({ length: currentYear - startYear + 1 }, (_, index) => {
    const year = startYear + index;
    return {
      start: `${year}-01-01`,
      end: `${year}-12-31`,
    };
  });
}

export function addDays(date: string, days: number) {
  const parsed = new Date(`${date}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) {
    return date;
  }
  parsed.setUTCDate(parsed.getUTCDate() + days);
  return parsed.toISOString().slice(0, 10);
}

export function addBusinessDays(date: string, days: number) {
  const parsed = new Date(`${date}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) {
    return date;
  }

  let remaining = days;
  while (remaining > 0) {
    parsed.setUTCDate(parsed.getUTCDate() + 1);
    const day = parsed.getUTCDay();
    if (day !== 0 && day !== 6) {
      remaining -= 1;
    }
  }
  return parsed.toISOString().slice(0, 10);
}

export function diffCalendarDaysBetween(
  fromDate: string | null | undefined,
  toDate: string | null | undefined
) {
  if (!isValidDate(fromDate) || !isValidDate(toDate)) {
    return null;
  }
  const from = new Date(`${fromDate}T00:00:00Z`).getTime();
  const to = new Date(`${toDate}T00:00:00Z`).getTime();
  if (!Number.isFinite(from) || !Number.isFinite(to)) {
    return null;
  }
  return Math.max(0, Math.floor((to - from) / (24 * 60 * 60 * 1000)));
}

// CP215 — 두 날짜 사이의 평일(월~금) 수. fromDate < toDate 면 양수, 같거나 역순이면 0.
// 휴장/공휴일 무관 — 평일 기준 근사. 가격 datasets 가 없어도 동작.
export function diffBusinessDaysBetween(
  fromDate: string | null | undefined,
  toDate: string | null | undefined
) {
  if (!isValidDate(fromDate) || !isValidDate(toDate)) {
    return null;
  }
  if (fromDate >= toDate) {
    return 0;
  }
  const cursor = new Date(`${fromDate}T00:00:00Z`);
  const end = new Date(`${toDate}T00:00:00Z`);
  if (Number.isNaN(cursor.getTime()) || Number.isNaN(end.getTime())) {
    return null;
  }
  let count = 0;
  while (cursor < end) {
    cursor.setUTCDate(cursor.getUTCDate() + 1);
    const day = cursor.getUTCDay();
    if (day !== 0 && day !== 6) {
      count += 1;
    }
  }
  return count;
}

export function countPriceRowsAfter(
  rows: PriceBar[],
  fromDate: string | null | undefined,
  toDate: string | null | undefined
) {
  if (!isValidDate(fromDate) || !isValidDate(toDate)) {
    return null;
  }
  return sortPriceRows(rows).filter((row) => row.date > fromDate && row.date <= toDate).length;
}
