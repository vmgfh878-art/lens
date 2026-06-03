// 실험 상세의 "비교 지표" 테이블. ComparisonRow 배열을 받아 표로 렌더.

import type { ComparisonRow } from "@/lib/training/comparisonTypes";

export function ComparisonTable({ rows }: { rows: ComparisonRow[] }) {
  if (rows.length === 0) {
    return <div className="compact-note">제품 기준 미확정 상태입니다.</div>;
  }
  return (
    <div className="comparison-table-wrap">
      <table className="comparison-table">
        <thead>
          <tr>
            <th>항목</th>
            <th>제품 모델</th>
            <th>이 실험</th>
            <th>차이</th>
            <th>해석</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id}>
              <td>{row.label}</td>
              <td>{row.productText}</td>
              <td>{row.experimentText}</td>
              <td>{row.diffText}</td>
              <td>{row.interpretation}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
