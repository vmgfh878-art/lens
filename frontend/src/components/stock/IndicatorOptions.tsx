// StockView의 보조지표 선택 위젯 (그룹 + 옵션 체크박스). 표현 전용 컴포넌트.

import type { IndicatorDefinition, IndicatorId } from "@/lib/indicators";

interface IndicatorOptionGroupProps {
  title: string;
  options: IndicatorDefinition[];
  selectedIndicators: IndicatorId[];
  onChange: (id: IndicatorId, checked: boolean) => void;
}

export function IndicatorOptionGroup({ title, options, selectedIndicators, onChange }: IndicatorOptionGroupProps) {
  return (
    <div className="indicator-selector__group">
      <span>{title}</span>
      {options.map((option) => (
        <IndicatorOption
          key={option.id}
          option={option}
          checked={selectedIndicators.includes(option.id)}
          onChange={onChange}
        />
      ))}
    </div>
  );
}

interface IndicatorOptionProps {
  option: IndicatorDefinition;
  checked: boolean;
  onChange: (id: IndicatorId, checked: boolean) => void;
}

function IndicatorOption({ option, checked, onChange }: IndicatorOptionProps) {
  return (
    <label className={`indicator-option${checked ? " is-on" : ""}`}>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(option.id, event.target.checked)}
      />
      <span className="indicator-option__dot" aria-hidden="true" />
      <span className="indicator-option__body">
        <span className="indicator-option__label">{option.label}</span>
        <span className="indicator-option__description">{option.description}</span>
      </span>
    </label>
  );
}
