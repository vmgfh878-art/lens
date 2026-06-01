interface LayerToggleProps {
  label: string;
  checked: boolean;
  disabled?: boolean;
  /** disable 사유 한 줄. disable 상태에서 라벨 아래 인라인 caption 으로 보여줌. CP213. */
  disabledReason?: string;
  onChange: (checked: boolean) => void;
}

export default function LayerToggle({ label, checked, disabled = false, disabledReason, onChange }: LayerToggleProps) {
  return (
    <div className="layer-toggle__wrapper">
      <label className={`layer-toggle${disabled ? " layer-toggle--disabled" : ""}`}>
        <input
          type="checkbox"
          checked={checked}
          disabled={disabled}
          onChange={(event) => onChange(event.target.checked)}
        />
        <span className="layer-toggle__track" aria-hidden="true">
          <span className="layer-toggle__thumb" />
        </span>
        <span>{label}</span>
      </label>
      {disabled && disabledReason ? (
        <small className="layer-toggle__reason" role="note">
          비활성: {disabledReason}
        </small>
      ) : null}
    </div>
  );
}
