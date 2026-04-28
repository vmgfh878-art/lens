interface LayerToggleProps {
  label: string;
  checked: boolean;
  disabled?: boolean;
  onChange: (checked: boolean) => void;
}

export default function LayerToggle({ label, checked, disabled = false, onChange }: LayerToggleProps) {
  return (
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
  );
}
