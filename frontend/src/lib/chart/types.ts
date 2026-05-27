export interface OverlayPoint {
  time: string;
  value: number;
}

export interface VolumePoint extends OverlayPoint {
  color: string;
}

export interface OverlayState {
  canDrawBand: boolean;
  canDrawConservativeLine: boolean;
  conservativeData: OverlayPoint[];
  conservativeHistoryData: OverlayPoint[];
  upperBandData: OverlayPoint[];
  lowerBandData: OverlayPoint[];
  upperBandHistoryData: OverlayPoint[];
  lowerBandHistoryData: OverlayPoint[];
  modelMarkerDate: string | null;
  warning: string | null;
}

export const PREDICTION_SCALE_ID = "prediction-overlay";
export const MAX_ROLLING_HISTORY_GAP_DAYS = 10;
