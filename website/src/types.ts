export interface Metric {
  model: string;
  mae: number;
  rmse: number;
  r2: number;
  improvement_pct: number;
  p95_absolute_error: number;
  nmae_test_mean: number;
  n: number;
}
export interface Prediction {
  origin: string;
  target_start: string;
  target_end: string;
  actual: number;
  [key: string]: string | number;
}
export interface Split {
  n: number;
  origin_start: string;
  origin_end: string;
  target_start: string;
  target_end: string;
}
export interface Experiment {
  metadata: {
    selected_model: string;
    selected_ml: string;
    split: Record<string, Split>;
    seed: number;
    executed_utc: string;
    elapsed_seconds: number;
    eligible_origins: number;
    refit_samples: number;
    selection_frozen_utc: string;
    versions: Record<string, string>;
    dataset: { sha256: string };
    thresholds_train_only: { low: number; high: number; sudden_change: number };
    best_parameters: Record<string, Record<string, unknown>>;
    cv: {
      folds: {
        n_train: number;
        n_validation: number;
        train_last_origin: string;
        train_target_end: string;
        validation_first_origin: string;
      }[];
    };
  };
  audit: {
    rows: number;
    columns: number;
    start: string;
    end: string;
    missing_total: number;
    duplicate_timestamps: number;
    irregular_intervals: number;
    negative_energy: number;
    invalid_humidity: number;
    zero_energy_count: number;
    energy_wh: Record<string, number>;
  };
  metrics: {
    validation: Metric[];
    test: Metric[];
    nonoverlapping_test: Metric[];
    bootstrap: {
      improvement_pct: number;
      ci95_pct: number[];
      days: number;
      repetitions: number;
      caveat: string;
    };
  };
  predictions: Prediction[];
  ablation: {
    group: string;
    features: number;
    mae: number;
    rmse: number;
    r2: number;
  }[];
  importance: { feature: string; mae_increase: number; repeat_std: number }[];
  errors: { group: string; mae: number; rmse: number; n: number }[];
  eda: {
    daily: { date: string; kwh: number }[];
    hourly: { hour: number; wh: number }[];
    distribution: { wh: number; count: number }[];
    weekday_weekend: { weekday: number; weekend: number };
    autocorrelation: { lag_minutes: number; correlation: number }[];
    figures: { file: string; title: string; takeaway: string }[];
  };
  worst_origins: string[];
  features: {
    name: string;
    group: string;
    definition: string;
    source: string;
    earliest_offset_minutes: number;
    latest_offset_minutes: number;
  }[];
}
