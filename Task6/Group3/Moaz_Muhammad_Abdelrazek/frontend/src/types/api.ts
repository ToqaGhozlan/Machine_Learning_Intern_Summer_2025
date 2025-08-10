/**
 * TypeScript type definitions for the Taxi Fare Prediction API
 */

export interface PredictionRequest {
  pickup_latitude: number;
  pickup_longitude: number;
  dropoff_latitude: number;
  dropoff_longitude: number;
  passenger_count: number;
  pickup_datetime: string;
  weather_condition?: string;
  traffic_condition?: string;
}

export interface PredictionResponse {
  predicted_fare: number | null;
  confidence: number;
  model_name: string;
  model_type?: string;
  prediction_timestamp: string;
  prediction_time?: number;
  status: 'success' | 'error';
  error_message?: string;
  input_validation?: string;
  api_version?: string;
}

export interface ModelInfo {
  model_name?: string;
  model_type?: string;
  version?: string;
  required_features?: number;
  feature_names?: string[];
  has_feature_processor?: boolean;
  deployment_ready?: boolean;
  performance_metrics?: {
    // Old model format (numeric metrics)
    test_r2?: number;
    test_mae?: number;
    test_rmse?: number;
    test_mape?: number;
    accuracy_within_1?: number;
    accuracy_within_2?: number;
    // New model format (string-based metrics)
    accuracy?: string;
    fare_structure?: string;
    distance_calculation?: string;
    confidence_range?: string;
  };
  deployment_info?: {
    api_version?: string;
    model_path?: string;
    load_time?: number | null;
    model_size?: string;
    prediction_count?: number;
    error_count?: number;
    uptime?: string;
  };
}

export interface HealthStatus {
  status: 'healthy' | 'unhealthy' | 'degraded';
  timestamp: string;
  uptime_seconds: number;
  version: string;
  environment: string;
  model_status?: {
    status: string;
    model_loaded: boolean;
    prediction_working: boolean;
    test_prediction?: number;
    last_check: string;
  };
  system_metrics?: {
    cpu_usage_percent: number;
    memory_usage_percent: number;
    memory_available_gb: number;
    memory_total_gb: number;
    disk_usage_percent: number;
    disk_free_gb: number;
    disk_total_gb: number;
  };
  api_metrics?: {
    total_predictions: number;
    successful_predictions: number;
    failed_predictions: number;
    success_rate_percent: number;
    avg_prediction_time_ms: number;
  };
}

export interface ModelCapabilities {
  input_parameters: Record<string, string>;
  output_format: Record<string, string>;
  supported_features: string[];
  constraints: {
    coordinate_bounds: {
      latitude: { min: number; max: number };
      longitude: { min: number; max: number };
    };
    passenger_limits: { min: number; max: number };
    datetime_format: string;
    weather_options: string[];
    traffic_options: string[];
    minimum_trip_distance: string;
    maximum_prediction_time: string;
  };
  performance_characteristics: {
    accuracy_r2: number;
    mean_absolute_error_usd: number;
    rmse_usd: number;
    accuracy_within_2_dollars_percent: number;
    typical_response_time_ms: string;
    model_size_mb: number;
    supported_requests_per_second: string;
  };
}

export interface ApiError {
  error: string;
  status_code: number;
  timestamp: string;
  message?: string;
}

// Form-related types
export interface FormData {
  pickupLatitude: number;
  pickupLongitude: number;
  dropoffLatitude: number;
  dropoffLongitude: number;
  passengerCount: number;
  pickupDateTime: string;
  weatherCondition: string;
  trafficCondition: string;
}

export interface ValidationErrors {
  [key: string]: string;
}

// Map-related types
export interface MapLocation {
  lat: number;
  lng: number;
  address?: string;
}

export interface MapBounds {
  north: number;
  south: number;
  east: number;
  west: number;
}

// Component prop types
export interface PredictionFormProps {
  onSubmit: (request: PredictionRequest) => void;
  loading: boolean;
  pickupLocation: [number, number] | null;
  dropoffLocation: [number, number] | null;
}

export interface ResultsDisplayProps {
  prediction: PredictionResponse | null;
  loading: boolean;
}

export interface MapComponentProps {
  pickupLocation: [number, number] | null;
  dropoffLocation: [number, number] | null;
  onLocationSelect: (type: 'pickup' | 'dropoff', location: [number, number]) => void;
}

export interface ModelInfoProps {
  modelInfo: ModelInfo;
}

// Constants
export const WEATHER_CONDITIONS = [
  { value: 'sunny', label: 'Sunny' },
  { value: 'cloudy', label: 'Cloudy' },
  { value: 'windy', label: 'Windy' },
  { value: 'stormy', label: 'Stormy' }
] as const;

export const TRAFFIC_CONDITIONS = [
  { value: 'flow traffic', label: 'Flow Traffic' },
  { value: 'congested traffic', label: 'Congested Traffic' }
] as const;

export const DEFAULT_MAP_CENTER: [number, number] = [40.7589, -73.9851]; // NYC
export const DEFAULT_MAP_ZOOM = 12;

// API endpoints
export const API_ENDPOINTS = {
  PREDICT: '/api/v1/predict',
  PREDICT_BATCH: '/api/v1/predict/batch',
  MODEL_INFO: '/api/v1/model/info',
  MODEL_CAPABILITIES: '/api/v1/model/capabilities',
  MODEL_PERFORMANCE: '/api/v1/model/performance',
  HEALTH: '/health',
  HEALTH_DETAILED: '/health/detailed',
  METRICS: '/metrics'
} as const;
