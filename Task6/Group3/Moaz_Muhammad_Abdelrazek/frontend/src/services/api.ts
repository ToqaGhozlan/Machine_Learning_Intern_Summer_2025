/**
 * API service for communicating with the Taxi Fare Prediction backend
 */

import axios, { AxiosResponse, AxiosError } from 'axios';
import { toast } from 'react-toastify';
import {
  PredictionRequest,
  PredictionResponse,
  ModelInfo,
  HealthStatus,
  ModelCapabilities,
  ApiError,
  API_ENDPOINTS
} from '../types/api';

// Create axios instance with base configuration
const getBaseURL = () => {
  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  // If API URL is just a hostname (from Render), prepend https://
  if (apiUrl && !apiUrl.startsWith('http://') && !apiUrl.startsWith('https://')) {
    return `https://${apiUrl}`;
  }
  return apiUrl;
};

const api = axios.create({
  baseURL: getBaseURL(),
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response: AxiosResponse) => {
    console.log(`API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error: AxiosError<ApiError>) => {
    console.error('API Response Error:', error);
    
    // Handle different error types
    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response;
      const errorMessage = data?.error || data?.message || `HTTP ${status} Error`;
      
      // Show user-friendly error messages
      if (status >= 500) {
        toast.error('Server error. Please try again later.');
      } else if (status === 404) {
        toast.error('Service not found. Please check your connection.');
      } else if (status === 400) {
        toast.error(errorMessage);
      } else {
        toast.error(`Error: ${errorMessage}`);
      }
      
      return Promise.reject(new Error(errorMessage));
    } else if (error.request) {
      // Network error
      toast.error('Network error. Please check your connection.');
      return Promise.reject(new Error('Network error'));
    } else {
      // Other error
      toast.error('An unexpected error occurred.');
      return Promise.reject(error);
    }
  }
);

/**
 * Predict taxi fare for a single trip
 */
export const predictFare = async (request: PredictionRequest): Promise<PredictionResponse> => {
  try {
    const response = await api.post<PredictionResponse>(API_ENDPOINTS.PREDICT, request);
    
    // Show success notification
    if (response.data.status === 'success') {
      toast.success(`Fare predicted: $${response.data.predicted_fare?.toFixed(2)}`);
    } else {
      toast.error(response.data.error_message || 'Prediction failed');
    }
    
    return response.data;
  } catch (error) {
    console.error('Prediction failed:', error);
    throw error;
  }
};

/**
 * Predict taxi fares for multiple trips (batch)
 */
export const predictFareBatch = async (requests: PredictionRequest[]): Promise<PredictionResponse[]> => {
  try {
    const response = await api.post<PredictionResponse[]>(API_ENDPOINTS.PREDICT_BATCH, {
      predictions: requests
    });
    
    const successCount = response.data.filter(r => r.status === 'success').length;
    toast.success(`Batch prediction completed: ${successCount}/${response.data.length} successful`);
    
    return response.data;
  } catch (error) {
    console.error('Batch prediction failed:', error);
    throw error;
  }
};

/**
 * Get model information and metadata
 */
export const getModelInfo = async (): Promise<ModelInfo> => {
  try {
    const response = await api.get<ModelInfo>(API_ENDPOINTS.MODEL_INFO);
    return response.data;
  } catch (error) {
    console.error('Failed to get model info:', error);
    throw error;
  }
};

/**
 * Get model capabilities and constraints
 */
export const getModelCapabilities = async (): Promise<ModelCapabilities> => {
  try {
    const response = await api.get<ModelCapabilities>(API_ENDPOINTS.MODEL_CAPABILITIES);
    return response.data;
  } catch (error) {
    console.error('Failed to get model capabilities:', error);
    throw error;
  }
};

/**
 * Get model performance metrics
 */
export const getModelPerformance = async (): Promise<any> => {
  try {
    const response = await api.get(API_ENDPOINTS.MODEL_PERFORMANCE);
    return response.data;
  } catch (error) {
    console.error('Failed to get model performance:', error);
    throw error;
  }
};

/**
 * Check API health status
 */
export const checkHealth = async (): Promise<HealthStatus> => {
  try {
    const response = await api.get<HealthStatus>(API_ENDPOINTS.HEALTH);
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    throw error;
  }
};

/**
 * Get detailed health status with system metrics
 */
export const getDetailedHealth = async (): Promise<HealthStatus> => {
  try {
    const response = await api.get<HealthStatus>(API_ENDPOINTS.HEALTH_DETAILED);
    return response.data;
  } catch (error) {
    console.error('Detailed health check failed:', error);
    throw error;
  }
};

/**
 * Get API metrics and statistics
 */
export const getMetrics = async (): Promise<any> => {
  try {
    const response = await api.get(API_ENDPOINTS.METRICS);
    return response.data;
  } catch (error) {
    console.error('Failed to get metrics:', error);
    throw error;
  }
};

/**
 * Get prediction example for testing
 */
export const getPredictionExample = async (): Promise<any> => {
  try {
    const response = await api.get('/api/v1/predict/example');
    return response.data;
  } catch (error) {
    console.error('Failed to get prediction example:', error);
    throw error;
  }
};

/**
 * Test API connectivity
 */
export const testConnection = async (): Promise<boolean> => {
  try {
    const response = await api.get('/ping');
    return response.status === 200;
  } catch (error) {
    console.error('Connection test failed:', error);
    return false;
  }
};

/**
 * Utility function to format API errors
 */
export const formatApiError = (error: any): string => {
  if (error.response?.data?.error) {
    return error.response.data.error;
  } else if (error.message) {
    return error.message;
  } else {
    return 'An unexpected error occurred';
  }
};

/**
 * Utility function to validate coordinates
 */
export const validateCoordinates = (lat: number, lng: number): boolean => {
  return lat >= -90 && lat <= 90 && lng >= -180 && lng <= 180;
};

/**
 * Utility function to calculate distance between two points (Haversine formula)
 */
export const calculateDistance = (
  lat1: number,
  lng1: number,
  lat2: number,
  lng2: number
): number => {
  const R = 3956; // Earth's radius in miles
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLng = (lng2 - lng1) * Math.PI / 180;
  const a = 
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
    Math.sin(dLng / 2) * Math.sin(dLng / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
};

/**
 * Utility function to format datetime for API
 */
export const formatDateTimeForApi = (date: Date): string => {
  return date.toISOString().slice(0, 19); // Remove milliseconds and timezone
};

/**
 * Utility function to get current datetime string
 */
export const getCurrentDateTime = (): string => {
  return formatDateTimeForApi(new Date());
};

export default api;
