import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  LinearProgress,
  Chip,
  Grid,
  Alert,
  Skeleton
} from '@mui/material';
import {
  AttachMoney,
  TrendingUp,
  Schedule,
  CheckCircle,
  Error,
  Info
} from '@mui/icons-material';

import { ResultsDisplayProps } from '../types/api';

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ prediction, loading }) => {
  if (loading) {
    return (
      <Box>
        <Skeleton variant="rectangular" height={120} sx={{ mb: 2 }} />
        <Skeleton variant="text" height={40} />
        <Skeleton variant="text" height={30} />
        <Skeleton variant="text" height={30} />
      </Box>
    );
  }

  if (!prediction) {
    return (
      <Alert severity="info" icon={<Info />}>
        Enter trip details and click "Predict Fare" to see results.
      </Alert>
    );
  }

  const isSuccess = prediction.status === 'success';
  const fare = prediction.predicted_fare;
  const confidence = prediction.confidence;

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return 'success';
    if (confidence >= 60) return 'warning';
    return 'error';
  };

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 80) return 'High Confidence';
    if (confidence >= 60) return 'Medium Confidence';
    return 'Low Confidence';
  };

  return (
    <Box>
      {isSuccess ? (
        <>
          {/* Main Fare Display */}
          <Card 
            elevation={2} 
            sx={{ 
              mb: 2, 
              background: 'linear-gradient(135deg, #1976d2 0%, #42a5f5 100%)',
              color: 'white'
            }}
          >
            <CardContent sx={{ textAlign: 'center', py: 3 }}>
              <AttachMoney sx={{ fontSize: 40, mb: 1 }} />
              <Typography variant="h3" component="div" sx={{ fontWeight: 'bold' }}>
                ${fare?.toFixed(2)}
              </Typography>
              <Typography variant="subtitle1" sx={{ opacity: 0.9 }}>
                Predicted Fare
              </Typography>
            </CardContent>
          </Card>

          {/* Confidence Score */}
          <Card elevation={1} sx={{ mb: 2 }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <TrendingUp sx={{ mr: 1 }} />
                <Typography variant="subtitle1">
                  Confidence Score
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Box sx={{ width: '100%', mr: 1 }}>
                  <LinearProgress
                    variant="determinate"
                    value={confidence}
                    color={getConfidenceColor(confidence) as any}
                    sx={{ height: 8, borderRadius: 4 }}
                  />
                </Box>
                <Typography variant="body2" sx={{ minWidth: 35 }}>
                  {confidence.toFixed(1)}%
                </Typography>
              </Box>
              <Chip
                label={getConfidenceLabel(confidence)}
                color={getConfidenceColor(confidence) as any}
                size="small"
              />
            </CardContent>
          </Card>

          {/* Additional Details */}
          <Grid container spacing={1}>
            <Grid item xs={6}>
              <Card elevation={1}>
                <CardContent sx={{ py: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    Model
                  </Typography>
                  <Typography variant="body1" sx={{ fontWeight: 500 }}>
                    {prediction.model_name}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={6}>
              <Card elevation={1}>
                <CardContent sx={{ py: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    Response Time
                  </Typography>
                  <Typography variant="body1" sx={{ fontWeight: 500 }}>
                    {prediction.prediction_time ? `${prediction.prediction_time}ms` : 'N/A'}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Success Status */}
          <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <CheckCircle color="success" sx={{ mr: 1 }} />
            <Typography variant="body2" color="success.main">
              Prediction successful
            </Typography>
          </Box>
        </>
      ) : (
        /* Error Display */
        <Alert 
          severity="error" 
          icon={<Error />}
          sx={{ mb: 2 }}
        >
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Prediction Failed
          </Typography>
          <Typography variant="body2">
            {prediction.error_message || 'An unknown error occurred'}
          </Typography>
        </Alert>
      )}

      {/* Timestamp */}
      <Box sx={{ mt: 2, textAlign: 'center' }}>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Schedule sx={{ fontSize: 14, mr: 0.5 }} />
          {new Date(prediction.prediction_timestamp).toLocaleString()}
        </Typography>
      </Box>

      {/* Fare Breakdown (if successful) */}
      {isSuccess && fare && (
        <Card elevation={1} sx={{ mt: 2 }}>
          <CardContent>
            <Typography variant="subtitle2" sx={{ mb: 2 }}>
              Fare Breakdown Estimate
            </Typography>
            <Grid container spacing={1}>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  Base Fare
                </Typography>
                <Typography variant="body2">
                  ${(fare * 0.3).toFixed(2)}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  Distance
                </Typography>
                <Typography variant="body2">
                  ${(fare * 0.5).toFixed(2)}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  Time
                </Typography>
                <Typography variant="body2">
                  ${(fare * 0.15).toFixed(2)}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  Other
                </Typography>
                <Typography variant="body2">
                  ${(fare * 0.05).toFixed(2)}
                </Typography>
              </Grid>
            </Grid>
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
              * Estimated breakdown for illustration purposes
            </Typography>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default ResultsDisplay;
