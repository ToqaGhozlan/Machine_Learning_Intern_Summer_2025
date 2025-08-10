import React, { useState } from 'react';
import {
  Box,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Grid,
  LinearProgress,
  Card,
  CardContent,
  Divider
} from '@mui/material';
import {
  ExpandMore,
  Psychology,
  Speed,
  Memory,
  Schedule,
  TrendingUp
} from '@mui/icons-material';

import { ModelInfoProps } from '../types/api';

const ModelInfo: React.FC<ModelInfoProps> = ({ modelInfo }) => {
  const [expanded, setExpanded] = useState<string | false>(false);

  const handleChange = (panel: string) => (event: React.SyntheticEvent, isExpanded: boolean) => {
    setExpanded(isExpanded ? panel : false);
  };

  const formatUptime = (uptime: string) => {
    return uptime || 'N/A';
  };



  return (
    <Box>
      {/* Model Overview */}
      <Card elevation={1} sx={{ mb: 2 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <Psychology sx={{ mr: 1, color: 'primary.main' }} />
            <Typography variant="h6">
              {modelInfo?.model_name || 'Unknown Model'}
            </Typography>
          </Box>
          
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Typography variant="body2" color="text.secondary">
                Type
              </Typography>
              <Typography variant="body1">
                {modelInfo?.model_type || 'Unknown'}
              </Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="body2" color="text.secondary">
                Version
              </Typography>
              <Typography variant="body1">
                {modelInfo?.version || 'N/A'}
              </Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="body2" color="text.secondary">
                Features
              </Typography>
              <Typography variant="body1">
                {modelInfo.required_features || 'N/A'}
              </Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="body2" color="text.secondary">
                Status
              </Typography>
              <Chip
                label={modelInfo.deployment_ready ? 'Production Ready' : 'Development'}
                color={modelInfo.deployment_ready ? 'success' : 'warning'}
                size="small"
              />
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Performance Metrics */}
      <Accordion expanded={expanded === 'performance'} onChange={handleChange('performance')}>
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <TrendingUp sx={{ mr: 1 }} />
            <Typography>Performance Metrics</Typography>
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            {/* Check if we have numeric performance metrics (old model) or string metrics (new model) */}
            {typeof modelInfo.performance_metrics?.test_r2 === 'number' ? (
              <>
                <Grid item xs={12} sm={6}>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      R² Score (Accuracy)
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <Box sx={{ width: '100%', mr: 1 }}>
                        <LinearProgress
                          variant="determinate"
                          value={modelInfo.performance_metrics.test_r2 * 100}
                          color="success"
                          sx={{ height: 8, borderRadius: 4 }}
                        />
                      </Box>
                      <Typography variant="body2" sx={{ minWidth: 50 }}>
                        {(modelInfo.performance_metrics.test_r2 * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                  </Box>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Mean Absolute Error
                    </Typography>
                    <Typography variant="h6" color="success.main">
                      ${modelInfo.performance_metrics.test_mae?.toFixed(3) || 'N/A'}
                    </Typography>
                  </Box>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      RMSE
                    </Typography>
                    <Typography variant="h6">
                      ${modelInfo.performance_metrics.test_rmse?.toFixed(3) || 'N/A'}
                    </Typography>
                  </Box>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Accuracy within $2
                    </Typography>
                    <Typography variant="h6" color="success.main">
                      {modelInfo.performance_metrics.accuracy_within_2?.toFixed(1) || 'N/A'}%
                    </Typography>
                  </Box>
                </Grid>
              </>
            ) : (
              /* New model with string-based performance metrics */
              <>
                <Grid item xs={12} sm={6}>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Accuracy
                    </Typography>
                    <Typography variant="h6" color="success.main">
                      {modelInfo.performance_metrics?.accuracy || 'N/A'}
                    </Typography>
                  </Box>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Fare Structure
                    </Typography>
                    <Typography variant="body1">
                      {modelInfo.performance_metrics?.fare_structure || 'N/A'}
                    </Typography>
                  </Box>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Distance Calculation
                    </Typography>
                    <Typography variant="body1">
                      {modelInfo.performance_metrics?.distance_calculation || 'N/A'}
                    </Typography>
                  </Box>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Confidence Range
                    </Typography>
                    <Typography variant="h6" color="success.main">
                      {modelInfo.performance_metrics?.confidence_range || 'N/A'}
                    </Typography>
                  </Box>
                </Grid>
              </>
            )}
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Deployment Info */}
      <Accordion expanded={expanded === 'deployment'} onChange={handleChange('deployment')}>
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Speed sx={{ mr: 1 }} />
            <Typography>Deployment Information</Typography>
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <Typography variant="body2" color="text.secondary">
                Model Size
              </Typography>
              <Typography variant="body1">
                {modelInfo.deployment_info?.model_size || 'N/A'}
              </Typography>
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <Typography variant="body2" color="text.secondary">
                Load Time
              </Typography>
              <Typography variant="body1">
                {modelInfo.deployment_info?.load_time ?
                  `${modelInfo.deployment_info.load_time.toFixed(2)}s` :
                  'Instant (Rule-based)'
                }
              </Typography>
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <Typography variant="body2" color="text.secondary">
                Predictions Served
              </Typography>
              <Typography variant="body1">
                {modelInfo.deployment_info?.prediction_count?.toLocaleString() || '0'}
              </Typography>
            </Grid>

            <Grid item xs={12} sm={6}>
              <Typography variant="body2" color="text.secondary">
                Uptime
              </Typography>
              <Typography variant="body1">
                {formatUptime(modelInfo.deployment_info?.uptime || '')}
              </Typography>
            </Grid>

            <Grid item xs={12}>
              <Typography variant="body2" color="text.secondary">
                API Version
              </Typography>
              <Typography variant="body1">
                {modelInfo.deployment_info?.api_version || 'N/A'}
              </Typography>
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Features */}
      <Accordion expanded={expanded === 'features'} onChange={handleChange('features')}>
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Memory sx={{ mr: 1 }} />
            <Typography>Model Features ({modelInfo.required_features || 0})</Typography>
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Input Features Used by the Model
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
              {(modelInfo.feature_names || []).map((feature, index) => (
                <Chip
                  key={index}
                  label={feature.replace(/_/g, ' ').replace(/scaled/g, '').trim()}
                  size="small"
                  variant="outlined"
                  sx={{ fontSize: '0.75rem' }}
                />
              ))}
            </Box>
          </Box>
          
          <Divider sx={{ my: 2 }} />
          
          <Typography variant="body2" color="text.secondary">
            Feature Categories:
          </Typography>
          <Box sx={{ mt: 1 }}>
            <Chip label="Location Features" color="primary" size="small" sx={{ mr: 1, mb: 1 }} />
            <Chip label="Temporal Features" color="secondary" size="small" sx={{ mr: 1, mb: 1 }} />
            <Chip label="Trip Features" color="success" size="small" sx={{ mr: 1, mb: 1 }} />
            <Chip label="External Conditions" color="warning" size="small" sx={{ mr: 1, mb: 1 }} />
          </Box>
        </AccordionDetails>
      </Accordion>

      {/* Model Status */}
      <Box sx={{ mt: 2, textAlign: 'center' }}>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Schedule sx={{ fontSize: 14, mr: 0.5 }} />
          Model loaded and ready for predictions
        </Typography>
      </Box>
    </Box>
  );
};

export default ModelInfo;
