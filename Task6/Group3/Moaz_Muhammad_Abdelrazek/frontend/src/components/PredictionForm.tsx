import React, { useState, useEffect } from 'react';
import {
  Box,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  Typography,
  Chip,
  Alert,
  CircularProgress
} from '@mui/material';
import { DateTimePicker } from '@mui/x-date-pickers/DateTimePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs';
import dayjs, { Dayjs } from 'dayjs';
import { LocationOn, Schedule, People, Cloud, Traffic } from '@mui/icons-material';

import {
  PredictionRequest,
  PredictionFormProps,
  WEATHER_CONDITIONS,
  TRAFFIC_CONDITIONS
} from '../types/api';
import { validateCoordinates, getCurrentDateTime } from '../services/api';

const PredictionForm: React.FC<PredictionFormProps> = ({
  onSubmit,
  loading,
  pickupLocation,
  dropoffLocation
}) => {
  const [formData, setFormData] = useState({
    pickupLatitude: 40.7589,
    pickupLongitude: -73.9851,
    dropoffLatitude: 40.7505,
    dropoffLongitude: -73.9934,
    passengerCount: 2,
    pickupDateTime: dayjs(),
    weatherCondition: 'sunny',
    trafficCondition: 'flow traffic'
  });

  const [errors, setErrors] = useState<Record<string, string>>({});

  // Update form when map locations change
  useEffect(() => {
    if (pickupLocation) {
      setFormData(prev => ({
        ...prev,
        pickupLatitude: pickupLocation[0],
        pickupLongitude: pickupLocation[1]
      }));
    }
  }, [pickupLocation]);

  useEffect(() => {
    if (dropoffLocation) {
      setFormData(prev => ({
        ...prev,
        dropoffLatitude: dropoffLocation[0],
        dropoffLongitude: dropoffLocation[1]
      }));
    }
  }, [dropoffLocation]);

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};

    // Validate coordinates
    if (!validateCoordinates(formData.pickupLatitude, formData.pickupLongitude)) {
      newErrors.pickup = 'Invalid pickup coordinates';
    }
    if (!validateCoordinates(formData.dropoffLatitude, formData.dropoffLongitude)) {
      newErrors.dropoff = 'Invalid dropoff coordinates';
    }

    // Validate passenger count
    if (formData.passengerCount < 1 || formData.passengerCount > 8) {
      newErrors.passengers = 'Passenger count must be between 1 and 8';
    }

    // Check if pickup and dropoff are the same
    if (
      Math.abs(formData.pickupLatitude - formData.dropoffLatitude) < 0.0001 &&
      Math.abs(formData.pickupLongitude - formData.dropoffLongitude) < 0.0001
    ) {
      newErrors.locations = 'Pickup and dropoff locations cannot be the same';
    }

    // Validate datetime (not too far in the past or future)
    const now = dayjs();
    const selectedDate = formData.pickupDateTime;
    if (selectedDate.isBefore(now.subtract(1, 'year'))) {
      newErrors.datetime = 'Date cannot be more than 1 year in the past';
    }
    if (selectedDate.isAfter(now.add(1, 'year'))) {
      newErrors.datetime = 'Date cannot be more than 1 year in the future';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }

    const request: PredictionRequest = {
      pickup_latitude: formData.pickupLatitude,
      pickup_longitude: formData.pickupLongitude,
      dropoff_latitude: formData.dropoffLatitude,
      dropoff_longitude: formData.dropoffLongitude,
      passenger_count: formData.passengerCount,
      pickup_datetime: formData.pickupDateTime.format('YYYY-MM-DDTHH:mm:ss'),
      weather_condition: formData.weatherCondition,
      traffic_condition: formData.trafficCondition
    };

    onSubmit(request);
  };

  const handleInputChange = (field: string, value: any) => {
    // Round coordinate values to 6 decimal places
    if (field.includes('Latitude') || field.includes('Longitude')) {
      value = Math.round(value * 1000000) / 1000000;
    }

    setFormData(prev => ({ ...prev, [field]: value }));
    // Clear error when user starts typing
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }));
    }
  };

  const handleUseCurrentLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setFormData(prev => ({
            ...prev,
            pickupLatitude: position.coords.latitude,
            pickupLongitude: position.coords.longitude
          }));
        },
        (error) => {
          console.error('Error getting location:', error);
        }
      );
    }
  };

  const handleSetCurrentTime = () => {
    setFormData(prev => ({ ...prev, pickupDateTime: dayjs() }));
  };

  return (
    <LocalizationProvider dateAdapter={AdapterDayjs}>
      <Box component="form" onSubmit={handleSubmit} sx={{ width: '100%' }}>
        {/* Location Section */}
        <Typography variant="subtitle1" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
          <LocationOn sx={{ mr: 1 }} />
          Locations
        </Typography>
        
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Pickup Latitude"
              type="number"
              value={formData.pickupLatitude.toFixed(6)}
              onChange={(e) => handleInputChange('pickupLatitude', parseFloat(e.target.value))}
              error={!!errors.pickup}
              helperText={errors.pickup}
              inputProps={{ step: 0.000001, min: -90, max: 90 }}
              size="small"
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Pickup Longitude"
              type="number"
              value={formData.pickupLongitude.toFixed(6)}
              onChange={(e) => handleInputChange('pickupLongitude', parseFloat(e.target.value))}
              error={!!errors.pickup}
              inputProps={{ step: 0.000001, min: -180, max: 180 }}
              size="small"
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Dropoff Latitude"
              type="number"
              value={formData.dropoffLatitude.toFixed(6)}
              onChange={(e) => handleInputChange('dropoffLatitude', parseFloat(e.target.value))}
              error={!!errors.dropoff}
              helperText={errors.dropoff}
              inputProps={{ step: 0.000001, min: -90, max: 90 }}
              size="small"
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Dropoff Longitude"
              type="number"
              value={formData.dropoffLongitude.toFixed(6)}
              onChange={(e) => handleInputChange('dropoffLongitude', parseFloat(e.target.value))}
              error={!!errors.dropoff}
              inputProps={{ step: 0.000001, min: -180, max: 180 }}
              size="small"
            />
          </Grid>
        </Grid>

        {errors.locations && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {errors.locations}
          </Alert>
        )}

        <Box sx={{ mb: 3, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <Chip
            label="Use Current Location"
            onClick={handleUseCurrentLocation}
            variant="outlined"
            size="small"
          />
          <Chip
            label="NYC Times Square"
            onClick={() => {
              setFormData(prev => ({
                ...prev,
                pickupLatitude: 40.7589,
                pickupLongitude: -73.9851
              }));
            }}
            variant="outlined"
            size="small"
          />
          <Chip
            label="JFK Airport"
            onClick={() => {
              setFormData(prev => ({
                ...prev,
                dropoffLatitude: 40.6413,
                dropoffLongitude: -73.7781
              }));
            }}
            variant="outlined"
            size="small"
          />
        </Box>

        {/* Trip Details Section */}
        <Typography variant="subtitle1" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
          <People sx={{ mr: 1 }} />
          Trip Details
        </Typography>

        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Number of Passengers"
              type="number"
              value={formData.passengerCount}
              onChange={(e) => handleInputChange('passengerCount', parseInt(e.target.value))}
              error={!!errors.passengers}
              helperText={errors.passengers}
              inputProps={{ min: 1, max: 8 }}
              size="small"
            />
          </Grid>
          <Grid item xs={12}>
            <DateTimePicker
              label="Pickup Date & Time"
              value={formData.pickupDateTime}
              onChange={(newValue) => handleInputChange('pickupDateTime', newValue)}
              slotProps={{
                textField: {
                  fullWidth: true,
                  size: 'small',
                  error: !!errors.datetime,
                  helperText: errors.datetime
                }
              }}
            />
          </Grid>
        </Grid>

        <Box sx={{ mb: 3, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <Chip
            label="Now"
            onClick={handleSetCurrentTime}
            variant="outlined"
            size="small"
            icon={<Schedule />}
          />
          <Chip
            label="Morning Rush (8 AM)"
            onClick={() => {
              setFormData(prev => ({
                ...prev,
                pickupDateTime: dayjs().hour(8).minute(0).second(0)
              }));
            }}
            variant="outlined"
            size="small"
          />
          <Chip
            label="Evening Rush (6 PM)"
            onClick={() => {
              setFormData(prev => ({
                ...prev,
                pickupDateTime: dayjs().hour(18).minute(0).second(0)
              }));
            }}
            variant="outlined"
            size="small"
          />
        </Box>

        {/* Conditions Section */}
        <Typography variant="subtitle1" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
          <Cloud sx={{ mr: 1 }} />
          Conditions
        </Typography>

        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={6}>
            <FormControl fullWidth size="small">
              <InputLabel>Weather</InputLabel>
              <Select
                value={formData.weatherCondition}
                label="Weather"
                onChange={(e) => handleInputChange('weatherCondition', e.target.value)}
              >
                {WEATHER_CONDITIONS.map((condition) => (
                  <MenuItem key={condition.value} value={condition.value}>
                    {condition.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={6}>
            <FormControl fullWidth size="small">
              <InputLabel>Traffic</InputLabel>
              <Select
                value={formData.trafficCondition}
                label="Traffic"
                onChange={(e) => handleInputChange('trafficCondition', e.target.value)}
              >
                {TRAFFIC_CONDITIONS.map((condition) => (
                  <MenuItem key={condition.value} value={condition.value}>
                    {condition.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
        </Grid>

        {/* Submit Button */}
        <Button
          type="submit"
          variant="contained"
          fullWidth
          size="large"
          disabled={loading}
          sx={{ mt: 2 }}
        >
          {loading ? (
            <>
              <CircularProgress size={20} sx={{ mr: 1 }} />
              Predicting...
            </>
          ) : (
            'Predict Fare'
          )}
        </Button>
      </Box>
    </LocalizationProvider>
  );
};

export default PredictionForm;
