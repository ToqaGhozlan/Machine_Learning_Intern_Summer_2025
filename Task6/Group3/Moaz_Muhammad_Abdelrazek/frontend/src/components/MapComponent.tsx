import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMapEvents } from 'react-leaflet';
import { Icon, LatLng } from 'leaflet';
import { Box, Typography, Chip, Alert } from '@mui/material';
import { LocationOn, Flag } from '@mui/icons-material';

import { MapComponentProps, DEFAULT_MAP_CENTER, DEFAULT_MAP_ZOOM } from '../types/api';
import { calculateDistance } from '../services/api';

// Import Leaflet CSS
import 'leaflet/dist/leaflet.css';

// Fix for default markers in React Leaflet
delete (Icon.Default.prototype as any)._getIconUrl;
Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

// Custom icons for pickup and dropoff
const pickupIcon = new Icon({
  iconUrl: 'data:image/svg+xml;base64,' + btoa(`
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#4CAF50" width="32" height="32">
      <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
    </svg>
  `),
  iconSize: [32, 32],
  iconAnchor: [16, 32],
  popupAnchor: [0, -32],
});

const dropoffIcon = new Icon({
  iconUrl: 'data:image/svg+xml;base64,' + btoa(`
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#F44336" width="32" height="32">
      <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
    </svg>
  `),
  iconSize: [32, 32],
  iconAnchor: [16, 32],
  popupAnchor: [0, -32],
});

interface MapEventsProps {
  onLocationSelect: (type: 'pickup' | 'dropoff', location: [number, number]) => void;
  nextMarkerType: 'pickup' | 'dropoff';
}

// Component to handle map click events
const MapEvents: React.FC<MapEventsProps> = ({ onLocationSelect, nextMarkerType }) => {
  useMapEvents({
    click: (e) => {
      const { lat, lng } = e.latlng;
      // Round coordinates to 6 decimal places for precision
      const roundedLat = Math.round(lat * 1000000) / 1000000;
      const roundedLng = Math.round(lng * 1000000) / 1000000;
      onLocationSelect(nextMarkerType, [roundedLat, roundedLng]);
    },
  });
  return null;
};

const MapComponent: React.FC<MapComponentProps> = ({
  pickupLocation,
  dropoffLocation,
  onLocationSelect
}) => {
  const [nextMarkerType, setNextMarkerType] = useState<'pickup' | 'dropoff'>('pickup');
  const [distance, setDistance] = useState<number | null>(null);

  // Calculate distance when both locations are set
  useEffect(() => {
    if (pickupLocation && dropoffLocation) {
      const dist = calculateDistance(
        pickupLocation[0],
        pickupLocation[1],
        dropoffLocation[0],
        dropoffLocation[1]
      );
      setDistance(dist);
    } else {
      setDistance(null);
    }
  }, [pickupLocation, dropoffLocation]);

  // Auto-switch to dropoff after pickup is set
  useEffect(() => {
    if (pickupLocation && !dropoffLocation) {
      setNextMarkerType('dropoff');
    } else if (!pickupLocation) {
      setNextMarkerType('pickup');
    }
  }, [pickupLocation, dropoffLocation]);

  const handleLocationSelect = (type: 'pickup' | 'dropoff', location: [number, number]) => {
    onLocationSelect(type, location);
    
    // Auto-switch marker type
    if (type === 'pickup' && !dropoffLocation) {
      setNextMarkerType('dropoff');
    } else if (type === 'dropoff' && !pickupLocation) {
      setNextMarkerType('pickup');
    }
  };

  const handleMarkerTypeChange = (type: 'pickup' | 'dropoff') => {
    setNextMarkerType(type);
  };

  return (
    <Box sx={{ height: '100%', position: 'relative' }}>
      {/* Map Controls */}
      <Box
        sx={{
          position: 'absolute',
          top: 10,
          left: 10,
          zIndex: 1000,
          display: 'flex',
          flexDirection: 'column',
          gap: 1,
        }}
      >
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Chip
            label="Set Pickup"
            icon={<LocationOn />}
            color={nextMarkerType === 'pickup' ? 'primary' : 'default'}
            onClick={() => handleMarkerTypeChange('pickup')}
            variant={nextMarkerType === 'pickup' ? 'filled' : 'outlined'}
            size="small"
            sx={{ backgroundColor: nextMarkerType === 'pickup' ? '#4CAF50' : undefined }}
          />
          <Chip
            label="Set Dropoff"
            icon={<Flag />}
            color={nextMarkerType === 'dropoff' ? 'primary' : 'default'}
            onClick={() => handleMarkerTypeChange('dropoff')}
            variant={nextMarkerType === 'dropoff' ? 'filled' : 'outlined'}
            size="small"
            sx={{ backgroundColor: nextMarkerType === 'dropoff' ? '#F44336' : undefined }}
          />
        </Box>
        
        {distance && (
          <Chip
            label={`Distance: ${distance.toFixed(2)} miles`}
            size="small"
            variant="outlined"
            sx={{ backgroundColor: 'rgba(255, 255, 255, 0.9)' }}
          />
        )}
      </Box>

      {/* Instructions */}
      <Box
        sx={{
          position: 'absolute',
          bottom: 10,
          left: 10,
          right: 10,
          zIndex: 1000,
        }}
      >
        <Alert severity="info" sx={{ opacity: 0.9 }}>
          <Typography variant="body2">
            Click on the map to set {nextMarkerType} location. 
            {!pickupLocation && !dropoffLocation && ' Start with pickup location.'}
            {pickupLocation && !dropoffLocation && ' Now set dropoff location.'}
            {pickupLocation && dropoffLocation && ' Both locations set. Click to update.'}
          </Typography>
        </Alert>
      </Box>

      {/* Map */}
      <MapContainer
        center={DEFAULT_MAP_CENTER}
        zoom={DEFAULT_MAP_ZOOM}
        style={{ height: '100%', width: '100%' }}
        scrollWheelZoom={true}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        
        {/* Map click handler */}
        <MapEvents
          onLocationSelect={handleLocationSelect}
          nextMarkerType={nextMarkerType}
        />
        
        {/* Pickup marker */}
        {pickupLocation && (
          <Marker position={pickupLocation} icon={pickupIcon}>
            <Popup>
              <Box>
                <Typography variant="subtitle2" sx={{ color: '#4CAF50', mb: 1 }}>
                  <LocationOn sx={{ fontSize: 16, mr: 0.5 }} />
                  Pickup Location
                </Typography>
                <Typography variant="body2">
                  Lat: {pickupLocation[0].toFixed(6)}
                </Typography>
                <Typography variant="body2">
                  Lng: {pickupLocation[1].toFixed(6)}
                </Typography>
              </Box>
            </Popup>
          </Marker>
        )}
        
        {/* Dropoff marker */}
        {dropoffLocation && (
          <Marker position={dropoffLocation} icon={dropoffIcon}>
            <Popup>
              <Box>
                <Typography variant="subtitle2" sx={{ color: '#F44336', mb: 1 }}>
                  <Flag sx={{ fontSize: 16, mr: 0.5 }} />
                  Dropoff Location
                </Typography>
                <Typography variant="body2">
                  Lat: {dropoffLocation[0].toFixed(6)}
                </Typography>
                <Typography variant="body2">
                  Lng: {dropoffLocation[1].toFixed(6)}
                </Typography>
              </Box>
            </Popup>
          </Marker>
        )}
      </MapContainer>
    </Box>
  );
};

export default MapComponent;
