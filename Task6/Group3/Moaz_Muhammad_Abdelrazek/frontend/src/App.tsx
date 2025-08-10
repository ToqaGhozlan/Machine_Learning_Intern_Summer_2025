import React, { useState, useEffect } from 'react';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  Container,
  AppBar,
  Toolbar,
  Typography,
  Box,
  Paper,
  Grid,
  Alert,
  Snackbar
} from '@mui/material';
import { LocalTaxiOutlined } from '@mui/icons-material';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

import PredictionForm from './components/PredictionForm';
import MapComponent from './components/MapComponent';
import ResultsDisplay from './components/ResultsDisplay';
import ModelInfo from './components/ModelInfo';
import { PredictionRequest, PredictionResponse } from './types/api';
import { predictFare, getModelInfo } from './services/api';
import './App.css';

// Create Material-UI theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#ffc107',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    h4: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 500,
    },
  },
});

interface AppState {
  prediction: PredictionResponse | null;
  loading: boolean;
  error: string | null;
  modelInfo: any;
  pickupLocation: [number, number] | null;
  dropoffLocation: [number, number] | null;
}

function App() {
  const [state, setState] = useState<AppState>({
    prediction: null,
    loading: false,
    error: null,
    modelInfo: null,
    pickupLocation: null,
    dropoffLocation: null,
  });

  // Load model info on component mount
  useEffect(() => {
    const loadModelInfo = async () => {
      try {
        const info = await getModelInfo();
        setState(prev => ({ ...prev, modelInfo: info }));
      } catch (error) {
        console.error('Failed to load model info:', error);
      }
    };

    loadModelInfo();
  }, []);

  const handlePrediction = async (request: PredictionRequest) => {
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      const result = await predictFare(request);
      setState(prev => ({ 
        ...prev, 
        prediction: result, 
        loading: false,
        error: result.status === 'error' ? result.error_message || 'Prediction failed' : null
      }));
    } catch (error) {
      setState(prev => ({ 
        ...prev, 
        loading: false, 
        error: error instanceof Error ? error.message : 'An unexpected error occurred'
      }));
    }
  };

  const handleLocationSelect = (type: 'pickup' | 'dropoff', location: [number, number]) => {
    if (type === 'pickup') {
      setState(prev => ({ ...prev, pickupLocation: location }));
    } else {
      setState(prev => ({ ...prev, dropoffLocation: location }));
    }
  };

  const handleCloseError = () => {
    setState(prev => ({ ...prev, error: null }));
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <div className="App">
        {/* Header */}
        <AppBar position="static" elevation={2}>
          <Toolbar>
            <LocalTaxiOutlined sx={{ mr: 2 }} />
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              Taxi Fare Prediction
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.8 }}>
              Powered by ML | v1.0.0
            </Typography>
          </Toolbar>
        </AppBar>

        {/* Main Content */}
        <Container maxWidth="xl" sx={{ mt: 3, mb: 3 }}>
          <Grid container spacing={3}>
            {/* Left Column - Form and Results */}
            <Grid item xs={12} lg={4}>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                {/* Prediction Form */}
                <Paper elevation={3} sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Trip Details
                  </Typography>
                  <PredictionForm
                    onSubmit={handlePrediction}
                    loading={state.loading}
                    pickupLocation={state.pickupLocation}
                    dropoffLocation={state.dropoffLocation}
                  />
                </Paper>

                {/* Results Display */}
                {(state.prediction || state.loading) && (
                  <Paper elevation={3} sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom>
                      Prediction Results
                    </Typography>
                    <ResultsDisplay
                      prediction={state.prediction}
                      loading={state.loading}
                    />
                  </Paper>
                )}

                {/* Model Information */}
                {state.modelInfo && (
                  <Paper elevation={3} sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom>
                      Model Information
                    </Typography>
                    <ModelInfo modelInfo={state.modelInfo} />
                  </Paper>
                )}
              </Box>
            </Grid>

            {/* Right Column - Map */}
            <Grid item xs={12} lg={8}>
              <Paper elevation={3} sx={{ height: '80vh', overflow: 'hidden' }}>
                <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
                  <Typography variant="h6">
                    Interactive Map
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Click on the map to set pickup and dropoff locations
                  </Typography>
                </Box>
                <Box sx={{ height: 'calc(100% - 80px)' }}>
                  <MapComponent
                    pickupLocation={state.pickupLocation}
                    dropoffLocation={state.dropoffLocation}
                    onLocationSelect={handleLocationSelect}
                  />
                </Box>
              </Paper>
            </Grid>
          </Grid>
        </Container>

        {/* Error Snackbar */}
        <Snackbar
          open={!!state.error}
          autoHideDuration={6000}
          onClose={handleCloseError}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        >
          <Alert onClose={handleCloseError} severity="error" sx={{ width: '100%' }}>
            {state.error}
          </Alert>
        </Snackbar>

        {/* Toast Container */}
        <ToastContainer
          position="top-right"
          autoClose={5000}
          hideProgressBar={false}
          newestOnTop={false}
          closeOnClick
          rtl={false}
          pauseOnFocusLoss
          draggable
          pauseOnHover
        />
      </div>
    </ThemeProvider>
  );
}

export default App;
