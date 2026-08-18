const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 5000;
const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://127.0.0.1:8000';

app.use(cors());
app.use(express.json());

// Logger middleware
app.use((req, res, next) => {
  console.log(`[NodeServer] ${req.method} ${req.url}`);
  next();
});

// Proxy Health
app.get('/api/health', async (req, res) => {
  try {
    const response = await axios.get(`${PYTHON_API_URL}/api/health`);
    res.json(response.data);
  } catch (error) {
    console.error('[NodeServer] Error forwarding health check:', error.message);
    res.status(502).json({ error: "ML Service is offline or unreachable" });
  }
});

// Proxy Timeline
app.get('/api/timeline', async (req, res) => {
  try {
    const response = await axios.get(`${PYTHON_API_URL}/api/timeline`);
    res.json(response.data);
  } catch (error) {
    console.error('[NodeServer] Error forwarding timeline request:', error.message);
    res.status(502).json({ error: "ML Service timeline service unreachable" });
  }
});

// Proxy Metrics
app.get('/api/metrics', async (req, res) => {
  try {
    const response = await axios.get(`${PYTHON_API_URL}/api/metrics`);
    res.json(response.data);
  } catch (error) {
    console.error('[NodeServer] Error forwarding metrics request:', error.message);
    res.status(502).json({ error: "ML Service metrics service unreachable" });
  }
});

// Proxy Aircraft by Timestamp
app.get('/api/aircraft', async (req, res) => {
  const { timestamp } = req.query;
  if (!timestamp) {
    return res.status(400).json({ error: "Query parameter 'timestamp' is required" });
  }

  try {
    const response = await axios.get(`${PYTHON_API_URL}/api/aircraft`, {
      params: { timestamp }
    });
    res.json(response.data);
  } catch (error) {
    console.error('[NodeServer] Error forwarding aircraft request:', error.message);
    res.status(502).json({ error: "ML Service aircraft service unreachable" });
  }
});

// Proxy Graph snapshot
app.get('/api/graph/:timestamp', async (req, res) => {
  const { timestamp } = req.params;
  try {
    const response = await axios.get(`${PYTHON_API_URL}/api/graph/${encodeURIComponent(timestamp)}`);
    res.json(response.data);
  } catch (error) {
    console.error('[NodeServer] Error forwarding graph request:', error.message);
    res.status(502).json({ error: "ML Service graph service unreachable" });
  }
});

// Fallback Route
app.use((req, res) => {
  res.status(404).json({ error: "API Route Not Found" });
});

app.listen(PORT, () => {
  console.log(`[NodeServer] Express API server running on port ${PORT}`);
  console.log(`[NodeServer] Forwarding requests to FastAPI at ${PYTHON_API_URL}`);
});
