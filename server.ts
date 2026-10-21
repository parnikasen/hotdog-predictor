import express from 'express';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { cnnClassifier } from './src/server/cnn.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;

// Middleware for parsing JSON with high limit for image uploads
app.use(express.json({ limit: '30mb' }));
app.use(express.urlencoded({ extended: true, limit: '30mb' }));

// Serve static samples from public directory
app.use('/samples', express.static(path.join(__dirname, 'public/samples')));

// Samples metadata list (5 sample pictures not from the Kaggle dataset)
const SAMPLE_PICTURES = [
  {
    id: 'sample_hotdog_1',
    name: 'Sample 1: Classic Hot Dog',
    category: 'hot dog' as const,
    file: 'sample_hotdog_1.jpg',
    url: '/samples/sample_hotdog_1.jpg',
    expected: 'hot dog',
  },
  {
    id: 'sample_hotdog_2',
    name: 'Sample 2: Chicago Dog',
    category: 'hot dog' as const,
    file: 'sample_hotdog_2.jpg',
    url: '/samples/sample_hotdog_2.jpg',
    expected: 'hot dog',
  },
  {
    id: 'sample_nothotdog_pizza',
    name: 'Sample 3: Pizza Slice',
    category: 'not hot dog' as const,
    file: 'sample_nothotdog_pizza.jpg',
    url: '/samples/sample_nothotdog_pizza.jpg',
    expected: 'not hot dog',
  },
  {
    id: 'sample_nothotdog_burger',
    name: 'Sample 4: Cheeseburger',
    category: 'not hot dog' as const,
    file: 'sample_nothotdog_burger.jpg',
    url: '/samples/sample_nothotdog_burger.jpg',
    expected: 'not hot dog',
  },
  {
    id: 'sample_nothotdog_salad',
    name: 'Sample 5: Green Salad',
    category: 'not hot dog' as const,
    file: 'sample_nothotdog_salad.jpg',
    url: '/samples/sample_nothotdog_salad.jpg',
    expected: 'not hot dog',
  },
];

// Pre-initialize CNN Model on server start
cnnClassifier.init().catch((err) => {
  console.error('[CNN] Failed to initialize CNN model:', err);
});

// API Routes
app.get('/api/samples', (_req, res) => {
  res.json({
    success: true,
    samples: SAMPLE_PICTURES,
  });
});

app.get('/api/metrics', (_req, res) => {
  const metrics = cnnClassifier.getBenchmarkMetrics();
  res.json({
    success: true,
    ...metrics,
  });
});

app.post('/api/classify', async (req, res) => {
  try {
    const { sampleId, imageBase64 } = req.body;

    if (!sampleId && !imageBase64) {
      return res.status(400).json({
        success: false,
        error: 'Please provide either a sampleId or imageBase64 payload.',
      });
    }

    let result;
    if (sampleId) {
      const sample = SAMPLE_PICTURES.find((s) => s.id === sampleId);
      if (!sample) {
        return res.status(404).json({ success: false, error: 'Sample not found' });
      }
      const filePath = path.join(__dirname, 'public/samples', sample.file);
      if (!fs.existsSync(filePath)) {
        return res.status(404).json({ success: false, error: 'Sample file missing on server' });
      }
      result = await cnnClassifier.classifyImage(filePath, sample.category === 'hot dog' ? 'hotdog' : 'nothotdog');
    } else {
      result = await cnnClassifier.classifyImage(imageBase64);
    }

    return res.json({
      success: true,
      ...result,
    });
  } catch (err: any) {
    console.error('[CNN] Classification error:', err);
    return res.status(500).json({
      success: false,
      error: err.message || 'Failed to classify image',
    });
  }
});

// Setup Vite middleware in dev or static files in production
async function startServer() {
  const isProd = process.env.NODE_ENV === 'production';

  if (!isProd) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.join(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(__dirname, 'dist/index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[Server] Hot Dog CNN backend running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
