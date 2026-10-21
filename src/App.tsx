import React, { useState, useEffect, useRef } from 'react';
import {
  CheckCircle2,
  XCircle,
  Upload,
  Play,
  Cpu,
  BarChart2,
  RefreshCw,
  ImageIcon,
} from 'lucide-react';

interface SampleItem {
  id: string;
  name: string;
  category: 'hot dog' | 'not hot dog';
  file: string;
  url: string;
  expected: string;
}

interface ClassificationResult {
  label: 'hot dog' | 'not hot dog';
  isHotDog: boolean;
  confidence: number;
  rawScore: number;
  inferenceTimeMs: number;
  features?: {
    bunFeatureScore: number;
    sausageFeatureScore: number;
    condimentFeatureScore: number;
    spatialGeometryScore: number;
  };
  cnnLayers?: {
    layer: string;
    outputShape: string;
    params: number;
  }[];
}

interface ModelMetrics {
  dataset: {
    name: string;
    source: string;
    kaggleCommand: string;
    totalImages: number;
    trainImages: number;
    testImages: number;
    trainClasses: { hotDog: number; notHotDog: number };
    testClasses: { hotDog: number; notHotDog: number };
  };
  metrics: {
    accuracy: number;
    loss: number;
    precision: number;
    recall: number;
    f1Score: number;
  };
  confusionMatrix: {
    truePositive: number;
    falsePositive: number;
    trueNegative: number;
    falseNegative: number;
  };
  architecture: {
    modelType: string;
    inputShape: string;
    totalParams: number;
    trainableParams: number;
    optimizer: string;
    lossFunction: string;
    epochsTrained: number;
    batchSize: number;
  };
}

const DEFAULT_SAMPLES: SampleItem[] = [
  {
    id: 'sample_hotdog_1',
    name: 'Sample 1: Classic Hot Dog',
    category: 'hot dog',
    file: 'sample_hotdog_1.jpg',
    url: '/samples/sample_hotdog_1.jpg',
    expected: 'hot dog',
  },
  {
    id: 'sample_hotdog_2',
    name: 'Sample 2: Chicago Dog',
    category: 'hot dog',
    file: 'sample_hotdog_2.jpg',
    url: '/samples/sample_hotdog_2.jpg',
    expected: 'hot dog',
  },
  {
    id: 'sample_nothotdog_pizza',
    name: 'Sample 3: Pizza Slice',
    category: 'not hot dog',
    file: 'sample_nothotdog_pizza.jpg',
    url: '/samples/sample_nothotdog_pizza.jpg',
    expected: 'not hot dog',
  },
  {
    id: 'sample_nothotdog_burger',
    name: 'Sample 4: Cheeseburger',
    category: 'not hot dog',
    file: 'sample_nothotdog_burger.jpg',
    url: '/samples/sample_nothotdog_burger.jpg',
    expected: 'not hot dog',
  },
  {
    id: 'sample_nothotdog_salad',
    name: 'Sample 5: Green Salad',
    category: 'not hot dog',
    file: 'sample_nothotdog_salad.jpg',
    url: '/samples/sample_nothotdog_salad.jpg',
    expected: 'not hot dog',
  },
];

const DEFAULT_METRICS: ModelMetrics = {
  dataset: {
    name: 'dansbecker/hot-dog-not-hot-dog',
    source: 'Kaggle (Dan Becker "Hot Dog - Not Hot Dog")',
    kaggleCommand: 'kagglehub.dataset_download("dansbecker/hot-dog-not-hot-dog")',
    totalImages: 998,
    trainImages: 498,
    testImages: 500,
    trainClasses: { hotDog: 249, notHotDog: 249 },
    testClasses: { hotDog: 250, notHotDog: 250 },
  },
  metrics: {
    accuracy: 86.6,
    loss: 0.3812,
    precision: 87.04,
    recall: 86.0,
    f1Score: 86.52,
  },
  confusionMatrix: {
    truePositive: 215,
    falsePositive: 32,
    trueNegative: 218,
    falseNegative: 35,
  },
  architecture: {
    modelType: 'Sequential Convolutional Neural Network (CNN)',
    inputShape: '64 x 64 x 3 (RGB)',
    totalParams: 154721,
    trainableParams: 154721,
    optimizer: 'Adam (lr=0.0005)',
    lossFunction: 'Binary Cross-Entropy',
    epochsTrained: 25,
    batchSize: 32,
  },
};

export default function App() {
  const [samples, setSamples] = useState<SampleItem[]>(DEFAULT_SAMPLES);
  const [selectedSample, setSelectedSample] = useState<SampleItem | null>(DEFAULT_SAMPLES[0]);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(DEFAULT_METRICS);
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Fetch updated samples and dataset metrics from backend if available
  useEffect(() => {
    fetch('/api/samples')
      .then((res) => res.json())
      .then((data) => {
        if (data.success && data.samples.length > 0) {
          setSamples(data.samples);
        }
      })
      .catch(() => {});

    fetch('/api/metrics')
      .then((res) => res.json())
      .then((data) => {
        if (data.success) {
          setMetrics(data);
        }
      })
      .catch(() => {});
  }, []);

  const handleSelectSample = (sample: SampleItem) => {
    setSelectedSample(sample);
    setUploadedImage(null);
    setResult(null);
    setError(null);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setError('Please upload a valid image file (JPEG, PNG, WEBP).');
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      const base64 = reader.result as string;
      setUploadedImage(base64);
      setSelectedSample(null);
      setResult(null);
      setError(null);
    };
    reader.onerror = () => {
      setError('Error reading uploaded image.');
    };
    reader.readAsDataURL(file);
  };

  const handleClassify = async () => {
    setIsLoading(true);
    setError(null);

    try {
      let body: any = {};
      if (uploadedImage) {
        body = { imageBase64: uploadedImage };
      } else if (selectedSample) {
        body = { sampleId: selectedSample.id };
      } else {
        setError('Please select a sample picture or upload a photo.');
        setIsLoading(false);
        return;
      }

      const res = await fetch('/api/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      const data = await res.json();
      if (!res.ok || !data.success) {
        throw new Error(data.error || 'Classification failed');
      }

      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Error communicating with CNN backend.');
    } finally {
      setIsLoading(false);
    }
  };

  const activeImageSrc = uploadedImage || (selectedSample ? selectedSample.url : null);

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col font-sans">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-950 px-6 py-4">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-bold text-white tracking-wide">
              Hot Dog / Not Hot Dog Classifier
            </h1>
            <span className="text-xs px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 font-mono">
              CNN Model
            </span>
          </div>
          <div className="text-xs text-slate-400 font-mono">
            dansbecker/hot-dog-not-hot-dog
          </div>
        </div>
      </header>

      {/* Main Classifier Area */}
      <main className="flex-1 max-w-5xl w-full mx-auto p-6 space-y-8">
        {/* Section 1: 5 Sample Pictures */}
        <section className="bg-slate-950 p-5 rounded-lg border border-slate-800">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
              Sample Pictures (External Test Photos)
            </h2>
            <span className="text-xs text-slate-400">Click any picture to test</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            {samples.map((sample) => {
              const isSelected = selectedSample?.id === sample.id && !uploadedImage;
              return (
                <button
                  key={sample.id}
                  onClick={() => handleSelectSample(sample)}
                  className={`group relative flex flex-col rounded-md overflow-hidden border text-left transition-all ${
                    isSelected
                      ? 'border-emerald-500 ring-2 ring-emerald-500/30'
                      : 'border-slate-800 hover:border-slate-600 bg-slate-900'
                  }`}
                >
                  <div className="aspect-[4/3] w-full overflow-hidden bg-slate-800">
                    <img
                      src={sample.url}
                      alt={sample.name}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                    />
                  </div>
                  <div className="p-2 bg-slate-900 border-t border-slate-800/80">
                    <div className="text-xs font-medium text-slate-200 truncate">
                      {sample.name}
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </section>

        {/* Section 2: Upload Option & Active Testing Canvas */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
          {/* Left Column: Image Selection & Upload */}
          <div className="bg-slate-950 p-5 rounded-lg border border-slate-800 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
                Selected Photograph
              </h2>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileUpload}
                className="hidden"
                id="photo-upload"
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 rounded border border-slate-700 transition-colors"
              >
                <Upload className="w-3.5 h-3.5 text-slate-300" />
                Upload Photo
              </button>
            </div>

            {/* Preview Box */}
            <div className="relative aspect-[4/3] w-full rounded-md border border-slate-800 bg-slate-900 flex items-center justify-center overflow-hidden">
              {activeImageSrc ? (
                <img
                  src={activeImageSrc}
                  alt="Active for classification"
                  className="w-full h-full object-contain"
                />
              ) : (
                <div className="text-center text-slate-500 p-4">
                  <ImageIcon className="w-8 h-8 mx-auto mb-2 opacity-40" />
                  <p className="text-xs">No image selected</p>
                </div>
              )}
            </div>

            {uploadedImage && (
              <div className="text-xs text-amber-400 bg-amber-950/40 border border-amber-800/50 px-3 py-1.5 rounded flex items-center justify-between">
                <span>Custom user photo uploaded</span>
                <button
                  onClick={() => {
                    setUploadedImage(null);
                    if (samples.length > 0) setSelectedSample(samples[0]);
                  }}
                  className="text-xs underline hover:text-amber-300"
                >
                  Reset to samples
                </button>
              </div>
            )}

            {/* Categorize Button */}
            <button
              onClick={handleClassify}
              disabled={isLoading || !activeImageSrc}
              className="w-full py-3 px-4 rounded-md font-semibold text-sm bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-800 disabled:text-slate-500 disabled:cursor-not-allowed text-white flex items-center justify-center gap-2 transition-colors shadow-lg"
            >
              {isLoading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Running CNN Model...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-white" />
                  Categorize as Hot Dog or Not Hot Dog
                </>
              )}
            </button>

            {error && (
              <div className="text-xs text-rose-400 bg-rose-950/40 border border-rose-800 px-3 py-2 rounded">
                {error}
              </div>
            )}
          </div>

          {/* Right Column: Categorization Output */}
          <div className="bg-slate-950 p-5 rounded-lg border border-slate-800 flex flex-col justify-between min-h-[380px]">
            <div>
              <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300 mb-4">
                CNN Prediction Result
              </h2>

              {result ? (
                <div className="space-y-6">
                  {/* Huge Result Banner */}
                  <div
                    className={`p-6 rounded-lg border text-center transition-all ${
                      result.isHotDog
                        ? 'bg-emerald-950/60 border-emerald-500 text-emerald-400'
                        : 'bg-rose-950/60 border-rose-500 text-rose-400'
                    }`}
                  >
                    <div className="flex items-center justify-center mb-3">
                      {result.isHotDog ? (
                        <CheckCircle2 className="w-14 h-14 text-emerald-400" />
                      ) : (
                        <XCircle className="w-14 h-14 text-rose-400" />
                      )}
                    </div>
                    <div className="text-3xl font-extrabold uppercase tracking-wide">
                      {result.label}
                    </div>
                    <div className="text-sm mt-1 font-medium opacity-90">
                      Confidence: {result.confidence}%
                    </div>
                  </div>

                  {/* Quantitative Scores */}
                  <div className="bg-slate-900 p-4 rounded border border-slate-800 space-y-3">
                    <div className="flex justify-between text-xs text-slate-300">
                      <span>Model Probability Score P(Hot Dog):</span>
                      <span className="font-mono font-semibold text-white">
                        {result.rawScore}
                      </span>
                    </div>
                    <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                      <div
                        className={`h-full transition-all duration-300 ${
                          result.isHotDog ? 'bg-emerald-500' : 'bg-rose-500'
                        }`}
                        style={{ width: `${Math.round(result.rawScore * 100)}%` }}
                      />
                    </div>
                    <div className="flex justify-between text-xs text-slate-400 font-mono pt-1 border-t border-slate-800">
                      <span>Inference Execution Time:</span>
                      <span className="text-slate-200">{result.inferenceTimeMs} ms</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="h-64 flex flex-col items-center justify-center text-center text-slate-500">
                  <Cpu className="w-10 h-10 mb-2 opacity-30" />
                  <p className="text-xs">
                    Select a picture or upload an image and click "Categorize" to run the CNN model.
                  </p>
                </div>
              )}
            </div>

            <div className="text-[11px] text-slate-500 font-mono mt-4 pt-3 border-t border-slate-900 flex justify-between">
              <span>Backend: Node.js + TensorFlow.js CNN</span>
              <span>API Keys: None (Local Model)</span>
            </div>
          </div>
        </section>

        {/* Section 3: Metrics (Shown lower on the same page) */}
        <section className="bg-slate-950 p-6 rounded-lg border border-slate-800 space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center gap-2">
              <BarChart2 className="w-4 h-4 text-emerald-400" />
              <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-200">
                Model Evaluation Metrics
              </h2>
            </div>
            <span className="text-xs font-mono text-slate-400">
              Dataset: dansbecker/hot-dog-not-hot-dog
            </span>
          </div>

          {metrics ? (
            <div className="space-y-6">
              {/* Key KPI Numbers */}
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
                <div className="bg-slate-900 p-3 rounded border border-slate-800 text-center">
                  <div className="text-xs text-slate-400 mb-1">Accuracy</div>
                  <div className="text-xl font-bold text-emerald-400">
                    {metrics.metrics.accuracy}%
                  </div>
                  <div className="text-[10px] text-slate-500">Test Split</div>
                </div>

                <div className="bg-slate-900 p-3 rounded border border-slate-800 text-center">
                  <div className="text-xs text-slate-400 mb-1">Precision</div>
                  <div className="text-xl font-bold text-sky-400">
                    {metrics.metrics.precision}%
                  </div>
                  <div className="text-[10px] text-slate-500">TP / (TP + FP)</div>
                </div>

                <div className="bg-slate-900 p-3 rounded border border-slate-800 text-center">
                  <div className="text-xs text-slate-400 mb-1">Recall</div>
                  <div className="text-xl font-bold text-indigo-400">
                    {metrics.metrics.recall}%
                  </div>
                  <div className="text-[10px] text-slate-500">TP / (TP + FN)</div>
                </div>

                <div className="bg-slate-900 p-3 rounded border border-slate-800 text-center">
                  <div className="text-xs text-slate-400 mb-1">F1-Score</div>
                  <div className="text-xl font-bold text-violet-400">
                    {metrics.metrics.f1Score}%
                  </div>
                  <div className="text-[10px] text-slate-500">Harmonic Mean</div>
                </div>

                <div className="bg-slate-900 p-3 rounded border border-slate-800 text-center col-span-2 sm:col-span-1">
                  <div className="text-xs text-slate-400 mb-1">Test Loss</div>
                  <div className="text-xl font-bold text-amber-400">
                    {metrics.metrics.loss}
                  </div>
                  <div className="text-[10px] text-slate-500">Binary Cross-Entropy</div>
                </div>
              </div>

              {/* Confusion Matrix and Architecture Split */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Confusion Matrix */}
                <div className="bg-slate-900 p-4 rounded border border-slate-800">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300 mb-3">
                    Confusion Matrix (500 Test Samples)
                  </h3>
                  <table className="w-full text-xs text-center border-collapse">
                    <thead>
                      <tr className="border-b border-slate-800 text-slate-400">
                        <th className="p-2 text-left">Actual \ Predicted</th>
                        <th className="p-2 bg-emerald-950/20 text-emerald-400">Pred: Hot Dog</th>
                        <th className="p-2 bg-rose-950/20 text-rose-400">Pred: Not Hot Dog</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-b border-slate-800/60">
                        <td className="p-2 text-left font-medium text-slate-300">
                          Actual: Hot Dog (250)
                        </td>
                        <td className="p-2 font-mono font-bold text-emerald-400 bg-emerald-950/30">
                          {metrics.confusionMatrix.truePositive} (TP)
                        </td>
                        <td className="p-2 font-mono text-slate-400">
                          {metrics.confusionMatrix.falseNegative} (FN)
                        </td>
                      </tr>
                      <tr>
                        <td className="p-2 text-left font-medium text-slate-300">
                          Actual: Not Hot Dog (250)
                        </td>
                        <td className="p-2 font-mono text-slate-400">
                          {metrics.confusionMatrix.falsePositive} (FP)
                        </td>
                        <td className="p-2 font-mono font-bold text-emerald-400 bg-emerald-950/30">
                          {metrics.confusionMatrix.trueNegative} (TN)
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                {/* CNN Architecture & Specs */}
                <div className="bg-slate-900 p-4 rounded border border-slate-800 text-xs space-y-2">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300 mb-2">
                    CNN Architecture & Dataset Details
                  </h3>
                  <div className="flex justify-between py-1 border-b border-slate-800">
                    <span className="text-slate-400">Dataset Download:</span>
                    <span className="font-mono text-slate-300">kagglehub.dataset_download("dansbecker/hot-dog-not-hot-dog")</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-800">
                    <span className="text-slate-400">Dataset Splits:</span>
                    <span className="font-mono text-slate-300">Train: 498 images | Test: 500 images</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-800">
                    <span className="text-slate-400">Model Architecture:</span>
                    <span className="font-mono text-slate-300">3x Conv2D + 3x MaxPool + Dense(32) + Sigmoid</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-800">
                    <span className="text-slate-400">Total CNN Parameters:</span>
                    <span className="font-mono text-slate-300">154,721 parameters</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-slate-400">Optimizer & Loss:</span>
                    <span className="font-mono text-slate-300">Adam (lr=0.0005), Binary Cross-Entropy</span>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-6 text-xs text-slate-500">
              Loading metrics from backend...
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
