import React, { useState, useEffect, useRef } from 'react';
import { DNSObservation, DetectionResult, InferenceMode, StreamState } from './types';
import { checkHealth, detectBatch } from './api/client';
import { DeepDNSStreamClient } from './api/websocket';
import { Header, ActiveTab } from './components/Header';
import { LandingPage } from './components/LandingPage';
import { WorkflowGuide } from './components/WorkflowGuide';
import { PipelineOverview } from './components/PipelineOverview';
import { ObservationHorizonBar } from './components/ObservationHorizonBar';
import { EvidenceTrajectoryChart } from './components/EvidenceTrajectoryChart';
import { DecisionPanel } from './components/DecisionPanel';
import { StreamConsole } from './components/StreamConsole';
import { InputController } from './components/InputController';
import { CustomStreamBuilder } from './components/CustomStreamBuilder';
import { InputSchemaModal } from './components/InputSchemaModal';
import { SAMPLE_ATTACK_STREAM } from './data/sampleStreams';
import { AlertCircle } from 'lucide-react';

export const App: React.FC = () => {
  // Navigation State: Always default to landing page on initial site load
  const [activeTab, setActiveTab] = useState<ActiveTab>('landing');
  const [isSchemaModalOpen, setIsSchemaModalOpen] = useState<boolean>(false);

  // Service & Engine State
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Inference Configuration
  const [mode, setMode] = useState<InferenceMode>('in_distribution');
  const [activeDatasetName, setActiveDatasetName] = useState<string>('CIC-Bell Light Text Attack');

  // Stream & Evidence State
  const [observations, setObservations] = useState<DNSObservation[]>(SAMPLE_ATTACK_STREAM);
  const [streamCount, setStreamCount] = useState<number>(0);
  const [evidenceHistory, setEvidenceHistory] = useState<Record<string, number>>({});
  const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null);
  const [streamState, setStreamState] = useState<StreamState>('IDLE');
  const [evaluating, setEvaluating] = useState<boolean>(false);

  // WebSocket reference
  const wsClientRef = useRef<DeepDNSStreamClient | null>(null);
  const streamIntervalRef = useRef<number | null>(null);
  const isDecisionMadeRef = useRef<boolean>(false);

  // Check Health on Mount
  const fetchHealth = async () => {
    setErrorMessage(null);
    try {
      await checkHealth();
    } catch (err: any) {
      setErrorMessage(
        'Unable to connect to DeepDNS serving layer (http://localhost:8000). Ensure the FastAPI backend is running.'
      );
    }
  };

  useEffect(() => {
    fetchHealth();
    return () => {
      resetSession();
    };
  }, []);

  const resetSession = () => {
    if (streamIntervalRef.current) {
      window.clearInterval(streamIntervalRef.current);
      streamIntervalRef.current = null;
    }
    if (wsClientRef.current) {
      wsClientRef.current.disconnect();
      wsClientRef.current = null;
    }
    isDecisionMadeRef.current = false;
    setStreamCount(0);
    setEvidenceHistory({});
    setDetectionResult(null);
    setStreamState('IDLE');
    setEvaluating(false);
    setErrorMessage(null);
  };

  const handleLoadStream = (stream: DNSObservation[], name: string) => {
    resetSession();
    setObservations(stream);
    setActiveDatasetName(name);
  };

  // Instant Batch Detection (POST /detect)
  const handleBatchDetect = async () => {
    if (observations.length < 5) {
      setErrorMessage('DeepDNS requires at least 5 observations for sequential evaluation.');
      return;
    }

    setEvaluating(true);
    setErrorMessage(null);
    try {
      const result = await detectBatch(observations, mode, `batch_${Date.now()}`);
      setDetectionResult(result);
      setEvidenceHistory(result.evidence_history);
      setStreamCount(result.stopping_horizon);
      setStreamState('DECISION_REACHED');
    } catch (err: any) {
      setErrorMessage(err.message || 'Batch detection failed.');
      setStreamState('ERROR');
    } finally {
      setEvaluating(false);
    }
  };

  // Real-time Progressive WebSocket Streaming
  const handleStartStreaming = () => {
    if (observations.length === 0) return;
    resetSession();

    const streamId = `stream_${Date.now()}`;
    setStreamState('CONNECTING');
    isDecisionMadeRef.current = false;

    const client = new DeepDNSStreamClient({
      onConnected: () => {
        setStreamState('RECEIVING');
        let currentIdx = 0;

        // Progressively send observations
        streamIntervalRef.current = window.setInterval(async () => {
          // If already decided, stop sending
          if (isDecisionMadeRef.current) {
            if (streamIntervalRef.current) {
              window.clearInterval(streamIntervalRef.current);
              streamIntervalRef.current = null;
            }
            return;
          }

          // Check if we've sent all queries
          if (currentIdx >= observations.length) {
            if (streamIntervalRef.current) {
              window.clearInterval(streamIntervalRef.current);
              streamIntervalRef.current = null;
            }

            // If all queries have been sent and no early decision was reached,
            // finalize evaluation with batch fallback so it never hangs
            if (!isDecisionMadeRef.current && observations.length >= 5) {
              try {
                setEvaluating(true);
                const finalRes = await detectBatch(observations, mode, `stream_complete_${Date.now()}`);
                isDecisionMadeRef.current = true;
                setDetectionResult(finalRes);
                setEvidenceHistory(finalRes.evidence_history);
                setStreamCount(finalRes.stopping_horizon);
                setStreamState('DECISION_REACHED');
              } catch (e: any) {
                setErrorMessage(e.message || 'Stream finalization failed.');
                setStreamState('ERROR');
              } finally {
                setEvaluating(false);
              }
            }
            return;
          }

          const obs = observations[currentIdx];
          client.sendObservation(obs);
          currentIdx += 1;
          setStreamCount(currentIdx);

          if (currentIdx % 5 === 0) {
            setEvaluating(true);
          }
        }, 300); // 300ms per query for realistic telemetry visualization
      },
      onBuffered: (count) => {
        setStreamCount(count);
      },
      onEvidenceUpdate: (data) => {
        setEvaluating(false);
        setEvidenceHistory(data.evidence_history);
      },
      onDecision: (result) => {
        isDecisionMadeRef.current = true;
        if (streamIntervalRef.current) {
          window.clearInterval(streamIntervalRef.current);
          streamIntervalRef.current = null;
        }
        setEvaluating(false);
        setDetectionResult(result);
        setEvidenceHistory(result.evidence_history);
        setStreamCount(result.stopping_horizon);
        setStreamState('DECISION_REACHED');
      },
      onError: (err) => {
        if (streamIntervalRef.current) {
          window.clearInterval(streamIntervalRef.current);
          streamIntervalRef.current = null;
        }
        setEvaluating(false);
        setErrorMessage(err);
        setStreamState('ERROR');
      },
    });

    wsClientRef.current = client;
    client.connect(streamId, mode);
  };

  return (
    <div className="app-container">
      {/* Header with Navigation Tabs & Mode Switcher */}
      <Header
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        mode={mode}
        onSelectMode={(m) => {
          setMode(m);
          resetSession();
        }}
      />

      {/* Error Alert Banner */}
      {errorMessage && (
        <div
          style={{
            background: 'rgba(255, 51, 85, 0.1)',
            border: '1px solid var(--accent-crimson)',
            borderRadius: '4px',
            padding: '12px 16px',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            color: 'var(--accent-crimson)',
            fontSize: '13px',
          }}
        >
          <AlertCircle size={18} />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Main View Switcher */}
      {activeTab === 'landing' ? (
        <LandingPage onNavigateToInstrument={() => setActiveTab('instrument')} />
      ) : (
        <>
          {/* Workflow Step-by-Step Guide */}
          <WorkflowGuide />

          {/* Multi-View Pipeline Overview */}
          <PipelineOverview mode={mode} />

          {/* Stream Controls */}
          <InputController
            onLoadStream={handleLoadStream}
            onStartStreaming={handleStartStreaming}
            onBatchDetect={handleBatchDetect}
            onReset={resetSession}
            onOpenSchemaModal={() => setIsSchemaModalOpen(true)}
            streamState={streamState}
            hasObservations={observations.length > 0}
            activeDatasetName={activeDatasetName}
          />

          {/* Interactive Custom Stream Form Builder */}
          <CustomStreamBuilder onApplyCustomStream={handleLoadStream} />

          {/* Adaptive Horizon Progress */}
          <ObservationHorizonBar
            currentCount={streamCount}
            result={detectionResult}
          />

          {/* Core Dual Analytics Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(460px, 1fr))', gap: '20px' }}>
            <EvidenceTrajectoryChart
              evidenceHistory={evidenceHistory}
              result={detectionResult}
              mode={mode}
            />
            <DecisionPanel
              result={detectionResult}
              evaluating={evaluating}
            />
          </div>

          {/* Stream Observation Console */}
          <StreamConsole
            observations={observations}
            stoppingHorizon={detectionResult ? detectionResult.stopping_horizon : null}
          />
        </>
      )}

      {/* Input Schema & Sample Download Modal */}
      <InputSchemaModal
        isOpen={isSchemaModalOpen}
        onClose={() => setIsSchemaModalOpen(false)}
      />
    </div>
  );
};
