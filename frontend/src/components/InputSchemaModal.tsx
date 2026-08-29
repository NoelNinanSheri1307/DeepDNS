import React, { useState } from 'react';
import { X, Download, Eye } from 'lucide-react';
import {
  SAMPLE_ATTACK_1,
  SAMPLE_ATTACK_2,
  SAMPLE_ATTACK_3,
  SAMPLE_BENIGN_1,
  SAMPLE_BENIGN_2,
  SAMPLE_BENIGN_3,
  SAMPLE_OOD_1,
  SAMPLE_OOD_2,
  SAMPLE_OOD_3,
} from '../data/sampleStreams';

interface InputSchemaModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const InputSchemaModal: React.FC<InputSchemaModalProps> = ({ isOpen, onClose }) => {
  const [previewSample, setPreviewSample] = useState<{ title: string; data: any } | null>(null);

  if (!isOpen) return null;

  const downloadFile = (content: string, filename: string, type: string) => {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const sampleDatasets = [
    { id: 'atk1', name: 'Attack 1: Light Text Tunnel', desc: 'Real CIC-Bell Light Text exfiltration query stream', data: SAMPLE_ATTACK_1, type: 'json', file: 'sample_attack_light_text.json' },
    { id: 'atk2', name: 'Attack 2: Heavy Audio Tunnel', desc: 'Real CIC-Bell Heavy Audio high-frequency exfiltration', data: SAMPLE_ATTACK_2, type: 'json', file: 'sample_attack_heavy_audio.json' },
    { id: 'atk3', name: 'Attack 3: Light Audio Tunnel', desc: 'Real CIC-Bell Light Audio stealth exfiltration queries', data: SAMPLE_ATTACK_3, type: 'json', file: 'sample_attack_light_audio.json' },
    { id: 'ben1', name: 'Benign 1: Normal Traffic Flow', desc: 'Real normal enterprise & web browsing queries', data: SAMPLE_BENIGN_1, type: 'json', file: 'sample_benign_normal_1.json' },
    { id: 'ben2', name: 'Benign 2: CDN Edge Traffic', desc: 'Real CDN resolver queries and standard web assets', data: SAMPLE_BENIGN_2, type: 'json', file: 'sample_benign_cdn_2.json' },
    { id: 'ben3', name: 'Benign 3: High-Volume Benign', desc: 'Real benign high-throughput corporate resolver queries', data: SAMPLE_BENIGN_3, type: 'json', file: 'sample_benign_heavy_3.json' },
    { id: 'ood1', name: 'OOD 1: Video Exfiltration (Zero-Shot)', desc: 'Held-out video exfiltration queries withheld during training', data: SAMPLE_OOD_1, type: 'json', file: 'sample_ood_video_1.json' },
    { id: 'ood2', name: 'OOD 2: Compressed Archive Exfil', desc: 'Held-out compressed file exfiltration payload streams', data: SAMPLE_OOD_2, type: 'json', file: 'sample_ood_compressed_2.json' },
    { id: 'ood3', name: 'OOD 3: Executable Binary Exfil', desc: 'Held-out binary payload tunneling queries', data: SAMPLE_OOD_3, type: 'json', file: 'sample_ood_binary_3.json' },
  ];

  const handleDownload = (sample: any) => {
    if (sample.type === 'json') {
      const payload = JSON.stringify({ mode: sample.id.startsWith('ood') ? 'ood' : 'in_distribution', stream_id: sample.file.replace('.json', ''), observations: sample.data }, null, 2);
      downloadFile(payload, sample.file, 'application/json');
    }
  };

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        background: 'rgba(0, 0, 0, 0.8)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 9999,
        padding: '20px',
      }}
    >
      <div
        className="card-panel-elevated"
        style={{
          width: '100%',
          maxWidth: '860px',
          maxHeight: '92vh',
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: '24px',
          padding: '36px',
        }}
      >
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h2 style={{ fontSize: '22px', margin: 0 }}>Input Format, Schema Contract & Benchmark Datasets</h2>
            <p style={{ fontSize: '13px', color: 'var(--text-secondary)', margin: '4px 0 0 0' }}>
              Inspect and download genuine CIC-Bell dataset sequences formatted for DeepDNS.
            </p>
          </div>
          <button
            onClick={onClose}
            style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}
          >
            <X size={22} />
          </button>
        </div>

        {/* 9 Benchmark Datasets Grid */}
        <div>
          <span className="text-label" style={{ color: 'var(--accent-blue)', display: 'block', marginBottom: '10px' }}>
            9 Pre-Formatted Benchmark Capture Samples (Click to View or Download):
          </span>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '10px' }}>
            {sampleDatasets.map((sample) => (
              <div
                key={sample.id}
                style={{
                  background: 'var(--bg-elevated)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: '3px',
                  padding: '12px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '6px',
                }}
              >
                <div style={{ fontSize: '13px', fontWeight: 600, color: sample.id.startsWith('atk') ? 'var(--accent-crimson)' : sample.id.startsWith('ben') ? 'var(--accent-cyan)' : 'var(--accent-amber)' }}>
                  {sample.name}
                </div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                  {sample.desc}
                </div>
                <div style={{ display: 'flex', gap: '6px', marginTop: '4px' }}>
                  <button
                    onClick={() => setPreviewSample({ title: sample.name, data: sample.data })}
                    className="btn-precision"
                    style={{ fontSize: '11px', padding: '4px 8px' }}
                  >
                    <Eye size={12} />
                    <span>View JSON</span>
                  </button>
                  <button
                    onClick={() => handleDownload(sample)}
                    className="btn-precision"
                    style={{ fontSize: '11px', padding: '4px 8px', color: 'var(--accent-blue)' }}
                  >
                    <Download size={12} />
                    <span>Download</span>
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* In-Browser Preview Box */}
        {previewSample && (
          <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-medium)', borderRadius: '4px', padding: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <span className="text-label" style={{ color: 'var(--accent-cyan)' }}>
                In-Browser Preview: {previewSample.title} (First 3 Observations shown)
              </span>
              <button
                onClick={() => setPreviewSample(null)}
                style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '12px' }}
              >
                Close Preview
              </button>
            </div>
            <pre style={{ margin: 0, maxHeight: '180px', overflowY: 'auto', fontSize: '11px', color: 'var(--accent-blue)', fontFamily: 'monospace' }}>
              {JSON.stringify(previewSample.data.slice(0, 3), null, 2)}
            </pre>
          </div>
        )}

        {/* Why Schema Is Structured This Way */}
        <div className="card-panel" style={{ background: 'var(--bg-primary)', padding: '18px', gap: '8px' }}>
          <span className="text-label" style={{ color: 'var(--accent-amber)' }}>
            Why is the Schema Structured This Way?
          </span>
          <div style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            DeepDNS performs <strong>causal multi-view fusion</strong>. The <code>timestamp</code> enables the causal extractor to calculate packet inter-arrival bursts (&Delta;t = log(1 + &delta;t)), while the raw <code>domain_name</code> is tokenized by the Character-CNN to detect high-entropy subdomain randomization and chunking artifacts. Numerical metrics (entropy, numeric ratio) provide instant behavioral signatures for the Temporal GRU.
          </div>
        </div>
      </div>
    </div>
  );
};
