import { useEffect, useMemo, useState } from 'react';
import { downloadAnalysisReport } from './reportDocument';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
const ACCEPTED_TYPES = '.jpg,.jpeg,.png,.gif,.webp,.bmp,.tif,.tiff,.mp4,.m4v,.mov,.avi,.mkv,.webm,.mp3,.wav,.flac,.m4a,.aac';
const SAMPLE_EXAMPLES = [
  {
    id: 'original',
    title: 'Original-style example',
    description: 'Unmodified synthetic portrait illustration.',
    filename: 'sample_original_illustration.png',
    label: 'BASE IMAGE',
    caseTitle: 'Original-style synthetic illustration review',
  },
  {
    id: 'edited',
    title: 'Synthetic deepfake-style example',
    description: 'Intentionally edited illustration; not a real deepfake.',
    filename: 'sample_edited_illustration.png',
    label: 'EDITED DEMO',
    caseTitle: 'Synthetic edited illustration review',
  },
  {
    id: 'color-shift',
    title: 'Color-transform illustration',
    description: 'Project-generated color and contrast edit; not face manipulation.',
    filename: 'sample_color-shift_illustration.png',
    label: 'SYNTHETIC TRANSFORM',
    caseTitle: 'Synthetic color-transform sample review',
  },
  {
    id: 'recompressed',
    title: 'JPEG-recompression illustration',
    description: 'Project-generated low-quality JPEG re-save; not a deepfake.',
    filename: 'sample_recompressed_illustration.png',
    label: 'SYNTHETIC TRANSFORM',
    caseTitle: 'Synthetic JPEG-recompression sample review',
  },
];

const SAMPLE_REPORT_PREVIEW = {
  case_id: 'sample-original-style',
  case_title: 'Original-style synthetic illustration (sample only)',
  investigator: 'DeepTrace AI demonstration',
  filename: 'sample_original_illustration.png',
  media_type: 'image',
  file_hash: 'a1a7187eb279ecd919ac2091a10d80a137dc964b5d8420a0b365ea7aef9f7307',
  file_size: 14667,
  created_at: null,
  verdict: 'Insufficient evidence',
  decision_status: 'not_configured',
  decision_basis: 'No trained image, video, audio, frequency, or multimodal classifier is configured.',
  summary: 'This is a synthetic illustration used to demonstrate intake. The prototype fingerprinted the bytes and reviewed basic file properties, but cannot determine whether it is original or a deepfake.',
  file_integrity: {
    extension: '.png',
    expected_mime_type: 'image/png',
    detected_mime_type: 'image/png',
    signature: 'PNG signature',
    signature_recognized: true,
    extension_matches_signature: true,
    sha256: 'a1a7187eb279ecd919ac2091a10d80a137dc964b5d8420a0b365ea7aef9f7307',
    size_bytes: 14667,
    status: 'signature_and_extension_match',
    content_validation: 'decoded',
    meaning: 'The signature check compares file-header bytes with the extension expected type. It does not prove provenance or authenticity; SHA-256 identifies these exact bytes for later comparison.',
  },
  metadata_summary: { width: 800, height: 560, mode: 'RGB', format: 'PNG', has_exif: false, pixel_bands: ['R', 'G', 'B'], band_count: 3, is_animated: false, frame_count: 1, has_transparency: false },
  metadata_findings: [{ signal: 'EXIF metadata absent', interpretation: 'Provenance fields are unavailable. Metadata absence alone does not indicate manipulation.', severity: 'context' }],
  evidence_assessment: [
    { name: 'Spatial / face analysis', status: 'not_configured', interpretation: 'No face detector or trained spatial classifier is configured.' },
    { name: 'Temporal analysis', status: 'not_applicable', interpretation: 'Temporal video/frame analysis does not apply to this still image or audio-only file.' },
    { name: 'Frequency / compression analysis', status: 'not_configured', interpretation: 'No FFT/DCT manipulation classifier or validated compression model is configured.' },
    { name: 'Audio / synchronization analysis', status: 'not_applicable', interpretation: 'Audio authenticity and synchronization analysis are not applicable to an image.' },
    { name: 'Metadata and container review', status: 'reviewed', interpretation: 'Image dimensions, pixel mode and available embedded metadata were read; these properties do not prove authenticity.' },
  ],
  pipeline_stages: [
    { number: 1, name: 'Evidence integrity', status: 'completed', summary: 'SHA-256 fingerprint generated and binary file signature checked.' },
    { number: 2, name: 'Metadata review', status: 'completed', summary: 'Image dimensions, pixel mode, frame properties, and available embedded tags were read; absent tags are not proof of manipulation.' },
    { number: 3, name: 'Signal assessment', status: 'model_not_configured', summary: 'No trained detector is installed; spatial, temporal, frequency, audio and fusion scores were not fabricated.' },
    { number: 4, name: 'Analyst interpretation', status: 'awaiting_review', summary: 'An analyst must record an independent interpretation with rationale; automated conclusion remains inconclusive.' },
  ],
  explanation: [
    'Original-versus-deepfake classification is unavailable because no trained detector is configured.',
    'A definitive authenticity or manipulation probability was not generated.',
    'The SHA-256 value identifies these exact file bytes; without a trusted reference hash it does not prove the source or originality.',
    'Missing or editable metadata is a provenance limitation, not proof of manipulation.',
    'Obtain source material and chain-of-custody records, then review with a validated detector and qualified analyst.',
  ],
};

const DATASET_PREVIEWS = [
  {
    id: 'faceforensics',
    name: 'FaceForensics++',
    description: 'Official project preview showing manipulated-media examples.',
    image: 'https://raw.githubusercontent.com/ondyari/FaceForensics/master/images/DDD_samples.gif',
    imageAlt: 'Animated preview from the official FaceForensics++ repository',
    source: 'https://github.com/ondyari/FaceForensics',
    access: 'Request dataset access',
    accessUrl: 'https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform',
    gated: true,
  },
  {
    id: 'dfdc',
    name: 'DFDC',
    description: 'Official dataset portal; example media access is provided through AWS.',
    image: '',
    imageAlt: '',
    source: 'https://ai.meta.com/datasets/dfdc/',
    access: 'Open dataset portal',
    accessUrl: 'https://ai.meta.com/datasets/dfdc/',
    gated: true,
  },
  {
    id: 'celeb-df',
    name: 'Celeb-DF v2',
    description: 'Official project preview; original and synthesized video dataset.',
    image: 'https://raw.githubusercontent.com/yuezunli/celeb-deepfakeforensics/master/Celeb-DF-v2/demo.png',
    imageAlt: 'Official Celeb-DF v2 project demonstration image',
    source: 'https://github.com/yuezunli/celeb-deepfakeforensics',
    access: 'Request dataset access',
    accessUrl: 'https://forms.gle/2jYBby6y1FBU3u6q9',
    gated: true,
  },
];

const NAV_ITEMS = [
  { id: 'overview', label: 'Dashboard', icon: 'grid', view: 'overview' },
  { id: 'image-analysis', label: 'Image analysis', icon: 'image', view: 'analysis', mode: 'image' },
  { id: 'video-analysis', label: 'Video analysis', icon: 'video', view: 'analysis', mode: 'video' },
  { id: 'audio-analysis', label: 'Audio analysis', icon: 'audio', view: 'analysis', mode: 'audio' },
  { id: 'metadata', label: 'Metadata inspector', icon: 'file', view: 'analysis' },
  { id: 'model-comparison', label: 'Model comparison', icon: 'models', view: 'cases' },
  { id: 'reports', label: 'Forensic reports', icon: 'file', view: 'reports' },
  { id: 'cases', label: 'Case history', icon: 'folder', view: 'cases' },
  { id: 'settings', label: 'Settings', icon: 'settings', view: 'overview' },
];

const formatDate = (value) => {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '—' : date.toLocaleString();
};

const formatBytes = (bytes) => {
  if (!Number.isFinite(bytes)) return '—';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const formatMetadataValue = (value, key = '') => {
  if (Array.isArray(value)) return value.join(', ');
  if (value && typeof value === 'object') return JSON.stringify(value);
  if (typeof value === 'string' && ['status', 'content_validation'].includes(key)) {
    return value.replaceAll('_', ' ');
  }
  return String(value);
};

const formatProbability = (value) => (
  Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : 'Not computed'
);

const formatMetric = (value) => (
  Number.isFinite(value) ? value.toFixed(3) : 'n/a'
);

const formatBenchmarkMetrics = (metrics) => metrics
  ? `n=${metrics.samples ?? 'n/a'} · ROC-AUC ${formatMetric(metrics.roc_auc)} · PR-AUC/AP ${formatMetric(metrics.pr_auc)} · balanced accuracy ${formatMetric(metrics.balanced_accuracy_at_0_5)} · F1 ${formatMetric(metrics.f1_at_0_5)} · Brier ${formatMetric(metrics.brier_score)} · ECE ${formatMetric(metrics.expected_calibration_error)}`
  : 'No independent test metrics recorded';

const formatCalibrationMetrics = (before, after) => (
  before && after
    ? `Before: Brier ${formatMetric(before.brier_score)}, ECE ${formatMetric(before.expected_calibration_error)} · After: Brier ${formatMetric(after.brier_score)}, ECE ${formatMetric(after.expected_calibration_error)}`
    : 'No held-out calibration diagnostics recorded'
);

const describeRequestError = (error, fallback) => {
  if (error instanceof TypeError && error.message === 'Failed to fetch') {
    return `Cannot reach the analysis API at ${API_BASE}. Start or restart the backend and confirm its CORS settings allow this dashboard.`;
  }
  return error instanceof Error ? error.message : fallback;
};

function Icon({ name, size = 18 }) {
  const paths = {
    grid: <><rect x="3" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="3" width="7" height="7" rx="1.5" /><rect x="3" y="14" width="7" height="7" rx="1.5" /><rect x="14" y="14" width="7" height="7" rx="1.5" /></>,
    scan: <><path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3" /><circle cx="12" cy="12" r="3" /><path d="M12 2v2m0 16v2M2 12h2m16 0h2" /></>,
    folder: <><path d="M3 7a2 2 0 0 1 2-2h5l2 2h7a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" /><path d="M3 10h18" /></>,
    file: <><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><path d="M14 2v6h6M8 13h8m-8 4h8" /></>,
    chevron: <path d="m9 18 6-6-6-6" />,
    upload: <><path d="M12 16V4m-5 5 5-5 5 5" /><path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
    shield: <><path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11z" /><path d="m9 12 2 2 4-4" /></>,
    arrow: <><path d="M7 17 17 7M7 7h10v10" /></>,
    download: <><path d="M12 3v12m-5-5 5 5 5-5" /><path d="M5 17v4h14v-4" /></>,
    copy: <><rect x="8" y="8" width="12" height="12" rx="2" /><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3" /></>,
    image: <><rect x="3" y="3" width="18" height="18" rx="3" /><circle cx="8.5" cy="8.5" r="1.5" /><path d="m21 15-5-5L5 21" /></>,
    video: <><rect x="3" y="5" width="14" height="14" rx="2" /><path d="m17 10 4-3v10l-4-3z" /><path d="m9 9 4 3-4 3z" /></>,
    audio: <><path d="M9 18V5l12-2v13" /><circle cx="6" cy="18" r="3" /><circle cx="18" cy="16" r="3" /></>,
    models: <><circle cx="12" cy="5" r="2" /><circle cx="5" cy="19" r="2" /><circle cx="19" cy="19" r="2" /><path d="M12 7v5m0 0-7 5m7-5 7 5" /></>,
    settings: <><circle cx="12" cy="12" r="3" /><path d="m19.4 15 .1.1a1.7 1.7 0 0 1-2.4 2.4L17 17.4a1.7 1.7 0 0 0-2.9 1.2v.2a1.7 1.7 0 0 1-3.4 0v-.2a1.7 1.7 0 0 0-2.9-1.2l-.1.1a1.7 1.7 0 0 1-2.4-2.4l.1-.1a1.7 1.7 0 0 0-1.2-2.9H4a1.7 1.7 0 0 1 0-3.4h.2a1.7 1.7 0 0 0 1.2-2.9l-.1-.1a1.7 1.7 0 0 1 2.4-2.4l.1.1a1.7 1.7 0 0 0 2.9-1.2V2a1.7 1.7 0 0 1 3.4 0v.2a1.7 1.7 0 0 0 2.9 1.2l.1-.1a1.7 1.7 0 0 1 2.4 2.4l-.1.1a1.7 1.7 0 0 0 1.2 2.9h.2a1.7 1.7 0 0 1 0 3.4h-.2a1.7 1.7 0 0 0-1.2 2.9z" /></>,
    search: <><circle cx="11" cy="11" r="7" /><path d="m20 20-4-4" /></>,
    bell: <><path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9" /><path d="M10 21h4" /></>,
  };

  return (
    <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      {paths[name] || paths.grid}
    </svg>
  );
}

function ReportPreview({ report, onClose, onDownload, downloading }) {
  const pipelineStages = report.pipeline_stages || [];
  const modelPipeline = report.model_pipeline || [];
  const xception = report.model_results?.xception;
  const chainOfCustody = report.chain_of_custody;
  const uncalibrated = xception?.status === 'uncalibrated_prediction';
  const integrity = Object.entries(report.file_integrity || {}).filter(([key, value]) => (
    !['sha256', 'size_bytes', 'header_hex', 'meaning'].includes(key)
    && value !== null && value !== undefined && value !== ''
  ));
  const metadata = Object.entries(report.metadata_summary || {}).filter(([, value]) => (
    value !== null && value !== undefined && value !== '' && value !== 'unknown'
  ));

  return (
    <div className="report-preview-backdrop" role="presentation" onClick={onClose}>
      <section className="report-preview-dialog" role="dialog" aria-modal="true" aria-labelledby="report-preview-title" onClick={(event) => event.stopPropagation()}>
        <header className="report-preview-toolbar">
          <div><span className="section-kicker">DOCUMENT PREVIEW</span><strong>Forensic Analysis Report</strong></div>
          <button className="icon-button" onClick={onClose} aria-label="Close report preview">×</button>
        </header>
        <div className="report-preview-scroll">
          <article className="report-preview-page">
            <div className="preview-brand">DEEPTRACE AI <span>/ MEDIA FORENSICS</span></div>
            <h1 id="report-preview-title">Forensic Analysis Report</h1>
            <p className="preview-subtitle">Evidence-based decision support · Research prototype</p>
            <dl className="preview-case-grid">
              <div><dt>Case ID</dt><dd>{report.case_id || 'Not available'}</dd></div>
              <div><dt>Analysis date</dt><dd>{formatDate(report.created_at)}</dd></div>
              <div><dt>Case title</dt><dd>{report.case_title || 'Not available'}</dd></div>
              <div><dt>Investigator</dt><dd>{report.investigator || 'Not available'}</dd></div>
              <div><dt>Evidence file</dt><dd>{report.filename || 'Not available'}</dd></div>
              <div><dt>Media type</dt><dd>{report.media_type || 'Not available'}</dd></div>
            </dl>
            <section className="preview-conclusion">
              <span>{report.verdict || 'Insufficient evidence'}</span>
              <strong>{xception
                ? `Automated assessment: ${report.verdict}`
                : report.evidence_context?.is_synthetic_demo
                  ? 'Synthetic demo sample — classifier intentionally skipped'
                  : 'Verdict withheld — no trained classifier configured'}</strong>
              <p>{report.decision_basis || 'A trained and validated classifier is not configured.'}</p>
            </section>
            <h2>Executive summary</h2>
            <p>{report.summary || 'No summary is available.'}</p>
            {report.evidence_context?.is_synthetic_demo && <p className="preview-caveat"><strong>Synthetic demonstration sample — not ground truth:</strong> {report.evidence_context.notice}</p>}
            <p><strong>Model status:</strong> {String(report.decision_status || 'unknown').replaceAll('_', ' ')}</p>
            <p><strong>{uncalibrated ? 'Authenticity softmax score' : 'Authenticity probability'}:</strong> {formatProbability(report.authenticity_probability)} · <strong>{uncalibrated ? 'Deepfake softmax score' : 'Deepfake probability'}:</strong> {formatProbability(report.deepfake_probability)} · <strong>Model confidence:</strong> {xception ? 'Not separately calibrated' : 'Not computed — inference did not run'}</p>
            <h2>Descriptive image measurements</h2>
            {report.frequency_analysis?.measurements || report.noise_analysis?.measurements || report.face_review
              ? <dl className="preview-property-grid">
                {report.face_review && <div><dt>Face-candidate review</dt><dd>{report.face_review.status === 'unavailable' ? 'Detector unavailable; no candidate count.' : `${report.face_review.detected_face_count} candidates · ${String(report.face_review.status).replaceAll('_', ' ')}.`} Not a manipulation score.</dd></div>}
                {Object.entries(report.frequency_analysis?.measurements || {}).map(([key, value]) => <div key={`fft-${key}`}><dt>FFT · {key.replaceAll('_', ' ')}</dt><dd>{typeof value === 'number' ? value.toFixed(4) : value}</dd></div>)}
                {Object.entries(report.noise_analysis?.measurements || {}).map(([key, value]) => <div key={`noise-${key}`}><dt>Residual · {key.replaceAll('_', ' ')}</dt><dd>{typeof value === 'number' ? value.toFixed(4) : value}</dd></div>)}
              </dl>
              : <p>No descriptive image measurements were recorded for this media type.</p>}
            <p className="preview-caveat">Face candidates, spectrum values, and high-pass residuals describe decoded image properties; they are not validated deepfake indicators and must not be interpreted as authenticity scores.</p>
            {xception?.explainability?.status === 'generated' && <><h2>Model Attention / Evidence Visualization</h2><img className="preview-heatmap-image" src={xception.explainability.overlay_data_url} alt="Model attention visualization showing regions that contributed to the fake-class prediction" /><p>Grad-CAM tells us <strong>which regions contributed most to the model's prediction</strong>. It does <strong>not</strong> independently establish that those pixels were manipulated.</p></>}
            {xception && <><h2>Xception model assessment</h2><dl className="preview-property-grid">
              <div><dt>Architecture</dt><dd>{xception.model.architecture}</dd></div>
              {xception.model.inference_device && <div><dt>Inference device</dt><dd>{xception.model.inference_device}</dd></div>}
              <div><dt>Validation</dt><dd>{xception.model.reported_validation_accuracy || (Number.isFinite(xception.model.validation_auc) ? `${Number(xception.model.validation_auc).toFixed(4)} AUC · validation ranking metric, not per-file confidence` : 'No independently verified validation result')}</dd></div>
              <div><dt>Calibration</dt><dd>{xception.model.calibration}{Number.isFinite(xception.model.calibration_temperature) ? ` · temperature ${Number(xception.model.calibration_temperature).toFixed(4)}` : ''}</dd></div>
              <div><dt>Operating thresholds</dt><dd>{JSON.stringify(xception.model.thresholds)}</dd></div>
              {xception.model.input_preprocessing && <div><dt>Model input</dt><dd>{xception.model.input_preprocessing}</dd></div>}
              {xception.model.training_data && <div><dt>Training-data scope</dt><dd>{xception.model.training_data}</dd></div>}
              {xception.model.calibration_metrics_after_scaling && <div><dt>Calibration diagnostics</dt><dd>{formatCalibrationMetrics(xception.model.calibration_metrics_before_scaling, xception.model.calibration_metrics_after_scaling)}</dd></div>}
              {xception.conformal_prediction && <div><dt>Conformal prediction set</dt><dd>{xception.conformal_prediction.prediction_set.length ? xception.conformal_prediction.prediction_set.join(', ') : 'Empty'} · α={xception.conformal_prediction.alpha} · {xception.conformal_prediction.abstained ? 'abstained' : 'threshold-selected class is supported'}</dd></div>}
              {xception.model.test_metrics && <div><dt>Independent test benchmark</dt><dd>{formatBenchmarkMetrics(xception.model.test_metrics)} · Dataset-level metrics, not this file's confidence.</dd></div>}
              {xception.model.test_metrics_by_source && Object.entries(xception.model.test_metrics_by_source).map(([source, metrics]) => <div key={`test-source-${source}`}><dt>Test source · {source}</dt><dd>{formatBenchmarkMetrics(metrics)} · {metrics.source_not_seen_in_train_validation_or_calibration ? 'Source held out from model development.' : 'Source also appears in model-development splits.'}</dd></div>)}
              {xception.model.dataset_audit?.group_overlap_status && <div><dt>Group-leakage audit</dt><dd>{String(xception.model.dataset_audit.group_overlap_status).replaceAll('_', ' ')} · {xception.model.dataset_audit.manifest_sha256 || 'No audit digest'}</dd></div>}
              <div><dt>Face review</dt><dd>{JSON.stringify(xception.face_review)}</dd></div>
            </dl><ul>{xception.limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}</ul></>}
            <h2>Classifier pipeline execution trace</h2>
            {modelPipeline.length
              ? <ol className="preview-stage-list">{modelPipeline.map((stage) => <li key={stage.number}><strong>{stage.number}. {stage.name}</strong><span>{String(stage.status).replaceAll('_', ' ')}</span><p>{stage.detail}</p></li>)}</ol>
              : <p>This stored report predates detailed classifier pipeline tracing. Reanalyze the retained evidence to record current per-stage status.</p>}
            <h2>Four-stage evidence review</h2>
            <ol className="preview-stage-list">
              {pipelineStages.map((stage) => <li key={stage.number}><strong>{stage.number}. {stage.name}</strong><span>{String(stage.status).replaceAll('_', ' ')}</span><p>{stage.summary}</p></li>)}
            </ol>
            <h2>Evidence integrity — SHA-256 fingerprint</h2>
            <p className="preview-hash">{report.file_hash || 'Not available'}</p>
            <p><strong>Audit chain:</strong> {chainOfCustody?.verification_status || 'not sealed'} · head {chainOfCustody?.head_sha256 || 'not available'}</p>
            {chainOfCustody?.storage_limitation && <p className="preview-caveat">{chainOfCustody.storage_limitation}</p>}
            <dl className="preview-property-grid">
              {integrity.map(([key, value]) => <div key={key}><dt>{key.replaceAll('_', ' ')}</dt><dd>{formatMetadataValue(value, key)}</dd></div>)}
            </dl>
            <p className="preview-caveat">A hash identifies these bytes for later comparison. It does not prove provenance or originality without a trusted reference.</p>
            <h2>Metadata review — container and file properties</h2>
            <dl className="preview-property-grid">
              {metadata.map(([key, value]) => <div key={key}><dt>{key.replaceAll('_', ' ')}</dt><dd>{formatMetadataValue(value, key)}</dd></div>)}
            </dl>
            {(report.metadata_findings || []).map((finding) => <div className="preview-finding" key={finding.signal}><strong>{finding.signal}</strong><p>{finding.interpretation}</p></div>)}
            <p className="preview-caveat">Metadata can be missing or edited. Its absence does not prove manipulation.</p>
            <h2>Signal assessment</h2>
            <div className="preview-assessments">
              {(report.evidence_assessment || []).map((item) => <div key={item.name}><span>{String(item.status).replaceAll('_', ' ')}</span><strong>{item.name}</strong><p>{item.interpretation}</p></div>)}
            </div>
            <h2>Analyst interpretation</h2>
            {report.analyst_review
              ? <div className="preview-finding"><strong>{String(report.analyst_review.outcome).replaceAll('_', ' ')} — {report.analyst_review.reviewer}</strong><p>{report.analyst_review.rationale}</p><small>{formatDate(report.analyst_review.reviewed_at)} · Human assessment, not an automated model output</small></div>
              : <p>No reviewer conclusion has been recorded. Review trusted source material and chain-of-custody context before recording an opinion.</p>}
            <h2>Interpretation and limitations</h2>
            <p className="preview-caveat">{xception
              ? uncalibrated
                ? 'The classifier provides raw, uncalibrated softmax scores and a higher-scoring class, not a calibrated probability or proof. Its training scope is FFHQ/StyleGAN faces; performance may change under domain shift. Corroborate independently and have a qualified analyst review the original media.'
                : xception.status === 'conformal_abstention'
                  ? 'The calibrated model withheld its class because the conformal prediction set did not uniquely support the threshold-selected label. Coverage assumes exchangeability with calibration data and is not guaranteed under domain shift. The score is not proof or a per-file confidence guarantee.'
                  : 'The classifier provides a calibrated model estimate, not proof or a per-file confidence guarantee. Performance may change under dataset/domain shift. Corroborate independently and have a qualified analyst review the original media.'
              : 'This prototype has no trained authenticity classifier and withholds an original-versus-deepfake verdict. Do not treat this report as proof of authenticity or manipulation. Corroborate evidence independently and have a qualified analyst review the original media.'}</p>
            {(report.explanation || []).length > 0 && <><h2>Analysis notes</h2>{report.explanation.map((note, index) => <p key={`note-${index}`}>{note}</p>)}</>}
            {(report.audit_log || []).length > 0 && <><h2>Audit trail</h2><div className="preview-assessments">{report.audit_log.map((entry, index) => <div key={`audit-${index}`}><strong>{formatDate(entry.timestamp)} · {entry.event_type || 'legacy event'}</strong><p>{entry.message}</p>{entry.event_sha256 && <small>Event SHA-256: {entry.event_sha256}</small>}</div>)}</div></>}
          </article>
        </div>
        <footer className="report-preview-actions">
          <button className="button button-secondary" onClick={onClose}>Close preview</button>
          <button className="button button-primary" disabled={downloading} onClick={() => onDownload(report)}>{downloading ? 'Creating Word…' : 'Download Word .docx'}</button>
        </footer>
      </section>
    </div>
  );
}

function TwoImageComparison() {
  const [imageA, setImageA] = useState(null);
  const [imageB, setImageB] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [previewA, setPreviewA] = useState('');
  const [previewB, setPreviewB] = useState('');

  useEffect(() => {
    if (!imageA) {
      setPreviewA('');
      return undefined;
    }
    const url = URL.createObjectURL(imageA);
    setPreviewA(url);
    return () => URL.revokeObjectURL(url);
  }, [imageA]);

  useEffect(() => {
    if (!imageB) {
      setPreviewB('');
      return undefined;
    }
    const url = URL.createObjectURL(imageB);
    setPreviewB(url);
    return () => URL.revokeObjectURL(url);
  }, [imageB]);

  const runComparison = async (event) => {
    event.preventDefault();
    if (!imageA || !imageB) {
      setError('Choose both images before running the comparison.');
      return;
    }
    setLoading(true);
    setError('');
    setComparison(null);
    const formData = new FormData();
    formData.append('image_a', imageA);
    formData.append('image_b', imageB);
    try {
      const response = await fetch(`${API_BASE}/api/compare-images`, { method: 'POST', body: formData });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `Comparison failed with status ${response.status}.`);
      setComparison(payload);
    } catch (requestError) {
      setError(describeRequestError(requestError, 'Could not compare these images.'));
    } finally {
      setLoading(false);
    }
  };

  const downloadComparison = () => {
    if (!comparison) return;
    const blob = new Blob([JSON.stringify(comparison, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${comparison.comparison_id || 'deeptrace-image-comparison'}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const imageCard = (key, title, file, preview) => {
    const item = comparison?.images?.[key];
    const face = item?.face_observation;
    const model = item?.model_assessment;
    return (
      <article className="comparison-image-card" key={key}>
        <div className="comparison-image-heading"><strong>{title}</strong><small>{file?.name || item?.filename || 'No image selected'}</small></div>
        {preview && <img className="comparison-preview" src={preview} alt={`${title} preview`} />}
        {item && (
          <dl className="comparison-facts">
            <div><dt>Model assessment</dt><dd>{model.verdict} · {String(model.decision_status).replaceAll('_', ' ')}</dd></div>
            <div><dt>Deepfake score</dt><dd>{formatProbability(model.deepfake_probability)}</dd></div>
            <div><dt>Image size</dt><dd>{item.width && item.height ? `${item.width} × ${item.height}` : 'Unavailable'}</dd></div>
            <div><dt>Face candidates</dt><dd>{face.candidate_count ?? 'Unavailable'} · {String(face.status).replaceAll('_', ' ')}</dd></div>
            {face.candidate_boxes_xyxy?.length > 0 && <div><dt>Face-candidate boxes</dt><dd>{face.candidate_boxes_xyxy.map((box) => `[${box.join(', ')}]`).join('; ')} (x1, y1, x2, y2)</dd></div>}
            <div><dt>EXIF</dt><dd>{item.exif_summary.has_exif ? 'Present' : 'Not present'} · {item.exif_summary.parse_status}</dd></div>
            {Object.entries(item.exif_summary.fields || {}).map(([name, value]) => (
              <div key={`${key}-${name}`}><dt>{name.replaceAll('_', ' ')}</dt><dd>{value}</dd></div>
            ))}
            {item.exif_summary.gps_coordinates && (
              <div><dt>EXIF GPS</dt><dd>{item.exif_summary.gps_coordinates.latitude}, {item.exif_summary.gps_coordinates.longitude}</dd></div>
            )}
          </dl>
        )}
      </article>
    );
  };

  return (
    <section className="forensic-card comparison-card">
      <div className="forensic-card-heading">
        <div><span className="section-kicker">PAIRWISE IMAGE REVIEW</span><h3>Compare two images</h3></div>
        <span className="evidence-pill muted-pill">Qualified assessment only</span>
      </div>
      <p className="forensic-caveat">Compare file fingerprints, available EXIF, face-candidate observations, pixel differences, and each image’s independent model assessment. A pair cannot prove which image is the original.</p>
      <form className="comparison-form" onSubmit={runComparison}>
        <label>Image A<input type="file" accept="image/jpeg,image/png,image/gif,image/webp,image/bmp,image/tiff,.jpg,.jpeg,.png,.gif,.webp,.bmp,.tif,.tiff" onChange={(event) => { setImageA(event.target.files[0] || null); setComparison(null); setError(''); }} /></label>
        <label>Image B<input type="file" accept="image/jpeg,image/png,image/gif,image/webp,image/bmp,image/tiff,.jpg,.jpeg,.png,.gif,.webp,.bmp,.tif,.tiff" onChange={(event) => { setImageB(event.target.files[0] || null); setComparison(null); setError(''); }} /></label>
        <button type="submit" className="button button-primary" disabled={loading || !imageA || !imageB}>{loading ? 'Comparing…' : 'Compare images'}</button>
      </form>
      {error && <div className="error-box" role="alert">{error}</div>}
      {comparison && (
        <div className="comparison-report" aria-live="polite">
          <div className={`comparison-finding ${comparison.status}`}>
            <div><span className="section-kicker">COMPARISON REPORT · {String(comparison.status).replaceAll('_', ' ')}</span><h4>{comparison.finding}</h4></div>
            <button type="button" className="button button-secondary" onClick={downloadComparison}>Download JSON report</button>
          </div>
          <div className="comparison-images">
            {imageCard('image_a', 'Image A', imageA, previewA)}
            {imageCard('image_b', 'Image B', imageB, previewB)}
          </div>
          <section className="comparison-pixel-review">
            <strong>Direct pixel comparison</strong>
            {comparison.pixel_difference.status === 'measured'
              ? <span>{comparison.pixel_difference.mean_absolute_difference_8bit} mean absolute difference · {(comparison.pixel_difference.changed_pixel_ratio * 100).toFixed(2)}% differing pixels{comparison.pixel_difference.resized_for_bounds ? ' · measured after bounded downscaling' : ''}</span>
              : <span>{comparison.pixel_difference.reason}</span>}
          </section>
          <p className="forensic-caveat">{comparison.limitations.join(' ')}</p>
          <p className="forensic-caveat">Face-candidate counts and boxes are non-identifying detections only; this report does not name or identify people or infer personal attributes.</p>
        </div>
      )}
    </section>
  );
}

const RESULT_TABS = [
  'Result',
  'Face Analysis',
  'Heatmap',
  'Frequency',
  'Noise Analysis',
  'Metadata',
  'Comparisons',
  'Timeline',
  'Evidence Graph',
  'Model Details',
];

const availableValue = (value, fallback = 'Not available') => (
  value === null || value === undefined || value === '' ? fallback : String(value)
);

function AnalysisResultDashboard({
  report,
  previewUrl,
  activeTab,
  setActiveTab,
  onCopyHash,
  onPreviewReport,
  onExportReport,
  onReanalyze,
  reanalyzing,
  exporting,
  children,
}) {
  const xception = report.model_results?.xception;
  const explainability = xception?.explainability;
  const explainabilityUnavailableReason = explainability?.reason
    || (xception
      ? 'This saved report has no model-attention visualization. Reanalyze the retained image to create one.'
      : 'Grad-CAM requires a compatible image classifier. If inference did not run, no heatmap is fabricated.');
  const uncalibrated = xception?.status === 'uncalibrated_prediction';
  const metadata = report.metadata_summary || {};
  const mediaTypeLabel = report.media_type
    ? `${report.media_type.charAt(0).toUpperCase()}${report.media_type.slice(1)} Analysis`
    : 'Media Analysis';
  const faceReview = report.face_review || xception?.face_review;
  const frequencyMeasurements = report.frequency_analysis?.measurements;
  const noiseMeasurements = report.noise_analysis?.measurements;
  const videoFrames = report.video_analysis?.frames || [];
  const videoFrameAnalysis = report.video_analysis?.model_frame_analysis;
  const hasVideoFrameEstimates = Boolean(videoFrameAnalysis?.analyzed_frame_count);
  const chainOfCustody = report.chain_of_custody || {};
  const firstVideoFrame = videoFrames[0];
  const audioMeasurements = report.audio_analysis?.measurements;
  const sampledVideoFaceCount = videoFrames.reduce(
    (total, frame) => total + (frame.face_review?.status === 'unavailable' ? 0 : frame.face_review?.detected_face_count || 0),
    0,
  );
  const videoFaceDetectorAvailable = videoFrames.some((frame) => frame.face_review?.status !== 'unavailable');
  const resultTabs = report.media_type === 'video'
    ? [...RESULT_TABS, 'Video Frames']
    : report.media_type === 'audio'
      ? [...RESULT_TABS, 'Audio Signal']
      : RESULT_TABS;
  const modelPipeline = report.model_pipeline || [
    { number: 1, name: 'File validation and decode', status: 'completed', detail: 'File signature and decoder inspection are recorded in evidence integrity.' },
    { number: 2, name: 'Evidence fingerprint', status: 'completed', detail: 'SHA-256 was calculated for the uploaded bytes.' },
    { number: 3, name: 'Metadata and container review', status: 'completed', detail: 'Available image, video, or audio properties are recorded in Metadata.' },
    { number: 4, name: 'Descriptive signal measurements', status: frequencyMeasurements || noiseMeasurements || report.video_analysis || audioMeasurements ? 'measured_descriptive' : 'not_applicable', detail: 'Available measurements describe pixels or decoded signal only; they are not a trained authenticity score.' },
    { number: 5, name: 'Face-candidate detection', status: faceReview ? (['face_detected', 'no_face_detected'].includes(faceReview.status) ? 'measured_descriptive' : 'unavailable') : report.media_type === 'audio' ? 'not_applicable' : 'not_run', detail: faceReview?.interpretation || (report.media_type === 'audio' ? 'Face detection does not apply to audio.' : 'No face-candidate observation was recorded in this report.') },
    { number: 6, name: 'Xception checkpoint', status: xception ? 'loaded' : report.decision_status === 'demo_sample_not_classified' ? 'skipped_demo' : report.media_type === 'image' ? 'not_configured' : 'not_applicable', detail: xception ? `${xception.model.architecture} checkpoint loaded.` : report.decision_status === 'demo_sample_not_classified' ? 'Classifier inference was intentionally skipped for this synthetic UI example.' : report.media_type === 'image' ? 'No compatible trained Xception checkpoint was loaded; expected at backend/weights/xception_deepfake.pth.' : 'No trained classifier for this media type is configured.' },
    { number: 7, name: 'Model-specific preprocessing', status: xception ? 'completed' : 'not_run', detail: xception ? (xception.model.input_preprocessing || 'The Xception inference path prepared its image tensor.') : 'Not run because no Xception inference occurred.' },
    { number: 8, name: 'Xception inference', status: xception ? 'completed' : 'not_run', detail: xception ? (uncalibrated ? 'Raw two-class softmax scores were produced; they are not calibrated probabilities.' : 'The model produced an image-level probability.') : 'No model logits or probabilities were generated.' },
    { number: 9, name: 'Calibration and thresholds', status: xception ? (uncalibrated ? 'not_calibrated' : report.decision_status === 'classified' ? 'applied' : 'probability_only') : 'not_run', detail: xception ? report.decision_basis : 'No model output was available to calibrate or threshold.' },
    { number: 10, name: 'Original / deepfake decision', status: report.decision_status === 'uncalibrated_prediction' ? 'uncalibrated_prediction' : report.decision_status === 'classified' ? 'classified' : 'withheld', detail: report.decision_basis || 'No class was assigned because a trained classifier did not produce a decision.' },
    { number: 11, name: 'Grad-CAM explainability', status: explainability?.status === 'generated' ? 'completed' : explainability?.status || 'not_configured', detail: explainability?.interpretation || explainability?.reason || 'No compatible image-model explanation pass was generated.' },
    { number: 12, name: 'Video temporal classifier', status: report.media_type === 'video' ? 'not_configured' : 'not_applicable', detail: report.media_type === 'video' ? 'Keyframe previews are descriptive; no temporal classifier is configured.' : 'Not a video input.' },
    { number: 13, name: 'Synthetic-voice classifier', status: report.media_type === 'audio' ? 'not_configured' : 'not_applicable', detail: report.media_type === 'audio' ? 'Waveform measurements are descriptive; no voice-authenticity classifier is configured.' : 'Not an audio-only input.' },
    { number: 14, name: 'Multimodal fusion', status: 'not_configured', detail: 'No trained and validated fusion model is installed.' },
  ];
  const pipelineStatusLabel = (status) => ({
    completed: 'Completed',
    loaded: 'Loaded',
    loaded_for_frame_analysis: 'Loaded for sampled frames',
    generated: 'Generated',
    applied: 'Applied',
    classified: 'Classified',
    uncalibrated_prediction: 'Uncalibrated model class',
    not_calibrated: 'Not calibrated',
    not_validated_for_video: 'Not validated for video',
    frame_estimates_available: 'Frame estimates available',
    sampled_frames_only: 'Sampled frames only',
    uncalibrated_model_estimate: 'Uncalibrated model estimate',
    measured_descriptive: 'Measured · descriptive',
    probability_only: 'Probability only',
    withheld: 'Verdict withheld',
    conformal_abstention: 'Abstained · conformal set',
    skipped_demo: 'Skipped · demo sample',
    not_configured: 'Not configured',
    not_run: 'Not run',
    not_applicable: 'Not applicable',
    unavailable: 'Unavailable',
    partial: 'Partial',
    review_required: 'Review required',
  }[status] || String(status).replaceAll('_', ' '));
  const modelPipelineCaveat = xception
    ? xception.status === 'conformal_abstention'
      ? 'The locally calibrated Xception model withheld its class because the conformal prediction set did not uniquely support the threshold-selected class. Its coverage interpretation assumes exchangeability with the calibration data and is not guaranteed under domain shift.'
      : uncalibrated
      ? 'The pinned Xception checkpoint ran on the full resized image. Its raw softmax scores and higher-scoring class are uncalibrated, trained on FFHQ/StyleGAN faces, and are not proof of authenticity. The other modalities have no trained detector.'
      : 'The Xception checkpoint ran for this image. The other modalities shown as not configured have no trained detector. Calibrated model estimates remain decision support, not proof.'
    : report.evidence_context?.is_synthetic_demo
      ? 'This bundled synthetic illustration is deliberately excluded from classifier testing. Its measured file/signal observations are for interface demonstration only and are not ground truth.'
    : hasVideoFrameEstimates
      ? 'The image Xception model ran on a bounded sample of decoded frames. Frame scores are not validated for video-level interpretation; no video verdict or temporal manipulation result was produced.'
    : report.media_type === 'image' && !metadata.is_animated
        ? 'No compatible trained and calibrated Xception checkpoint is installed in backend/weights/. No original/deepfake probability or model class can be computed until suitable model weights and validation data are supplied.'
        : `No trained ${report.media_type} authenticity classifier is configured. The report withholds an original/deepfake class and probability.`;
  const pipelineStepClass = (status) => (
    ['completed', 'loaded', 'loaded_for_frame_analysis', 'applied', 'classified', 'generated', 'measured_descriptive', 'frame_estimates_available', 'sampled_frames_only'].includes(status)
      ? 'completed'
      : ['review_required', 'partial', 'unavailable', 'withheld', 'conformal_abstention', 'not_validated_for_video'].includes(status)
        ? 'review'
        : 'not_run'
  );
  const evidenceUrl = previewUrl || (
    report.media_type === 'image' && report.case_id && report.evidence_preview_token
      ? `${API_BASE}/api/reports/${encodeURIComponent(report.case_id)}/evidence?token=${encodeURIComponent(report.evidence_preview_token)}`
      : ''
  );
  const [evidenceUnavailable, setEvidenceUnavailable] = useState(false);
  useEffect(() => setEvidenceUnavailable(false), [evidenceUrl]);
  const tabsNotRun = {
    Heatmap: explainabilityUnavailableReason,
  };
  const probability = Number.isFinite(report.deepfake_probability)
    ? report.deepfake_probability
    : null;
  const assessmentRows = (report.evidence_assessment || []).map((item) => ({
    ...item,
    probability: item.name === 'Spatial / face analysis' ? probability : null,
  }));
  const metadataFindings = report.metadata_findings || [];
  const pipeline = [
    { label: 'Upload', status: 'completed' },
    { label: 'Validate', status: report.file_integrity?.content_validation === 'decoded' ? 'completed' : 'review' },
    { label: 'Fingerprint', status: report.file_hash ? 'completed' : 'review' },
    { label: 'Metadata', status: 'completed' },
    { label: 'Face candidates', status: faceReview?.status === 'face_detected' || faceReview?.status === 'no_face_detected' || videoFrames.some((frame) => ['face_detected', 'no_face_detected'].includes(frame.face_review?.status)) ? 'completed' : faceReview || videoFrames.length ? 'review' : report.media_type === 'audio' ? 'not_applicable' : 'not_run' },
    { label: 'Xception', status: xception || hasVideoFrameEstimates ? 'completed' : 'not_run' },
    { label: 'Decision', status: report.decision_status === 'classified' ? 'completed' : 'review' },
    { label: 'Grad-CAM', status: explainability?.status === 'generated' ? 'completed' : 'not_run' },
  ];
  const graphNodes = [
    ['Uploaded file', report.filename, 'completed'],
    ['SHA-256 fingerprint', report.file_hash, 'completed'],
    ['Metadata review', metadata.format
      ? `${metadata.format}${metadata.width && metadata.height ? ` · ${metadata.width} × ${metadata.height}` : ''}`
      : `${report.media_type} properties reviewed`, 'completed'],
    ['Face-candidate review', faceReview
      ? faceReview.status === 'unavailable'
        ? 'Detector unavailable; no candidate count'
        : `${faceReview.detected_face_count} candidates · ${faceReview.status.replaceAll('_', ' ')}`
      : 'Not run for this media type',
    faceReview && faceReview.status !== 'unavailable' ? 'completed' : 'not_run'],
    ['Frequency / residual measurements', frequencyMeasurements && noiseMeasurements
      ? 'Descriptive FFT and high-pass residual recorded'
      : 'Not available for this media type', frequencyMeasurements ? 'completed' : 'not_run'],
    ['Xception inference', xception
      ? `${formatProbability(report.deepfake_probability)} model estimate`
      : hasVideoFrameEstimates
        ? `${videoFrameAnalysis.analyzed_frame_count} sampled-frame estimates; no clip verdict`
      : 'No trained checkpoint loaded', xception || hasVideoFrameEstimates ? 'completed' : 'not_run'],
    ['Audit chain', chainOfCustody.head_sha256 || 'Not sealed', chainOfCustody.verification_status === 'verified' ? 'completed' : 'review'],
    ['Human interpretation', report.analyst_review
      ? report.analyst_review.outcome.replaceAll('_', ' ')
      : 'Awaiting review', report.analyst_review ? 'completed' : 'not_run'],
  ];

  const statusLabel = pipelineStatusLabel;

  const evidenceImage = report.media_type === 'video' && firstVideoFrame ? (
    <img src={firstVideoFrame.preview_data_url} alt={`Video keyframe at ${firstVideoFrame.timestamp_seconds} seconds`} />
  ) : evidenceUrl && !evidenceUnavailable ? (
    <img src={evidenceUrl} alt={`Analyzed evidence ${report.filename}`} onError={() => setEvidenceUnavailable(true)} />
  ) : (
    <div className="result-image-unavailable">
      <Icon name="image" size={30} />
      <strong>{report.media_type === 'audio' ? 'Audio waveform review' : 'Evidence preview unavailable'}</strong>
      <span>{report.media_type === 'audio'
        ? 'Waveform and spectrum measurements are available in Audio Signal.'
        : report.media_type === 'video'
          ? 'No video frame was decoded for preview.'
          : 'Re-select this file to view its image in the current browser session.'}</span>
    </div>
  );

  const unavailableCard = (title, detail) => (
    <section className="forensic-card unavailable-evidence">
      <div className="forensic-card-heading"><div><span className="section-kicker">NOT RUN</span><h3>{title}</h3></div><span className="evidence-pill muted-pill">Unavailable</span></div>
      <p>{detail}</p>
    </section>
  );
  const measurementRows = (measurements, percentKeys = []) => Object.entries(measurements || {}).map(([key, value]) => [
    key.replaceAll('_', ' '),
    typeof value === 'number'
      ? percentKeys.includes(key) ? `${(value * 100).toFixed(2)}%` : Number.isInteger(value) ? String(value) : value.toFixed(4)
      : value,
  ]);
  const waveformBins = audioMeasurements?.waveform_bins || [];
  const spectrumBands = audioMeasurements?.spectrogram_summary?.band_power || [];
  const maxSpectrumPower = Math.max(...spectrumBands, 0);
  const descriptiveMeasurementCard = (title, measurements, interpretation, percentKeys = []) => (
    <section className="forensic-card">
      <div className="forensic-card-heading"><div><span className="section-kicker">MEASURED · DESCRIPTIVE ONLY</span><h3>{title}</h3></div><span className="evidence-pill muted-pill">Not a detector score</span></div>
      <dl className="forensic-facts metadata-facts">
        {measurementRows(measurements, percentKeys).map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
      </dl>
      <p className="forensic-caveat">{interpretation}</p>
    </section>
  );

  const metadataRows = [
    ['File name', report.filename],
    ['File type', metadata.format ? `${metadata.format} (${report.file_integrity?.detected_mime_type || 'MIME unavailable'})` : report.file_integrity?.detected_mime_type],
    ['File size', formatBytes(report.file_size)],
    ['Resolution', metadata.width && metadata.height ? `${metadata.width} × ${metadata.height}` : null],
    ['Color mode', metadata.mode],
    ['Pixel bands', Array.isArray(metadata.pixel_bands) ? metadata.pixel_bands.join(', ') : null],
    ['Frame count', metadata.frame_count],
    ['Animated', typeof metadata.is_animated === 'boolean' ? (metadata.is_animated ? 'Yes' : 'No') : null],
    ['EXIF', typeof metadata.has_exif === 'boolean' ? (metadata.has_exif ? 'Present' : 'Not present') : null],
    ['EXIF capture time', metadata.exif_fields?.date_time_original || metadata.exif_fields?.date_time],
    ['Embedded software', metadata.exif_fields?.software || metadata.embedded_text?.software],
    ['Estimated duration', metadata.duration_estimate_seconds ? `${metadata.duration_estimate_seconds} s (decoder estimate)` : null],
    ['Decoder-reported frame rate', metadata.decoder_reported_fps ? `${metadata.decoder_reported_fps} FPS` : null],
    ['Codec (FourCC)', metadata.codec_fourcc],
    ['Decoder backend', metadata.decoder_backend],
    ['Audio container', metadata.container],
    ['Audio codec', metadata.codec_mime],
    ['Sample rate', metadata.sample_rate_hz ? `${metadata.sample_rate_hz} Hz` : null],
    ['Channels', metadata.channels],
    ['Bitrate', metadata.bitrate_bps ? `${Math.round(metadata.bitrate_bps / 1000)} kbps` : null],
    ['Audio decode status', metadata.signal_status],
    ['GPS coordinates', metadata.gps_coordinates ? `${metadata.gps_coordinates.latitude}, ${metadata.gps_coordinates.longitude} (${metadata.gps_coordinates.source})` : null],
    ['SHA-256', report.file_hash],
  ].filter(([, value]) => value !== null && value !== undefined && value !== '');

  const signalList = (
    <div className="signal-analysis-list">
      {assessmentRows.map((item) => (
        <article className="signal-analysis-item" key={item.name}>
          <span className={`signal-status-dot ${item.status === 'analyzed' || item.status === 'reviewed' || item.status === 'measured_descriptive' ? 'signal-available' : ''}`} />
          <div className="signal-analysis-copy">
            <strong>{item.name}</strong>
            <small>{item.interpretation}</small>
          </div>
          {item.probability !== null
            ? <b>{formatProbability(item.probability)}</b>
            : <span className="evidence-pill muted-pill">{statusLabel(item.status)}</span>}
        </article>
      ))}
    </div>
  );

  const modelResultDetails = xception ? (
    <dl className="forensic-facts">
      <div><dt>Architecture</dt><dd>{xception.model.architecture}</dd></div>
      {xception.model.inference_device && <div><dt>Inference device</dt><dd>{xception.model.inference_device}</dd></div>}
      <div><dt>Validation</dt><dd>{xception.model.reported_validation_accuracy || (Number.isFinite(xception.model.validation_auc) ? `${Number(xception.model.validation_auc).toFixed(4)} ROC-AUC · validation-set metric, not per-file confidence` : 'No independently verified validation result')}</dd></div>
      <div><dt>Calibration</dt><dd>{xception.model.calibration}{Number.isFinite(xception.model.calibration_temperature) ? ` · T=${Number(xception.model.calibration_temperature).toFixed(4)}` : ''}</dd></div>
      {xception.model.calibration_samples_per_class && <div><dt>Calibration samples</dt><dd>{JSON.stringify(xception.model.calibration_samples_per_class)}</dd></div>}
      <div><dt>Decision thresholds</dt><dd>{JSON.stringify(xception.model.thresholds)}</dd></div>
      {xception.model.input_preprocessing && <div><dt>Model input</dt><dd>{xception.model.input_preprocessing}</dd></div>}
      {xception.model.training_data && <div><dt>Training-data scope</dt><dd>{xception.model.training_data}</dd></div>}
      {xception.model.calibration_metrics_after_scaling && <div><dt>Calibration diagnostics</dt><dd>{formatCalibrationMetrics(xception.model.calibration_metrics_before_scaling, xception.model.calibration_metrics_after_scaling)}</dd></div>}
      {xception.conformal_prediction && <div><dt>Conformal prediction set</dt><dd>{xception.conformal_prediction.prediction_set.length ? xception.conformal_prediction.prediction_set.join(', ') : 'Empty'} · α={xception.conformal_prediction.alpha} · {xception.conformal_prediction.abstained ? 'abstained' : 'threshold-selected class is supported'}</dd></div>}
      {xception.model.test_metrics && <div><dt>Independent test benchmark</dt><dd>{formatBenchmarkMetrics(xception.model.test_metrics)} · Dataset-level metrics, not this file's confidence.</dd></div>}
      {xception.model.test_metrics_by_source && Object.entries(xception.model.test_metrics_by_source).map(([source, metrics]) => <div key={`test-source-${source}`}><dt>Test source · {source}</dt><dd>{formatBenchmarkMetrics(metrics)} · {metrics.source_not_seen_in_train_validation_or_calibration ? 'Source held out from model development.' : 'Source also appears in model-development splits.'}</dd></div>)}
      {xception.model.dataset_audit?.group_overlap_status && <div><dt>Group-leakage audit</dt><dd>{String(xception.model.dataset_audit.group_overlap_status).replaceAll('_', ' ')} · {xception.model.dataset_audit.manifest_sha256 || 'No audit digest'}</dd></div>}
      <div><dt>Face preprocessing</dt><dd>{faceReview ? `${faceReview.detector}: ${faceReview.status.replaceAll('_', ' ')}` : 'Not recorded'}</dd></div>
      <div><dt>Face alignment</dt><dd>{faceReview?.alignment?.replaceAll('_', ' ') || 'Not performed'}</dd></div>
    </dl>
  ) : <p>No trained Xception checkpoint is available. No model score or class was generated.</p>;

  let tabContent;
  if (activeTab === 'Face Analysis') {
    tabContent = report.media_type === 'video' && videoFrames.length ? (
      <section className="forensic-card"><div className="forensic-card-heading"><div><span className="section-kicker">PER-SAMPLED-FRAME OBSERVATIONS</span><h3>Face-candidate review across video keyframes</h3></div><span className="evidence-pill muted-pill">Not identity tracking</span></div><div className="video-keyframe-grid">{videoFrames.map((frame) => <article className="video-keyframe-card" key={`face-${frame.index}`}><img src={frame.preview_data_url} alt={`Analyzed frame at ${frame.timestamp_seconds} seconds`} loading="lazy" /><div><strong>{frame.timestamp_seconds.toFixed(3)} s</strong><small>{frame.face_review.status === 'unavailable' ? frame.face_review.interpretation : `${frame.face_review.detected_face_count} face candidates · ${frame.face_review.status.replaceAll('_', ' ')}`}</small></div></article>)}</div><p className="forensic-caveat">These frame-level candidates are not linked into identities and do not detect face swaps or video manipulation.</p></section>
    ) : faceReview ? (
      <div className="result-feature-grid">
        <section className="forensic-card evidence-image-card"><div className="forensic-card-heading"><div><span className="section-kicker">SOURCE IMAGE</span><h3>Face detection input</h3></div></div><div className="result-image-frame">{evidenceImage}</div></section>
        <section className="forensic-card"><div className="forensic-card-heading"><div><span className="section-kicker">FACE REVIEW</span><h3>Detection details</h3></div><span className="evidence-pill">{faceReview.status.replaceAll('_', ' ')}</span></div><dl className="forensic-facts">
          <div><dt>Detector</dt><dd>{faceReview.detector}</dd></div>
          <div><dt>Detected candidates</dt><dd>{faceReview.status === 'unavailable' ? 'Not available — detector did not run' : faceReview.detected_face_count}</dd></div>
          <div><dt>Selected input</dt><dd>{faceReview.selected_crop.replaceAll('_', ' ')}</dd></div>
          <div><dt>Crop box (x1, y1, x2, y2)</dt><dd>{faceReview.crop_box_xyxy?.join(', ') || 'No face crop available'}</dd></div>
          <div><dt>Landmark alignment</dt><dd>{faceReview.alignment.replaceAll('_', ' ')}</dd></div>
        </dl><p className="forensic-caveat">Face detector candidates are not face-confidence scores. No facial landmarks or regional manipulation scores are available.</p></section>
      </div>
    ) : unavailableCard('Face analysis', 'Face-candidate detection is only run for static images. This file has no face-candidate result; face authenticity and manipulation are not inferred.');
  } else if (activeTab === 'Heatmap') {
    tabContent = explainability?.status === 'generated' ? (
      <section className="forensic-card heatmap-result">
        <div className="forensic-card-heading"><div><span className="section-kicker">MODEL ATTENTION / EVIDENCE VISUALIZATION</span><h3>Model attention evidence</h3></div><span className="evidence-pill muted-pill">{explainability.method} · coarse explanation</span></div>
        <img src={explainability.overlay_data_url} alt="Model attention visualization: Grad-CAM highlights regions that contributed to the model's fake-class prediction" />
        <dl className="forensic-facts metadata-facts"><div><dt>Target</dt><dd>{explainability.target}</dd></div><div><dt>Model layer</dt><dd>{explainability.target_layer}</dd></div><div><dt>Model input</dt><dd>{explainability.model_input_description || xception.face_review?.selected_crop?.replaceAll('_', ' ') || 'Full image'}</dd></div><div><dt>Overlay size</dt><dd>{explainability.overlay_width} × {explainability.overlay_height}</dd></div></dl>
        <p className="forensic-caveat"><strong>Important:</strong> Grad-CAM tells us <strong>which regions contributed most to the model's prediction</strong>. It does <strong>not</strong> independently establish that those pixels were manipulated.</p>
        <section className="evidence-pipeline-roadmap" aria-label="Forensic evidence pipeline status">
          <div className="forensic-card-heading"><div><span className="section-kicker">FORENSIC EVIDENCE PIPELINE</span><h4>Combined assessment components</h4></div><span className="evidence-pill muted-pill">Availability for this report</span></div>
          <ol>
            {[
              ['Xception prediction', xception ? (uncalibrated ? 'Uncalibrated model estimate produced' : `Model status: ${statusLabel(report.decision_status)}`) : 'No classifier prediction produced'],
              ['Grad-CAM++ explanation', explainability.method === 'Grad-CAM++' ? 'Generated' : 'Not configured; this report contains Grad-CAM, not Grad-CAM++'],
              ['Face / region localization', faceReview ? `${faceReview.detected_face_count ?? 'Unavailable'} face candidate(s) observed; manipulation-region localization is not configured` : 'Not run'],
              ['FFT / DCT evidence', frequencyMeasurements ? 'FFT descriptive measurements available; DCT manipulation classifier is not configured' : 'Frequency analysis not available for this media'],
              ['Compression analysis', noiseMeasurements ? 'High-pass residual measurements available; validated compression classifier is not configured' : 'No validated compression classifier configured'],
              ['Metadata', 'Available image metadata reviewed; metadata does not prove source or authenticity'],
              ['Ensemble models', 'Not configured; no model-voting or fusion score is produced'],
              ['Calibrated confidence', xception && !uncalibrated ? 'Checkpoint calibration metadata available; estimate is not a per-file guarantee' : 'Not calibrated for this prediction'],
              ['Final forensic assessment', `${report.verdict || 'Insufficient evidence'} · ${statusLabel(report.decision_status || 'withheld')}; analyst review remains separate`],
            ].map(([stage, detail]) => <li key={stage}><strong>{stage}</strong><span>{detail}</span></li>)}
          </ol>
          <p className="forensic-caveat">These components are not yet fused into one evidence-weighted model. “Not configured” and descriptive-only results are kept explicit rather than combined into an unsupported authenticity score.</p>
        </section>
      </section>
    ) : unavailableCard('Heatmap', explainability?.reason || tabsNotRun.Heatmap);
  } else if (activeTab === 'Frequency') {
    tabContent = frequencyMeasurements
      ? descriptiveMeasurementCard(
        'Image frequency spectrum',
        frequencyMeasurements,
        report.frequency_analysis.interpretation,
        ['low_frequency_power_fraction', 'mid_frequency_power_fraction', 'high_frequency_power_fraction'],
      )
      : unavailableCard('Frequency spectrum', 'Frequency measurements are generated for static images only. No classifier or manipulation score is inferred.');
  } else if (activeTab === 'Noise Analysis') {
    tabContent = noiseMeasurements
      ? descriptiveMeasurementCard('High-pass residual measurements', noiseMeasurements, report.noise_analysis.interpretation)
      : unavailableCard('Noise analysis', 'Residual measurements are generated for static images only. No noise-forensics classification or manipulation score is inferred.');
  } else if (activeTab === 'Video Frames') {
    tabContent = videoFrames.length
      ? <section className="forensic-card"><div className="forensic-card-heading"><div><span className="section-kicker">EVENLY SPACED · DECODED FRAMES</span><h3>Representative video keyframes</h3></div><span className="evidence-pill muted-pill">{videoFrames.length} frames{videoFaceDetectorAvailable ? ` · ${sampledVideoFaceCount} face candidates` : ''}</span></div><p className="forensic-caveat">{report.video_analysis.interpretation}</p>{videoFrameAnalysis && <p className="forensic-caveat"><strong>Sampled-frame model run:</strong> {videoFrameAnalysis.analyzed_frame_count} of {videoFrameAnalysis.sampled_frame_count} frame(s), {videoFrameAnalysis.architecture} · {videoFrameAnalysis.calibration}. No clip-level authenticity verdict was produced.</p>}<div className="video-keyframe-grid">{videoFrames.map((frame) => <article className="video-keyframe-card" key={`${frame.index}-${frame.timestamp_seconds}`}><img src={frame.preview_data_url} alt={`Frame at ${frame.timestamp_seconds} seconds`} loading="lazy" /><div><strong>Frame {frame.index + 1} · {frame.timestamp_seconds.toFixed(3)} s</strong><small>{frame.width} × {frame.height} · {frame.face_review.status === 'unavailable' ? 'Face detector unavailable' : `${frame.face_review.detected_face_count} face candidates`}</small>{frame.xception_estimate && <small>Xception · deepfake {formatProbability(frame.xception_estimate.deepfake_score)} · {frame.xception_estimate.calibration}</small>}</div></article>)}</div><p className="forensic-caveat">Keyframe sampling can miss events between frames. Per-frame image scores are not calibrated for video and are not aggregated into a clip verdict.</p></section>
      : unavailableCard('Video keyframes', report.video_analysis?.interpretation || 'No video frames were decoded.');
  } else if (activeTab === 'Audio Signal') {
    tabContent = audioMeasurements
      ? <section className="forensic-card"><div className="forensic-card-heading"><div><span className="section-kicker">DECODED AUDIO · FIRST {report.audio_analysis.analysis_limit_seconds} SECONDS MAX</span><h3>Waveform and spectrum summary</h3></div><span className="evidence-pill muted-pill">Descriptive only</span></div><div className="audio-summary-facts"><div><span>Sample rate</span><strong>{report.audio_analysis.sample_rate_hz} Hz</strong></div><div><span>Channels</span><strong>{report.audio_analysis.channels}</strong></div><div><span>Decoded duration</span><strong>{audioMeasurements.analysis_duration_seconds} s</strong></div><div><span>RMS amplitude</span><strong>{audioMeasurements.rms_amplitude.toFixed(5)}</strong></div><div><span>Peak amplitude</span><strong>{audioMeasurements.peak_amplitude.toFixed(5)}</strong></div><div><span>Clipped fraction</span><strong>{formatProbability(audioMeasurements.clipped_sample_fraction)}</strong></div><div><span>Zero-crossing rate</span><strong>{audioMeasurements.zero_crossing_rate.toFixed(4)}</strong></div><div><span>Spectral centroid</span><strong>{audioMeasurements.spectrogram_summary.power_weighted_spectral_centroid_hz.toFixed(1)} Hz</strong></div><div><span>Spectral flatness</span><strong>{audioMeasurements.spectrogram_summary.spectral_flatness.toFixed(4)}</strong></div></div><h4>Waveform envelope · min/max per time bin</h4><div className="audio-waveform" role="img" aria-label="Audio waveform amplitude envelope">{waveformBins.map((bin, index) => <span key={index} style={{ top: `${50 - Math.max(-1, Math.min(1, bin.maximum)) * 48}%`, height: `${Math.max(2, Math.abs(bin.maximum - bin.minimum) * 48)}%` }} />)}</div><h4>Coarse spectral power · equal-width bands</h4><div className="audio-spectrum" role="img" aria-label="Coarse audio spectral power bands">{spectrumBands.map((power, index) => <span key={index} title={`Band ${index + 1}`} style={{ height: `${maxSpectrumPower > 0 ? Math.max(2, (Math.log1p(power) / Math.log1p(maxSpectrumPower)) * 100) : 2}%` }} />)}</div><p className="forensic-caveat">{report.audio_analysis.interpretation} No synthetic-voice detector or speech transcription is configured.</p></section>
      : unavailableCard('Audio signal', report.audio_analysis?.interpretation || 'Audio signal decoding is not available for this file.');
  } else if (activeTab === 'Comparisons') {
    const comparisonRows = [
      ['Xception', 'Static image', xception ? formatProbability(report.deepfake_probability) : 'No checkpoint loaded'],
      ['EfficientNet-B4', 'Image', 'Not implemented / configured'],
      ['Swin Transformer', 'Image', 'Not implemented / configured'],
      ['Frequency CNN', 'Image signal', 'Not implemented / configured'],
      ['Video / audio classifiers', 'Video / audio', 'Not implemented / configured'],
      ['Ensemble fusion', 'Multimodal', 'Not implemented / configured'],
    ];
    tabContent = <section className="forensic-card"><div className="forensic-card-heading"><div><span className="section-kicker">MODEL AVAILABILITY</span><h3>Model comparison</h3></div><span className="evidence-pill muted-pill">No consensus inferred</span></div><div className="table-wrap"><table><thead><tr><th>Model</th><th>Modality</th><th>Result / status</th></tr></thead><tbody>{comparisonRows.map(([name, modality, result]) => <tr key={name}><td>{name}</td><td>{modality}</td><td>{result}</td></tr>)}</tbody></table></div><p className="forensic-caveat">Only a compatible, trained Xception checkpoint can produce an image-classification estimate. Missing models are not assigned substitute scores.</p></section>;
  } else if (activeTab === 'Metadata') {
    tabContent = <div className="forensic-card"><div className="forensic-card-heading"><div><span className="section-kicker">OBSERVED FILE PROPERTIES</span><h3>Metadata and evidence identity</h3></div><button className="text-button" onClick={onCopyHash}>Copy SHA-256</button></div><dl className="forensic-facts metadata-facts">{metadataRows.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{label === 'File size' ? value : availableValue(value)}</dd></div>)}</dl><div className="metadata-finding-list">{metadataFindings.map((finding) => <article key={finding.signal}><span className="evidence-pill muted-pill">{finding.severity.replaceAll('_', ' ')}</span><div><strong>{finding.signal}</strong><p>{finding.interpretation}</p></div></article>)}</div></div>;
  } else if (activeTab === 'Timeline') {
    tabContent = <div className="forensic-card"><div className="forensic-card-heading"><div><span className="section-kicker">AUDIT EVENTS</span><h3>Evidence processing timeline</h3></div></div><ol className="evidence-timeline">{(report.audit_log || []).map((event, index) => <li key={`${event.timestamp}-${index}`}><time>{formatDate(event.timestamp)}</time><p>{event.message}</p></li>)}</ol></div>;
  } else if (activeTab === 'Evidence Graph') {
    tabContent = <div className="forensic-card"><div className="forensic-card-heading"><div><span className="section-kicker">TRACEABLE DATA FLOW</span><h3>Evidence graph</h3></div></div><div className="evidence-graph">{graphNodes.map(([title, detail, status], index) => <div className="evidence-graph-node" key={title}><span className={`graph-node-index ${status}`}>{index + 1}</span><div><strong>{title}</strong><small>{detail}</small></div>{index < graphNodes.length - 1 && <span className="graph-connector" />}</div>)}</div><p className="forensic-caveat">This graph represents recorded processing steps, not a claim of provenance or authenticity.</p></div>;
  } else if (activeTab === 'Model Details') {
    tabContent = <div className="forensic-card"><div className="forensic-card-heading"><div><span className="section-kicker">MODEL OUTPUT</span><h3>Xception assessment</h3></div><span className={`evidence-pill ${xception ? '' : 'muted-pill'}`}>{xception ? 'Loaded checkpoint' : 'Not configured'}</span></div>{modelResultDetails}{xception && <ul className="model-limitations">{xception.limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}</ul>}<p className="forensic-caveat">A calibrated probability is a model estimate, not a guarantee that this individual classification is correct.</p></div>;
  } else if (activeTab === 'Result') {
    tabContent = (
      <div className="result-analysis-layout">
        <div className="result-primary-column">
          <div className="result-feature-grid">
            <section className="forensic-card evidence-image-card"><div className="forensic-card-heading"><div><span className="section-kicker">{report.media_type === 'video' ? 'DECODED VIDEO KEYFRAME' : report.media_type === 'audio' ? 'AUDIO EVIDENCE' : 'ORIGINAL / ANALYZED IMAGE'}</span><h3>{report.filename}</h3></div><button className="icon-button" onClick={() => window.open(evidenceUrl, '_blank', 'noopener,noreferrer')} disabled={!evidenceUrl || evidenceUnavailable} aria-label="Open evidence preview"><Icon name="arrow" size={15} /></button></div><div className="result-image-frame">{evidenceImage}</div></section>
            <section className="forensic-card explanation-placeholder"><div className="forensic-card-heading"><div><span className="section-kicker">MODEL ATTENTION / EVIDENCE VISUALIZATION</span><h3>Model attention evidence</h3></div><span className="evidence-pill muted-pill">{explainability?.status === 'generated' ? `Generated · ${explainability.method}` : 'Not generated'}</span></div>{explainability?.status === 'generated' ? <div className="heatmap-side-preview"><img src={explainability.overlay_data_url} alt="Model attention visualization showing regions that contributed to the model prediction" /><p>Shows which regions contributed to the model prediction; it does not establish manipulation or show a boundary.</p></div> : <div className="analysis-artifact-empty"><Icon name="scan" size={29} /><strong>Visualization unavailable</strong><p>{explainabilityUnavailableReason}</p></div>}</section>
          </div>
          <div className="result-thumbnail-strip"><span>Evidence item</span><strong>{report.filename}</strong><small>{formatBytes(report.file_size)} · SHA-256 recorded</small></div>
          <section className="forensic-card result-signal-card"><div className="forensic-card-heading"><div><span className="section-kicker">SIGNAL ANALYSIS</span><h3>Available analysis signals</h3></div><span className="evidence-pill muted-pill">No ensemble configured</span></div>{signalList}</section>
          <div className="result-feature-grid result-bottom-grid">
            {frequencyMeasurements
              ? <section className="forensic-card"><div className="forensic-card-heading"><div><span className="section-kicker">MEASURED · DESCRIPTIVE</span><h3>Frequency spectrum</h3></div></div><dl className="compact-facts"><div><dt>High-frequency power</dt><dd>{formatProbability(frequencyMeasurements.high_frequency_power_fraction)}</dd></div><div><dt>Spectrum log-magnitude mean</dt><dd>{frequencyMeasurements.log_magnitude_mean.toFixed(4)}</dd></div></dl><p className="forensic-caveat">Descriptive signal only; not a deepfake score.</p></section>
              : unavailableCard('Frequency domain', 'No spectrum measurement is available for this media type.')}
            {noiseMeasurements
              ? <section className="forensic-card"><div className="forensic-card-heading"><div><span className="section-kicker">MEASURED · DESCRIPTIVE</span><h3>High-pass residual</h3></div></div><dl className="compact-facts"><div><dt>Residual RMS</dt><dd>{noiseMeasurements.residual_rms.toFixed(4)} gray levels</dd></div><div><dt>Residual standard deviation</dt><dd>{noiseMeasurements.residual_standard_deviation.toFixed(4)} gray levels</dd></div></dl><p className="forensic-caveat">Fine detail, edges, sensor noise, and compression all contribute; not a manipulation score.</p></section>
              : unavailableCard('Noise / compression', 'No residual measurement is available for this media type.')}
          </div>
        </div>
        <aside className="result-side-column">
          <section className={`forensic-card detection-card ${xception ? 'model-result-available' : ''}`}>
            <div className="forensic-card-heading"><div><span className="section-kicker">DETECTION RESULT</span><h3>Model assessment</h3></div></div>
            <div className={`classification-banner ${xception ? (
              report.verdict === 'Likely deepfake' ? 'classification-risk'
                : report.verdict === 'Likely original' ? 'classification-real'
                  : report.verdict === 'Suspicious' ? 'classification-review'
                    : 'classification-unavailable'
            ) : 'classification-unavailable'}`}><Icon name={report.verdict === 'Likely original' ? 'check' : 'shield'} size={18} /><strong>{xception ? report.verdict.toUpperCase() : report.evidence_context?.is_synthetic_demo ? 'DEMO SAMPLE · NOT CLASSIFIED' : 'VERDICT WITHHELD · NO CLASSIFIER'}</strong></div>
            <div className="probability-gauge" style={probability === null ? undefined : { '--probability': `${probability * 100}%` }}><div><strong>{probability === null ? 'NO SCORE' : formatProbability(probability)}</strong><small>{xception ? (uncalibrated ? 'Uncalibrated model estimate' : 'Calibrated model estimate') : 'Inference did not run'}</small></div></div>
            <div className="probability-legend"><span><i className="legend-fake" />{uncalibrated ? 'Deepfake softmax score' : 'Deepfake probability'} {formatProbability(probability)}</span><span><i className="legend-real" />{uncalibrated ? 'Authenticity softmax score' : 'Authenticity probability'} {formatProbability(report.authenticity_probability)}</span></div>
            <p className="forensic-caveat">{report.decision_basis || 'No trained, validated classifier produced a result. The verdict is withheld.'} Descriptive metadata, face candidates, FFT and residual measurements cannot determine original versus deepfake.</p>
            <div className="decision-facts"><div><span>Classifier status</span><strong>{pipelineStatusLabel(xception ? report.decision_status : report.evidence_context?.is_synthetic_demo ? 'skipped_demo' : 'not_configured')}</strong></div><div><span>Per-file confidence</span><strong>{xception ? 'Not separately calibrated' : 'Not computed'}</strong></div><div><span>Evidence reliability</span><strong>{report.evidence_reliability?.startsWith('Not rated') ? 'Not rated' : report.evidence_reliability || 'Not rated'}</strong></div></div>
          </section>
          <section className="forensic-card"><div className="forensic-card-heading"><div><span className="section-kicker">MEDIA INFORMATION</span><h3>File properties</h3></div></div><dl className="compact-facts">{metadataRows.slice(0, 9).map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{label === 'File size' ? value : availableValue(value)}</dd></div>)}</dl></section>
          <section className="forensic-card"><div className="forensic-card-heading"><div><span className="section-kicker">METADATA FINDINGS</span><h3>Observed indicators</h3></div><span className="evidence-pill muted-pill">{metadataFindings.length} finding{metadataFindings.length === 1 ? '' : 's'}</span></div>{metadataFindings.length ? <div className="metadata-finding-list">{metadataFindings.map((finding) => <article key={finding.signal}><span className="evidence-pill muted-pill">{finding.severity.replaceAll('_', ' ')}</span><div><strong>{finding.signal}</strong><p>{finding.interpretation}</p></div></article>)}</div> : <p>No metadata findings were returned by the parser.</p>}</section>
          <section className="forensic-card"><div className="forensic-card-heading"><div><span className="section-kicker">MANIPULATION TYPE</span><h3>Subtype assessment</h3></div></div><p>{xception ? 'This image classifier does not distinguish face swap, reenactment, AI-generated media, or expression manipulation.' : 'Not assessed. No manipulation subtype classifier is configured.'}</p><span className="evidence-pill muted-pill">Not classified</span></section>
          <section className="forensic-card location-card"><div className="forensic-card-heading"><div><span className="section-kicker">GEOLOCATION</span><h3>Location evidence</h3></div></div>{metadata.gps_coordinates ? <div className="location-unavailable"><Icon name="shield" size={18} /><div><strong>EXIF GPS coordinates observed</strong><small>{metadata.gps_coordinates.latitude}, {metadata.gps_coordinates.longitude} · source: {metadata.gps_coordinates.source}. Coordinates are reported as embedded and have not been independently verified.</small></div></div> : <div className="location-unavailable"><Icon name="shield" size={18} /><div><strong>No verified location available</strong><small>No valid EXIF GPS coordinates or investigator-recorded acquisition location were supplied. No map pin is shown.</small></div></div>}</section>
        </aside>
      </div>
    );
  } else {
    tabContent = tabsNotRun[activeTab]
      ? unavailableCard(activeTab, tabsNotRun[activeTab])
      : unavailableCard(activeTab, 'No validated evidence artifact is available for this view.');
  }

  return (
    <section className="analysis-result-page" aria-labelledby="analysis-result-title">
      <div className="analysis-result-toolbar">
        <div className="result-breadcrumb"><span>Analyze</span><b>›</b><span>{mediaTypeLabel}</span><b>›</b><strong>{report.case_id}</strong></div>
        <div className="row-actions">
          <button className="button button-secondary" onClick={onCopyHash}><Icon name="copy" size={14} /> SHA-256</button>
          <button className="button button-secondary" onClick={onPreviewReport}>Report preview</button>
          <button className="button button-secondary" disabled={reanalyzing} onClick={onReanalyze}>{reanalyzing ? 'Reanalyzing…' : 'Reanalyze saved evidence'}</button>
          <button className="button button-primary" disabled={exporting} onClick={onExportReport}>{exporting ? 'Creating…' : 'Download report'}</button>
        </div>
      </div>
      <ol className="analysis-stepper" aria-label="Analysis processing steps">
        {pipeline.map((step, index) => <li className={`step-${step.status}`} key={step.label}><span>{step.status === 'completed' ? <Icon name="check" size={12} /> : index + 1}</span><strong>{step.label}</strong>{index < pipeline.length - 1 && <i />}</li>)}
      </ol>
      <section className="model-pipeline-panel forensic-card" aria-labelledby="model-pipeline-title">
        <div className="forensic-card-heading">
          <div><span className="section-kicker">MODEL PIPELINE · EXECUTION TRACE</span><h3 id="model-pipeline-title">What actually ran for this file</h3></div>
          <span className="evidence-pill muted-pill">{xception || hasVideoFrameEstimates ? 'Xception executed' : 'Classification withheld'}</span>
        </div>
        {!report.model_pipeline && <p className="forensic-caveat">This older saved report predates detailed pipeline tracing; statuses below are reconstructed from its recorded classifier and signal fields.</p>}
        <div className="model-pipeline-grid">
          {modelPipeline.map((stage) => (
            <article className={`model-pipeline-stage stage-${pipelineStepClass(stage.status)}`} key={`${stage.number}-${stage.name}`}>
              <span className="model-pipeline-number">{String(stage.number).padStart(2, '0')}</span>
              <div className="model-pipeline-copy"><strong>{stage.name}</strong><p>{stage.detail}</p></div>
              <span className={`model-pipeline-status status-${pipelineStepClass(stage.status)}`}>{pipelineStatusLabel(stage.status)}</span>
            </article>
          ))}
        </div>
        <p className="forensic-caveat">{modelPipelineCaveat} Descriptive measurements must not be interpreted as class probabilities.</p>
      </section>
      {report.evidence_context?.is_synthetic_demo && <div className="synthetic-demo-notice"><Icon name="shield" size={16} /><p><strong>Synthetic demonstration sample — not ground truth</strong><br />{report.evidence_context.notice} Inference was skipped to prevent a demo illustration from being presented as an authentic model test.</p></div>}
      <div className="analysis-evidence-header">
        <div className="analysis-file-identity"><div className="analysis-file-preview">{evidenceUrl && !evidenceUnavailable ? <img src={evidenceUrl} alt="" onError={() => setEvidenceUnavailable(true)} /> : <Icon name="file" size={22} />}</div><div><h1 id="analysis-result-title">{report.filename}</h1><p>{availableValue(metadata.format, report.media_type?.toUpperCase())} · {formatBytes(report.file_size)} · {metadata.width && metadata.height ? `${metadata.width} × ${metadata.height}` : 'Dimensions unavailable'}</p><code>SHA-256: {report.file_hash}</code></div></div>
        <span className={`analysis-complete-tag ${xception || hasVideoFrameEstimates ? 'model-complete' : ''}`}><span />{xception ? 'Analysis completed · Xception ran' : hasVideoFrameEstimates ? 'Forensic review completed · sampled-frame estimates only' : report.evidence_context?.is_synthetic_demo ? 'Forensic review completed · demo classifier skipped' : 'Forensic review completed · classifier not configured'}</span>
      </div>
      {report.media_type === 'image' && !metadata.is_animated
        && (!report.frequency_analysis || !report.noise_analysis)
        && <div className="synthetic-demo-notice"><Icon name="shield" size={16} /><p><strong>Earlier report — descriptive signal data is missing</strong><br />This record predates static-image spectrum and residual measurements. Re-analyze the image after the backend has been restarted to populate those views. A trained deepfake classifier is still required for an authenticity result.</p></div>}
      <nav className="analysis-result-tabs" aria-label="Analysis result sections">{resultTabs.map((tab) => <button key={tab} className={activeTab === tab ? 'active' : ''} onClick={() => setActiveTab(tab)}>{tab}</button>)}</nav>
      <div className="analysis-tab-content">{tabContent}</div>
      <section className="forensic-card interpretation-summary"><div className="forensic-card-heading"><div><span className="section-kicker">FORENSIC SUMMARY</span><h3>Interpretation and limitations</h3></div><span className="evidence-pill muted-pill">Analyst review separate</span></div><p>{report.summary}</p><ul>{(report.explanation || []).slice(0, 5).map((line) => <li key={line}>{line}</li>)}</ul></section>
      {children}
    </section>
  );
}

function App() {
  const [view, setView] = useState('overview');
  const [activeNav, setActiveNav] = useState('overview');
  const [mediaMode, setMediaMode] = useState('image');
  const [caseTitle, setCaseTitle] = useState('New media review');
  const [investigator, setInvestigator] = useState('Analyst');
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState(null);
  const [previewReport, setPreviewReport] = useState(null);
  const [cases, setCases] = useState([]);
  const [reports, setReports] = useState([]);
  const [recordQuery, setRecordQuery] = useState('');
  const [recordMediaType, setRecordMediaType] = useState('all');
  const [caseStatusFilter, setCaseStatusFilter] = useState('all');
  const [reportVerdictFilter, setReportVerdictFilter] = useState('all');
  const [apiStatus, setApiStatus] = useState('checking');
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [loadingExample, setLoadingExample] = useState('');
  const [exportingReport, setExportingReport] = useState('');
  const [reanalyzingReport, setReanalyzingReport] = useState('');
  const [reviewOutcome, setReviewOutcome] = useState('inconclusive');
  const [reviewRationale, setReviewRationale] = useState('');
  const [reviewerName, setReviewerName] = useState('Analyst');
  const [reviewSaving, setReviewSaving] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [previewUrl, setPreviewUrl] = useState('');
  const [activeResultTab, setActiveResultTab] = useState('Result');

  useEffect(() => {
    let cancelled = false;

    const refreshWorkspace = async () => {
      try {
        const [healthResponse, casesResponse, reportsResponse] = await Promise.all([
          fetch(`${API_BASE}/api/health`),
          fetch(`${API_BASE}/api/cases`),
          fetch(`${API_BASE}/api/reports`),
        ]);
        if (!healthResponse.ok || !casesResponse.ok || !reportsResponse.ok) {
          throw new Error('The analysis service returned an error.');
        }
        const [health, caseData, reportData] = await Promise.all([
          healthResponse.json(),
          casesResponse.json(),
          reportsResponse.json(),
        ]);
        if (!cancelled) {
          setApiStatus(health.status === 'ok' ? 'online' : 'offline');
          setCases(caseData.items || []);
          setReports(reportData.items || []);
        }
      } catch {
        if (!cancelled) setApiStatus('offline');
      }
    };

    refreshWorkspace();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!selectedFile || !selectedFile.type.startsWith('image/')) {
      setPreviewUrl('');
      return undefined;
    }
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [selectedFile]);

  useEffect(() => {
    if (!previewReport) return undefined;
    const closeOnEscape = (event) => {
      if (event.key === 'Escape') setPreviewReport(null);
    };
    document.addEventListener('keydown', closeOnEscape);
    return () => document.removeEventListener('keydown', closeOnEscape);
  }, [previewReport]);

  const metrics = useMemo(() => {
    const analyses = reports.length;
    const flagged = reports.filter((report) => report.verdict !== 'Authentic').length;
    return [
      { label: 'Analyses completed', value: analyses, note: 'In this session', icon: 'scan', tone: 'blue' },
      { label: 'Cases tracked', value: cases.length, note: 'Active investigations', icon: 'folder', tone: 'violet' },
      { label: 'Flagged for review', value: flagged, note: 'Requires analyst attention', icon: 'shield', tone: 'amber' },
    ];
  }, [cases, reports]);
  const filteredCases = useMemo(() => {
    const query = recordQuery.trim().toLocaleLowerCase();
    return cases.filter((item) => (
      (!query || [item.case_id, item.title, item.investigator, item.category]
        .some((value) => String(value || '').toLocaleLowerCase().includes(query)))
      && (caseStatusFilter === 'all' || item.status === caseStatusFilter)
    ));
  }, [cases, recordQuery, caseStatusFilter]);
  const filteredReports = useMemo(() => {
    const query = recordQuery.trim().toLocaleLowerCase();
    return reports.filter((report) => (
      (!query || [report.case_id, report.filename, report.case_title, report.investigator, report.file_hash]
        .some((value) => String(value || '').toLocaleLowerCase().includes(query)))
      && (recordMediaType === 'all' || report.media_type === recordMediaType)
      && (reportVerdictFilter === 'all' || report.verdict === reportVerdictFilter)
    ));
  }, [reports, recordQuery, recordMediaType, reportVerdictFilter]);

  const selectFile = (file) => {
    if (!file) return;
    setSelectedFile(file);
    setError('');
    setNotice('');
  };

  const loadExample = async (example) => {
    setLoadingExample(example.id);
    setError('');
    setNotice('');
    try {
      const response = await fetch(`/examples/${example.filename}`);
      if (!response.ok) throw new Error(`Could not load ${example.title.toLowerCase()}.`);
      const blob = await response.blob();
      selectFile(new File([blob], example.filename, { type: blob.type || 'image/png' }));
      setCaseTitle(example.caseTitle);
    } catch (loadError) {
      setError(loadError.message || 'Could not load this sample image.');
    } finally {
      setLoadingExample('');
    }
  };

  const refreshRecords = async () => {
    const [casesResponse, reportsResponse] = await Promise.all([
      fetch(`${API_BASE}/api/cases`),
      fetch(`${API_BASE}/api/reports`),
    ]);
    if (!casesResponse.ok || !reportsResponse.ok) {
      throw new Error('Analysis completed, but the case history could not be refreshed.');
    }
    const [caseData, reportData] = await Promise.all([
      casesResponse.json(),
      reportsResponse.json(),
    ]);
    setCases(caseData.items || []);
    setReports(reportData.items || []);
  };

  const handleUpload = async (event) => {
    event.preventDefault();
    if (!selectedFile) {
      setError('Choose an image, video, or audio file to continue.');
      return;
    }
    if (!caseTitle.trim() || !investigator.trim()) {
      setError('Add a case title and investigator before starting analysis.');
      return;
    }

    setLoading(true);
    setError('');
    setNotice('');
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('case_title', caseTitle.trim());
    formData.append('investigator', investigator.trim());
    formData.append('case_id', `case-${Date.now()}`);

    try {
      const response = await fetch(`${API_BASE}/api/analyze`, { method: 'POST', body: formData });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail || `Analysis failed with status ${response.status}.`);
      }
      setResult(payload);
      setActiveResultTab('Result');
      setReviewOutcome('inconclusive');
      setReviewRationale('');
      setReviewerName(investigator.trim());
      setNotice('Analysis complete. Review the evidence indicators before drawing conclusions.');
      setView('result');
      setActiveNav(mediaMode === 'image' ? 'image-analysis' : `${mediaMode}-analysis`);
      await refreshRecords();
    } catch (requestError) {
      if (requestError instanceof TypeError && requestError.message === 'Failed to fetch') setApiStatus('offline');
      setError(describeRequestError(requestError, 'Could not complete the analysis request.'));
    } finally {
      setLoading(false);
    }
  };

  const saveAnalystReview = async (event) => {
    event.preventDefault();
    if (!result?.case_id) {
      setError('Analyze a file before recording an analyst interpretation.');
      return;
    }
    setReviewSaving(true);
    setError('');
    try {
      const response = await fetch(`${API_BASE}/api/reports/${encodeURIComponent(result.case_id)}/review`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          outcome: reviewOutcome,
          rationale: reviewRationale,
          reviewer: reviewerName,
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `Could not save review (${response.status}).`);
      const reviewedResult = { ...result, analyst_review: payload };
      setResult(reviewedResult);
      setReports((current) => current.map((report) => report.case_id === result.case_id ? reviewedResult : report));
      setCases((current) => current.map((item) => item.case_id === result.case_id ? { ...item, status: 'reviewed', reviewed_at: payload.reviewed_at } : item));
      setNotice('Analyst interpretation saved and attached to this case.');
    } catch (reviewError) {
      if (reviewError instanceof TypeError && reviewError.message === 'Failed to fetch') setApiStatus('offline');
      setError(describeRequestError(reviewError, 'Could not save the analyst interpretation.'));
    } finally {
      setReviewSaving(false);
    }
  };

  const downloadReport = (report) => {
    const exportableReport = Object.fromEntries(
      Object.entries(report).filter(([key]) => !['evidence_preview_token', 'file_path'].includes(key)),
    );
    const blob = new Blob([JSON.stringify(exportableReport, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${report.case_id || 'deeptrace-report'}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const downloadAuditHistory = async () => {
    setError('');
    try {
      const response = await fetch(`${API_BASE}/api/reports/audit.csv`);
      if (!response.ok) throw new Error(`Could not export audit history (${response.status}).`);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'deeptrace-audit-history.csv';
      link.click();
      URL.revokeObjectURL(url);
      setNotice('Audit history exported as CSV.');
    } catch (exportError) {
      setError(describeRequestError(exportError, 'Could not export audit history.'));
    }
  };

  const downloadAllReports = () => {
    const exportableReports = reports.map((report) => Object.fromEntries(
      Object.entries(report).filter(([key]) => !['evidence_preview_token', 'file_path'].includes(key)),
    ));
    const blob = new Blob([JSON.stringify(exportableReports, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'deeptrace-reports.json';
    link.click();
    URL.revokeObjectURL(url);
    setNotice(`${reports.length} report${reports.length === 1 ? '' : 's'} exported as JSON.`);
  };

  const exportWordReport = async (report) => {
    const reportKey = report.case_id || report.filename || 'report';
    setExportingReport(reportKey);
    setError('');
    try {
      await downloadAnalysisReport(report);
      setNotice('Word analysis report downloaded.');
    } catch (exportError) {
      setError(exportError.message || 'Could not create the Word report.');
    } finally {
      setExportingReport('');
    }
  };

  const copyHash = async () => {
    try {
      await navigator.clipboard.writeText(result.file_hash);
      setNotice('SHA-256 hash copied to clipboard.');
    } catch {
      setNotice('Clipboard access is unavailable. Select and copy the SHA-256 value manually.');
    }
  };

  const openReport = (report) => {
    setResult(report);
    setActiveResultTab('Result');
    setSelectedFile(null);
    setReviewOutcome(report.analyst_review?.outcome || 'inconclusive');
    setReviewRationale(report.analyst_review?.rationale || '');
    setReviewerName(report.analyst_review?.reviewer || report.investigator || 'Analyst');
    setView('result');
    setActiveNav(report.media_type === 'image' ? 'image-analysis' : `${report.media_type}-analysis`);
  };

  const reanalyzeReport = async (report) => {
    if (!report?.case_id) return;
    setReanalyzingReport(report.case_id);
    setError('');
    setNotice('');
    try {
      const response = await fetch(`${API_BASE}/api/reports/${encodeURIComponent(report.case_id)}/reanalyze`, { method: 'POST' });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `Reanalysis failed with status ${response.status}.`);
      setResult(payload);
      setActiveResultTab('Result');
      setReviewOutcome(payload.analyst_review?.outcome || 'inconclusive');
      setReviewRationale(payload.analyst_review?.rationale || '');
      setReviewerName(payload.analyst_review?.reviewer || payload.investigator || 'Analyst');
      setView('result');
      setActiveNav(payload.media_type === 'image' ? 'image-analysis' : `${payload.media_type}-analysis`);
      setNotice('Saved evidence was reanalyzed. Its SHA-256 was verified before decoding.');
      await refreshRecords();
    } catch (requestError) {
      setError(describeRequestError(requestError, 'Could not reanalyze the saved evidence.'));
    } finally {
      setReanalyzingReport('');
    }
  };

  const titleByView = {
    overview: ['Workspace', 'Investigation overview'],
    analysis: ['Evidence intake', 'Start a new analysis'],
    result: ['Image analysis', 'Evidence analysis results'],
    cases: ['Workspace', 'Case timeline'],
    reports: ['Workspace', 'Forensic reports'],
  };
  const [eyebrow, pageTitle] = titleByView[view];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand-block" href="#overview" onClick={(event) => { event.preventDefault(); setView('overview'); }}>
          <span className="brand-mark"><span /></span>
          <span className="brand-copy">
            <strong>deeptrace<span>.</span></strong>
            <small>MEDIA FORENSICS</small>
          </span>
        </a>

        <div className="workspace-label">FORENSICS WORKSPACE</div>
        <nav className="nav-menu" aria-label="Analysis navigation">
          {NAV_ITEMS.map((item) => (
            <button
              className={`nav-link ${activeNav === item.id ? 'active' : ''}`}
              key={item.id}
              onClick={() => {
                setActiveNav(item.id);
                setView(item.view);
                if (item.mode) setMediaMode(item.mode);
                setError('');
              }}
              aria-current={activeNav === item.id ? 'page' : undefined}
            >
              <Icon name={item.icon} />
              <span>{item.label}</span>
              {item.id === 'reports' && reports.length > 0 && <span className="nav-count">{reports.length}</span>}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="support-card">
            <div className="support-icon"><Icon name="shield" size={17} /></div>
            <strong>Evidence, not verdicts.</strong>
            <p>AI indicators support review. They do not establish authenticity.</p>
          </div>
          <div className="profile-row">
            <div className="avatar">A</div>
            <div><strong>Analyst workspace</strong><small>Local session</small></div>
            <span className={`online-dot ${apiStatus}`} title={apiStatus === 'online' ? 'API connected' : 'API unavailable'} />
          </div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="header-brand"><span className="brand-mark"><span /></span><strong>DeepTrace <em>AI</em></strong><small>TRUST WHAT YOU SEE</small></div>
          <nav className="top-navigation" aria-label="Workspace navigation">
            {[
              ['analysis', 'Analyze'],
              ['cases', 'History'],
              ['analysis', 'Datasets'],
              ['cases', 'Models'],
              ['reports', 'Reports'],
            ].map(([target, label], index) => (
              <button
                className={`top-nav-link ${((label === 'Analyze' && (view === 'analysis' || view === 'result')) || (label === 'History' && view === 'cases') || (label === 'Reports' && view === 'reports')) ? 'active' : ''}`}
                key={`${label}-${index}`}
                onClick={() => {
                  setView(target);
                  setActiveNav(label === 'Reports' ? 'reports' : label === 'History' ? 'cases' : target === 'analysis' ? `${mediaMode}-analysis` : 'model-comparison');
                }}
              >{label}</button>
            ))}
          </nav>
          <div className="header-tools">
            <label className="search-field"><Icon name="search" size={14} /><input placeholder="Search analyses…" aria-label="Search analyses" /></label>
            <button className="header-icon" aria-label="Notifications"><span className="notification-dot" /><Icon name="bell" size={17} /></button>
            <div className="header-user"><span className="avatar">A</span><span className={`online-dot ${apiStatus}`} /></div>
          </div>
        </header>

        <div className="page-content">
          {view === 'result' && result && (
            <>
              {error && <div className="error-box" role="alert">{error}</div>}
              {notice && <div className="notice-box"><Icon name="check" size={17} />{notice}<button onClick={() => setNotice('')} aria-label="Dismiss message">×</button></div>}
              <AnalysisResultDashboard
                report={result}
                previewUrl={result.filename === selectedFile?.name ? previewUrl : ''}
                activeTab={activeResultTab}
                setActiveTab={setActiveResultTab}
                onCopyHash={copyHash}
                onPreviewReport={() => setPreviewReport(result)}
                onExportReport={() => exportWordReport(result)}
                onReanalyze={() => reanalyzeReport(result)}
                reanalyzing={reanalyzingReport === result.case_id}
                exporting={exportingReport === (result.case_id || result.filename)}
              >
                <section className="forensic-card result-review-card">
                  <div className="forensic-card-heading"><div><span className="section-kicker">HUMAN REVIEW · SEPARATE FROM MODEL OUTPUT</span><h3>Analyst interpretation</h3></div><span className="evidence-pill muted-pill">{result.analyst_review ? 'Review recorded' : 'Awaiting review'}</span></div>
                  <p>Document comparison with trusted source material, chain-of-custody context, independent observations, and remaining uncertainty. This is a human conclusion—not an AI prediction.</p>
                  <form className="review-form" onSubmit={saveAnalystReview}>
                    {result.analyst_review && <div className="saved-review"><strong>Saved reviewer assessment: {result.analyst_review.outcome.replaceAll('_', ' ')}</strong><small>{result.analyst_review.reviewer} · {formatDate(result.analyst_review.reviewed_at)}</small><p>{result.analyst_review.rationale}</p></div>}
                    <label>Reviewer conclusion
                      <select value={reviewOutcome} onChange={(event) => setReviewOutcome(event.target.value)}>
                        <option value="inconclusive">Inconclusive / insufficient evidence</option>
                        <option value="likely_original">Likely original (human assessment)</option>
                        <option value="potentially_manipulated">Potentially manipulated / deepfake (human assessment)</option>
                      </select>
                    </label>
                    <label>Reviewer name<input value={reviewerName} onChange={(event) => setReviewerName(event.target.value)} minLength={2} maxLength={120} required /></label>
                    <label>Rationale and corroborating context<textarea value={reviewRationale} onChange={(event) => setReviewRationale(event.target.value)} minLength={12} maxLength={2000} placeholder="Describe the trusted source, chain-of-custody context, independent evidence, uncertainty and follow-up needed." required /></label>
                    <button type="submit" className="button button-primary" disabled={reviewSaving}>{reviewSaving ? 'Saving interpretation…' : 'Save analyst interpretation'}</button>
                  </form>
                </section>
              </AnalysisResultDashboard>
            </>
          )}

          {view === 'overview' && (
            <>
              <section className="hero-banner">
                <div className="hero-copy">
                  <span className="section-kicker">DEEPFAKE DETECTION &amp; MEDIA FORENSICS AI</span>
                  <h1>Analyze. Detect. <span>Explain.</span><br />Ensure authenticity.</h1>
                  <p>Multimodal evidence review with clear provenance and analyst oversight.</p>
                  <div className="hero-actions">
                    <button className="button button-primary" onClick={() => { setView('analysis'); setActiveNav(`${mediaMode}-analysis`); setError(''); }}>
                      <Icon name="scan" size={17} /> Upload &amp; analyze
                    </button>
                    <button className="button button-secondary" onClick={() => { setView('analysis'); setActiveNav('image-analysis'); setMediaMode('image'); loadExample(SAMPLE_EXAMPLES[1]); }}>
                      <Icon name="image" size={16} /> Load deepfake-style example
                    </button>
                  </div>
                </div>
                <div className="hero-visual" aria-hidden="true">
                  <span className="hero-orbit orbit-one" /><span className="hero-orbit orbit-two" />
                  <div className="hero-signal signal-one">SPATIAL<span>NOT CONFIGURED</span></div>
                  <div className="hero-signal signal-two">METADATA<span>AVAILABLE</span></div>
                  <div className="hero-signal signal-three">TEMPORAL<span>NOT CONFIGURED</span></div>
                  <div className="hero-core"><Icon name="scan" size={42} /></div>
                </div>
              </section>

              {notice && <div className="notice-box"><Icon name="check" size={17} />{notice}<button onClick={() => setNotice('')} aria-label="Dismiss message">×</button></div>}

              <section className="metric-grid" aria-label="Workspace metrics">
                {metrics.map((metric) => (
                  <article className="metric-card" key={metric.label}>
                    <div className={`metric-icon ${metric.tone}`}><Icon name={metric.icon} size={19} /></div>
                    <div className="metric-copy"><span>{metric.label}</span><strong>{metric.value}</strong><small>{metric.note}</small></div>
                    <span className={`metric-accent ${metric.tone}`} />
                  </article>
                ))}
              </section>

              <section className="dashboard-grid">
                <article className="panel recent-panel">
                  <div className="panel-heading">
                    <div><span className="section-kicker">LATEST EVIDENCE</span><h2>Recent analyses</h2></div>
                    <button className="text-button" onClick={() => setView('reports')}>View reports <Icon name="chevron" size={14} /></button>
                  </div>
                  {reports.length ? (
                    <div className="table-wrap">
                      <table>
                        <thead><tr><th>Evidence</th><th>Type</th><th>Finding</th><th>Analyzed</th><th /></tr></thead>
                        <tbody>
                          {reports.slice().reverse().slice(0, 5).map((report, index) => (
                            <tr key={`${report.case_id}-${index}`}>
                              <td><div className="evidence-name"><span className="file-icon"><Icon name="file" size={16} /></span><span><strong>{report.filename}</strong><small>{report.case_id}</small></span></div></td>
                              <td className="capitalize">{report.media_type || '—'}</td>
                              <td><span className={`verdict-badge ${String(report.verdict).toLowerCase().replaceAll(' ', '-')}`}>{report.verdict}</span></td>
                              <td>{formatDate(report.created_at)}</td>
                              <td><button className="icon-button" aria-label={`Open report for ${report.filename}`} onClick={() => openReport(report)}><Icon name="chevron" size={16} /></button></td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="empty-state">
                      <div className="empty-illustration"><Icon name="scan" size={26} /></div>
                      <strong>Your evidence workspace is ready</strong>
                      <p>Start an analysis to create a report and build your case history.</p>
                      <button className="button button-secondary" onClick={() => setView('analysis')}>Add evidence <Icon name="arrow" size={15} /></button>
                    </div>
                  )}
                </article>

                <aside className="panel activity-panel latest-panel">
                  <div className="panel-heading"><div><span className="section-kicker">LATEST ANALYSIS</span><h2>{result ? 'Current assessment' : 'System readiness'}</h2></div></div>
                  {result ? (
                    <>
                      <span className="verdict-badge inconclusive">{result.verdict}</span>
                      <h3 className="latest-filename">{result.filename}</h3>
                      <p className="latest-summary">Original vs deepfake: <strong>not determined</strong></p>
                      <p className="latest-summary">{result.decision_basis}</p>
                      <button className="text-button" onClick={() => setView('reports')}>Review all reports <Icon name="chevron" size={14} /></button>
                    </>
                  ) : (
                    <>
                      <div className="model-readiness"><span className="readiness-indicator" /><strong>Metadata pipeline available</strong><small>SHA-256 and container review enabled</small></div>
                      <div className="model-readiness unavailable"><span className="readiness-indicator" /><strong>Deepfake classifier unavailable</strong><small>Install a trained, validated model for prediction</small></div>
                      <p className="prototype-note-text">Until then, the system reports <b>insufficient evidence</b> rather than guessing original or deepfake.</p>
                      <button className="text-button" onClick={() => setPreviewReport(SAMPLE_REPORT_PREVIEW)}>Preview sample Word report <Icon name="chevron" size={14} /></button>
                    </>
                  )}
                </aside>
              </section>

              <section className="panel workflow-detail-panel">
                <div className="panel-heading"><div><span className="section-kicker">FORENSIC WORKFLOW</span><h2>Four-stage evidence review</h2></div><span className="step-tag">{result ? 'REPORT DETAILS' : 'ANALYZE TO POPULATE'}</span></div>
                <div className="workflow-detail-grid">
                  <article className="workflow-stage">
                    <div className="stage-title"><span>01</span><div><strong>Evidence integrity</strong><small>SHA-256 fingerprinting</small></div></div>
                    <p>Hash the received bytes and inspect the filename extension against the binary signature. A hash proves only that later bytes match this upload; without a trusted reference hash, it does not prove where the file came from or whether it is original.</p>
                    {result ? <dl className="stage-facts">
                      <div><dt>SHA-256</dt><dd className="hash-value">{result.file_hash}</dd></div>
                      <div><dt>Size</dt><dd>{formatBytes(result.file_size)}</dd></div>
                      <div><dt>File signature</dt><dd>{result.file_integrity?.signature || 'Not available'}</dd></div>
                      <div><dt>Extension check</dt><dd>{result.file_integrity?.extension_matches_signature ? 'Matches signature' : 'Review required'}</dd></div>
                      <div><dt>Content check</dt><dd>{String(result.file_integrity?.content_validation || 'not performed').replaceAll('_', ' ')}</dd></div>
                    </dl> : <div className="stage-empty">Hash, file signature and byte size appear here after upload.</div>}
                  </article>
                  <article className="workflow-stage">
                    <div className="stage-title"><span>02</span><div><strong>Metadata review</strong><small>Container and file properties</small></div></div>
                    <p>Inspect available format, dimensions or stream properties, EXIF fields and software tags. Metadata may be missing, stripped or edited; interpret it alongside source records rather than as a standalone manipulation signal.</p>
                    {result ? <div className="stage-findings">{(result.metadata_findings || []).map((item) => <div key={item.signal}><strong>{item.signal}</strong><small>{item.interpretation}</small></div>)}</div> : <div className="stage-empty">Container, codec, dimensions and provenance fields are reviewed when the format supports them.</div>}
                  </article>
                  <article className="workflow-stage">
                    <div className="stage-title"><span>03</span><div><strong>Signal assessment</strong><small>Model status &amp; measured evidence</small></div></div>
                    <p>Spatial, temporal, frequency, compression, audio and synchronization signals should be scored by validated models. Unavailable modalities are marked not run—metadata heuristics are not substituted for model predictions.</p>
                    {result ? <div className="stage-findings">{(result.evidence_assessment || []).map((item) => <div className="assessment-item" key={item.name}><span className={`assessment-status ${item.status.replaceAll('_', '-')}`}>{item.status.replaceAll('_', ' ')}</span><strong>{item.name}</strong><small>{item.interpretation}</small></div>)}</div> : <div className="stage-empty">This prototype has no trained classifier. No fake/real probabilities or signal scores will be invented.</div>}
                  </article>
                  <article className="workflow-stage analyst-stage">
                    <div className="stage-title"><span>04</span><div><strong>Analyst interpretation</strong><small>Context before conclusions</small></div></div>
                    <p>Compare with the trusted source, document chain-of-custody gaps, check independent evidence, and record a reasoned human assessment. The following is a reviewer opinion, not an AI prediction.</p>
                    {result ? (
                      <form className="review-form" onSubmit={saveAnalystReview}>
                        {result.analyst_review && <div className="saved-review"><strong>Saved reviewer assessment: {result.analyst_review.outcome.replaceAll('_', ' ')}</strong><small>{result.analyst_review.reviewer} · {formatDate(result.analyst_review.reviewed_at)}</small><p>{result.analyst_review.rationale}</p></div>}
                        <label>Reviewer conclusion
                          <select value={reviewOutcome} onChange={(event) => setReviewOutcome(event.target.value)}>
                            <option value="inconclusive">Inconclusive / insufficient evidence</option>
                            <option value="likely_original">Likely original (human assessment)</option>
                            <option value="potentially_manipulated">Potentially manipulated / deepfake (human assessment)</option>
                          </select>
                        </label>
                        <label>Reviewer name<input value={reviewerName} onChange={(event) => setReviewerName(event.target.value)} minLength={2} maxLength={120} required /></label>
                        <label>Rationale and corroborating context<textarea value={reviewRationale} onChange={(event) => setReviewRationale(event.target.value)} minLength={12} maxLength={2000} placeholder="Describe what you compared, source/chain-of-custody context, observed evidence, uncertainty and follow-up needed." required /></label>
                        <button type="submit" className="button button-primary" disabled={reviewSaving}>{reviewSaving ? 'Saving interpretation…' : 'Save analyst interpretation'}</button>
                      </form>
                    ) : <div className="stage-empty">After analysis, record an original-like, potentially manipulated or inconclusive reviewer conclusion with its rationale.</div>}
                  </article>
                </div>
              </section>
            </>
          )}

          {view === 'analysis' && (
            <>
              <section className="page-intro">
                <div><span className="section-kicker">EVIDENCE INTAKE</span><h1>Analyze a media file</h1><p>Preserve the original file and add the context needed for a traceable review.</p></div>
              </section>
              {error && <div className="error-box" role="alert">{error}</div>}
              {mediaMode === 'image' && <TwoImageComparison />}
              <section className="intake-layout">
                <article className="panel intake-panel">
                  <div className="panel-heading"><div><span className="section-kicker">NEW INVESTIGATION</span><h2>Evidence details</h2></div><span className="step-tag">STEP 01 / 01</span></div>
                  <form onSubmit={handleUpload} className="upload-form">
                    <div className="form-grid">
                      <label>Case title<input value={caseTitle} onChange={(event) => setCaseTitle(event.target.value)} placeholder="e.g. Source verification — Case 14" /></label>
                      <label>Investigator<input value={investigator} onChange={(event) => setInvestigator(event.target.value)} placeholder="Name or team" /></label>
                    </div>
                    <label className={`drop-zone ${dragActive ? 'drag-active' : ''} ${selectedFile ? 'has-file' : ''}`}
                      onDragOver={(event) => { event.preventDefault(); setDragActive(true); }}
                      onDragLeave={() => setDragActive(false)}
                      onDrop={(event) => { event.preventDefault(); setDragActive(false); selectFile(event.dataTransfer.files[0]); }}>
                      <input type="file" accept={ACCEPTED_TYPES} onChange={(event) => selectFile(event.target.files[0])} />
                      {selectedFile ? (
                        <div className="selected-file">
                          {previewUrl ? <img src={previewUrl} alt="Selected evidence preview" /> : <span className="file-icon large"><Icon name="file" size={22} /></span>}
                          <div><strong>{selectedFile.name}</strong><small>{formatBytes(selectedFile.size)} · {selectedFile.type || 'Media file'}</small></div>
                          <button type="button" className="remove-file" aria-label="Remove selected file" onClick={(event) => { event.preventDefault(); setSelectedFile(null); }}>×</button>
                        </div>
                      ) : (
                        <div className="drop-content"><span className="upload-icon"><Icon name="upload" size={22} /></span><strong>Drop evidence here, or <em>browse files</em></strong><small>Images, video, or audio · Preserve your original evidence</small></div>
                      )}
                    </label>
                    <div className="file-types"><span>SUPPORTED FORMATS</span><div><span>JPG · PNG · GIF · WEBP · BMP · TIFF</span><span>MP4 · M4V · MOV · AVI · MKV · WEBM</span><span>MP3 · WAV · FLAC · M4A · AAC</span></div></div>
                    <section className="sample-gallery" aria-labelledby="sample-gallery-title">
                      <div className="sample-gallery-heading">
                        <div><span className="section-kicker">QUICK DEMO</span><h3 id="sample-gallery-title">Try an example</h3></div>
                        <span>PNG · 800 × 560</span>
                      </div>
                      <div className="sample-cards">
                        {SAMPLE_EXAMPLES.map((example) => (
                          <article className="sample-card" key={example.id}>
                            <img src={`/examples/${example.filename}`} alt={`${example.title}: synthetic illustrated example`} />
                            <div className="sample-card-content">
                              <span className={`sample-label ${example.id}`}>{example.label}</span>
                              <strong>{example.title}</strong>
                              <p>{example.description}</p>
                              <div className="sample-actions">
                                <button type="button" className="text-button" disabled={loadingExample === example.id} onClick={() => loadExample(example)}>
                                  {loadingExample === example.id ? 'Loading…' : 'Load for analysis'}
                                </button>
                                <a className="sample-download" href={`/examples/${example.filename}`} download={example.filename}>Download</a>
                              </div>
                            </div>
                          </article>
                        ))}
                      </div>
                      <p className="sample-disclaimer">All examples are project-generated synthetic illustrations, not genuine media or forensic ground-truth samples. The prototype skips classifier inference on these demo files.</p>
                    </section>
                    <section className="research-gallery" aria-labelledby="research-gallery-title">
                      <div className="sample-gallery-heading">
                        <div><span className="section-kicker">RESEARCH DATASETS</span><h3 id="research-gallery-title">Live official previews</h3></div>
                        <span>External sources</span>
                      </div>
                      <div className="research-cards">
                        {DATASET_PREVIEWS.map((dataset) => (
                          <article className="research-card" key={dataset.id}>
                            {dataset.image ? (
                              <a className="research-preview" href={dataset.source} target="_blank" rel="noreferrer" aria-label={`Open official ${dataset.name} project page`}>
                                <img src={dataset.image} alt={dataset.imageAlt} loading="lazy" referrerPolicy="no-referrer" />
                                <span>OFFICIAL PREVIEW</span>
                              </a>
                            ) : (
                              <div className="research-preview research-gated-preview">
                                <Icon name="shield" size={24} />
                                <strong>Access-controlled</strong>
                                <small>Examples require approved dataset access</small>
                              </div>
                            )}
                            <div className="research-card-content">
                              <strong>{dataset.name}</strong>
                              <p>{dataset.description}</p>
                              <div>
                                <a href={dataset.source} target="_blank" rel="noreferrer">Project details</a>
                                <a href={dataset.accessUrl} target="_blank" rel="noreferrer">{dataset.access}</a>
                              </div>
                            </div>
                          </article>
                        ))}
                      </div>
                      <p className="sample-disclaimer">Previews are loaded live from the dataset maintainers and are not bundled or uploaded for analysis. These datasets require access approval or account setup; comply with their terms before using any downloaded media. DFDC access requires an AWS account. The FaceForensics++ and Celeb-DF maintainers distribute data by request.</p>
                    </section>
                    <div className="privacy-note"><Icon name="shield" size={17} /><p><strong>Evidence integrity</strong><br />The API calculates a SHA-256 fingerprint during intake. Uploads are stored locally by the demo backend.</p></div>
                    <button type="submit" className="button button-primary submit-button" disabled={loading || apiStatus === 'offline'}>
                      {loading ? <><span className="spinner" /> Analyzing evidence…</> : <>Run forensic analysis <Icon name="arrow" size={16} /></>}
                    </button>
                    {apiStatus === 'offline' && <small className="api-hint">Start the FastAPI backend to enable analysis.</small>}
                  </form>
                </article>
                <aside className="panel process-panel">
                  <span className="section-kicker">WHAT HAPPENS NEXT</span><h2>Analysis workflow</h2>
                  <div className="process-steps">
                    <div><span>01</span><p><strong>Integrity check</strong><small>Identify file and calculate hash</small></p></div>
                    <div><span>02</span><p><strong>Metadata inspection</strong><small>Read available media properties</small></p></div>
                    <div><span>03</span><p><strong>Signal assessment</strong><small>Model status is reported; no predictions are fabricated</small></p></div>
                    <div><span>04</span><p><strong>Forensic summary</strong><small>Open and export the review report</small></p></div>
                  </div>
                  <div className="caution-card"><Icon name="shield" size={17} /><p><strong>Use careful language</strong><br />An indicator is not proof. Verify findings with independent evidence and qualified human review.</p></div>
                </aside>
              </section>
            </>
          )}

          {view === 'cases' && (
            <section className="panel records-panel">
              <div className="panel-heading"><div><span className="section-kicker">INVESTIGATION HISTORY</span><h2>Case timeline</h2></div><button className="button button-secondary" onClick={() => setView('analysis')}><Icon name="scan" size={16} /> New analysis</button></div>
              <div className="records-toolbar">
                <input aria-label="Search cases" placeholder="Search case, investigator, or ID…" value={recordQuery} onChange={(event) => setRecordQuery(event.target.value)} />
                <select aria-label="Filter cases by status" value={caseStatusFilter} onChange={(event) => setCaseStatusFilter(event.target.value)}><option value="all">All statuses</option><option value="new">New</option><option value="awaiting_review">Awaiting review</option><option value="reviewed">Reviewed</option></select>
                <button className="button button-secondary" onClick={downloadAuditHistory}>Export audit CSV</button>
              </div>
              {filteredCases.length ? (
                <div className="table-wrap"><table><thead><tr><th>Case</th><th>Investigator</th><th>Status</th><th>Created</th><th>Report</th></tr></thead><tbody>
                  {filteredCases.slice().reverse().map((item) => {
                    const caseReport = reports.find((report) => report.case_id === item.case_id);
                    return <tr key={item.case_id}><td><div className="evidence-name"><span className="file-icon"><Icon name="folder" size={16} /></span><span><strong>{item.title}</strong><small>{item.case_id}</small></span></div></td><td>{item.investigator || '—'}</td><td><span className="case-status">{item.status || 'open'}</span></td><td>{formatDate(item.created_at)}</td><td>{caseReport ? <button className="text-button" onClick={() => openReport(caseReport)}>Open report</button> : '—'}</td></tr>;
                  })}
                </tbody></table></div>
              ) : cases.length ? <div className="empty-state"><strong>No cases match these filters</strong><p>Change the search or status filter to see more records.</p></div> : <div className="empty-state"><div className="empty-illustration"><Icon name="folder" size={26} /></div><strong>No saved cases yet</strong><p>Case records appear here after evidence is analyzed.</p><button className="button button-secondary" onClick={() => setView('analysis')}>Start an analysis</button></div>}
              <p className="session-note">Case history is stored in the local SQLite database. Keep the database and original evidence files backed up together.</p>
            </section>
          )}

          {view === 'reports' && (
            <section className="panel records-panel">
              <div className="panel-heading"><div><span className="section-kicker">EVIDENCE REVIEW</span><h2>Forensic reports</h2></div><div className="row-actions"><span className="step-tag">{filteredReports.length} / {reports.length} REPORT{reports.length === 1 ? '' : 'S'}</span><button className="button button-secondary" onClick={downloadAllReports} disabled={!reports.length}>Export all JSON</button></div></div>
              <div className="records-toolbar">
                <input aria-label="Search reports" placeholder="Search filename, case, hash, investigator…" value={recordQuery} onChange={(event) => setRecordQuery(event.target.value)} />
                <select aria-label="Filter reports by media" value={recordMediaType} onChange={(event) => setRecordMediaType(event.target.value)}><option value="all">All media</option><option value="image">Images</option><option value="video">Videos</option><option value="audio">Audio</option></select>
                <select aria-label="Filter reports by finding" value={reportVerdictFilter} onChange={(event) => setReportVerdictFilter(event.target.value)}><option value="all">All findings</option><option value="Insufficient evidence">Insufficient evidence</option><option value="Likely original">Likely original</option><option value="Likely deepfake">Likely deepfake</option><option value="Suspicious">Suspicious</option></select>
                <button className="button button-secondary" onClick={downloadAuditHistory}>Export audit CSV</button>
              </div>
              {filteredReports.length ? <div className="table-wrap"><table><thead><tr><th>Evidence</th><th>Finding</th><th>Model status</th><th>Analyzed</th><th>Actions</th></tr></thead><tbody>
                {filteredReports.slice().reverse().map((report) => <tr key={report.case_id}><td><div className="evidence-name"><span className="file-icon"><Icon name="file" size={16} /></span><span><strong>{report.filename}</strong><small>{report.case_id} · {report.media_type}</small></span></div></td><td><span className={`verdict-badge ${String(report.verdict).toLowerCase().replaceAll(' ', '-')}`}>{report.verdict}</span></td><td>{String(report.decision_status || 'unknown').replaceAll('_', ' ')}</td><td>{formatDate(report.created_at)}</td><td><div className="row-actions"><button className="text-button" onClick={() => openReport(report)}>Open</button><button className="text-button" onClick={() => setPreviewReport(report)}>Preview</button><button className="text-button" disabled={reanalyzingReport === report.case_id} onClick={() => reanalyzeReport(report)}>{reanalyzingReport === report.case_id ? 'Running…' : 'Reanalyze'}</button><button className="text-button word-export" disabled={exportingReport === (report.case_id || report.filename)} onClick={() => exportWordReport(report)}>{exportingReport === (report.case_id || report.filename) ? 'Creating…' : 'Word'}</button><button className="icon-button" aria-label={`Download JSON report for ${report.filename}`} onClick={() => downloadReport(report)}><Icon name="download" size={16} /></button></div></td></tr>)}
              </tbody></table></div> : reports.length ? <div className="empty-state"><strong>No reports match these filters</strong><p>Change the search, media type, or finding filter.</p></div> : <div className="empty-state"><div className="empty-illustration"><Icon name="file" size={26} /></div><strong>Reports will appear here</strong><p>Run an analysis to generate a structured forensic summary.</p><button className="button button-secondary" onClick={() => setView('analysis')}>Analyze media</button></div>}
              <p className="session-note">Reports and audit events are stored in local SQLite. Export a JSON archive and CSV audit trail for a portable copy.</p>
            </section>
          )}

          {view === 'overview' && result && (
            <section className="panel result-panel">
              <div className="panel-heading">
                <div><span className="section-kicker">SELECTED REPORT</span><h2>Forensic summary</h2></div>
                <div className="row-actions">
                  <button className="button button-secondary" onClick={copyHash}><Icon name="copy" size={15} /> Copy SHA-256</button>
                  <button className="button button-secondary" onClick={() => setPreviewReport(result)}>Preview report</button>
                  <button className="button button-secondary" disabled={exportingReport === (result.case_id || result.filename)} onClick={() => exportWordReport(result)}>{exportingReport === (result.case_id || result.filename) ? 'Creating Word…' : 'Export Word .docx'}</button>
                  <button className="icon-button" aria-label="Download JSON report" onClick={() => downloadReport(result)}><Icon name="download" size={17} /></button>
                </div>
              </div>
              <div className="result-overview">
                <div className="result-verdict">
                  <span className={`verdict-badge ${String(result.verdict).toLowerCase().replaceAll(' ', '-')}`}>{result.verdict}</span>
                  <h3>{result.filename}</h3>
                  <p>{result.summary}</p>
                  <div className="result-tags"><span>{result.media_type}</span><span>{result.manipulation_type}</span><span>{result.decision_status?.replaceAll('_', ' ')}</span></div>
                </div>
                <div className="decision-callout"><span>ORIGINAL VS DEEPFAKE</span><strong>{result.model_results?.xception ? result.verdict : 'Not determined'}</strong><small>{result.decision_basis || 'A validated classifier is not configured.'}</small></div>
              </div>
              <div className="probability-grid">
                <div><span>{result.model_results?.xception?.status === 'uncalibrated_prediction' ? 'Authenticity softmax score' : 'Authenticity probability'}</span><strong>{formatProbability(result.authenticity_probability)}</strong><small>{result.model_results?.xception ? (result.model_results.xception.status === 'uncalibrated_prediction' ? 'Uncalibrated raw score; not per-file confidence' : 'Temperature-scaled model output; not per-file confidence') : 'No trained binary classifier configured'}</small></div>
                <div><span>{result.model_results?.xception?.status === 'uncalibrated_prediction' ? 'Deepfake softmax score' : 'Deepfake probability'}</span><strong>{formatProbability(result.deepfake_probability)}</strong><small>{result.model_results?.xception ? (result.model_results.xception.status === 'uncalibrated_prediction' ? 'Uncalibrated raw score; domain shift may affect reliability' : 'Temperature-scaled model output; domain shift may affect reliability') : 'No model inference was performed'}</small></div>
                <div><span>Model confidence</span><strong>Not provided</strong><small>A calibrated probability is not a guarantee of correctness</small></div>
              </div>
              <div className="report-lower-grid">
                <div><h3>Signal assessment</h3><ul className="assessment-list">{(result.evidence_assessment || []).map((item) => <li key={item.name}><span className={`assessment-status ${item.status.replaceAll('_', '-')}`}>{item.status.replaceAll('_', ' ')}</span><div><strong>{item.name}</strong><small>{item.interpretation}</small></div></li>)}</ul></div>
                <div><h3>Container &amp; metadata</h3><dl className="metadata-list">
                  {Object.entries(result.file_integrity || {}).filter(([key]) => !['sha256', 'size_bytes', 'header_hex', 'meaning'].includes(key)).filter(([, value]) => value !== null && value !== undefined && value !== '').map(([key, value]) => <div key={`integrity-${key}`}><dt>{key.replaceAll('_', ' ')}</dt><dd>{formatMetadataValue(value, key)}</dd></div>)}
                  {Object.entries(result.metadata_summary || {}).filter(([, value]) => value !== null && value !== undefined && value !== '' && value !== 'unknown').map(([key, value]) => <div key={`metadata-${key}`}><dt>{key.replaceAll('_', ' ')}</dt><dd>{formatMetadataValue(value, key)}</dd></div>)}
                </dl></div>
              </div>
              {!!result.metadata_findings?.length && <section className="metadata-findings"><h3>Metadata findings &amp; interpretation</h3>{result.metadata_findings.map((finding) => <article key={finding.signal}><span>{finding.severity.replaceAll('_', ' ')}</span><div><strong>{finding.signal}</strong><p>{finding.interpretation}</p></div></article>)}</section>}
              <div className="heatmap-disclaimer">
                <Icon name="shield" size={20} />
                <p><strong>Important interpretation</strong><br />{result.model_results?.xception
                  ? 'Xception provides model decision support for static images only. It does not establish provenance, guarantee this individual classification, or localize manipulated pixels. Review face-crop details and dataset/domain limitations.'
                  : 'This prototype does not run a trained detector. It has not concluded that this file is original or a deepfake. A missing metadata field or a SHA-256 fingerprint alone cannot determine authenticity.'}</p>
              </div>
              <div className="explanation-box"><h3>Detailed forensic notes</h3>{(result.explanation || []).map((line, index) => <p key={`${index}-${line}`}>{line}</p>)}</div>
              <div className="hash-row"><span>SHA-256</span><code>{result.file_hash}</code><button className="icon-button" aria-label="Copy SHA-256 hash" onClick={copyHash}><Icon name="copy" size={15} /></button></div>
            </section>
          )}

          <footer className="footer-note"><span>DEEPTRACE AI</span><span>Research prototype · Forensic decision support only</span></footer>
        </div>
      </main>
      {previewReport && <ReportPreview report={previewReport} onClose={() => setPreviewReport(null)} onDownload={exportWordReport} downloading={exportingReport === (previewReport.case_id || previewReport.filename)} />}
    </div>
  );
}

export default App;
