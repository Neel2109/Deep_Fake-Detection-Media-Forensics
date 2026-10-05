const COLORS = {
  ink: '17212B',
  muted: '5D6875',
  green: '527B36',
  paleGreen: 'EEF5E8',
  paleBlue: 'F1F4F8',
  border: 'D9E0E7',
  amber: '8E641C',
  red: '9E3D43',
};

let docx;

const safe = (value, fallback = 'Not available') => (
  value === undefined || value === null || value === '' ? fallback : String(value)
);

const probabilityText = (value) => (
  Number.isFinite(value) ? `${(value * 100).toFixed(1)}% (raw ${value.toFixed(6)})` : 'Not available'
);

const metricText = (value) => (
  Number.isFinite(value) ? value.toFixed(3) : 'n/a'
);

const benchmarkText = (metrics) => metrics
  ? `n=${safe(metrics.samples, 'n/a')} · ROC-AUC ${metricText(metrics.roc_auc)} · PR-AUC/AP ${metricText(metrics.pr_auc)} · balanced accuracy ${metricText(metrics.balanced_accuracy_at_0_5)} · F1 ${metricText(metrics.f1_at_0_5)} · Brier ${metricText(metrics.brier_score)} · ECE ${metricText(metrics.expected_calibration_error)}`
  : 'No independent test metrics recorded';

const calibrationText = (before, after) => (
  before && after
    ? `Before scaling: Brier ${metricText(before.brier_score)}, ECE ${metricText(before.expected_calibration_error)}; after scaling: Brier ${metricText(after.brier_score)}, ECE ${metricText(after.expected_calibration_error)}`
    : 'No held-out calibration diagnostics recorded'
);

const labelText = (value) => String(value).replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
const propertyValue = (key, value) => {
  if (['status', 'content_validation'].includes(key) && typeof value === 'string') {
    return value.replaceAll('_', ' ');
  }
  return typeof value === 'object' ? JSON.stringify(value) : safe(value);
};

const text = (value, options = {}) => new docx.TextRun({
  text: safe(value),
  font: 'Aptos',
  size: 20,
  color: COLORS.ink,
  ...options,
});

const paragraph = (value, options = {}) => new docx.Paragraph({
  children: [text(value)],
  spacing: { after: 110 },
  ...options,
});

const heading = (value, level = docx.HeadingLevel.HEADING_2) => new docx.Paragraph({
  text: value,
  heading: level,
  spacing: { before: 250, after: 100 },
  keepNext: true,
});

const cell = (value, options = {}) => new docx.TableCell({
  children: [new docx.Paragraph({
    children: [text(value, {
      bold: options.bold || false,
      color: options.color || COLORS.ink,
      size: options.size || 19,
    })],
    spacing: { before: 35, after: 35 },
  })],
  shading: options.shading ? { fill: options.shading } : undefined,
  margins: { top: 95, bottom: 95, left: 115, right: 115 },
  borders: {
    top: { style: docx.BorderStyle.SINGLE, size: 4, color: COLORS.border },
    bottom: { style: docx.BorderStyle.SINGLE, size: 4, color: COLORS.border },
    left: { style: docx.BorderStyle.SINGLE, size: 4, color: COLORS.border },
    right: { style: docx.BorderStyle.SINGLE, size: 4, color: COLORS.border },
  },
});

const makeTable = (rows, widths) => new docx.Table({
  width: { size: 100, type: docx.WidthType.PERCENTAGE },
  columnWidths: widths,
  rows: rows.map((row, rowIndex) => new docx.TableRow({
    children: row.map((value, columnIndex) => cell(value, {
      bold: rowIndex === 0,
      color: rowIndex === 0 ? 'FFFFFF' : COLORS.ink,
      shading: rowIndex === 0 ? COLORS.ink : (rowIndex % 2 === 0 ? COLORS.paleBlue : 'FFFFFF'),
      size: rowIndex === 0 ? 18 : 19,
    })),
  })),
});

export async function createAnalysisReportDocument(report) {
  docx = await import('docx');
  const xception = report.model_results?.xception;
  const explainability = xception?.explainability;
  const uncalibrated = xception?.status === 'uncalibrated_prediction';
  const evidenceScores = Object.entries(report.evidence_scores || {});
  const regions = report.suspicious_regions || [];
  const assessments = report.evidence_assessment || [];
  const videoAnalysis = report.video_analysis;
  const videoFrames = videoAnalysis?.frames || [];
  const videoFrameAnalysis = videoAnalysis?.model_frame_analysis;
  const chainOfCustody = report.chain_of_custody || {};
  const audioAnalysis = report.audio_analysis;
  const audioMeasurements = audioAnalysis?.measurements;
  const descriptiveMeasurements = [
    ...(report.face_review
      ? [['Face-candidate review', report.face_review.status === 'unavailable'
        ? 'Detector unavailable; no candidate count was computed.'
        : `${safe(report.face_review.detected_face_count, '0')} candidates · ${safe(report.face_review.status)}`]]
      : []),
    ...Object.entries(report.frequency_analysis?.measurements || {})
      .map(([key, value]) => [`FFT · ${labelText(key)}`, safe(value)]),
    ...Object.entries(report.noise_analysis?.measurements || {})
      .map(([key, value]) => [`High-pass residual · ${labelText(key)}`, safe(value)]),
  ];
  const pipelineStages = report.pipeline_stages || [];
  const modelPipeline = report.model_pipeline || [];
  const integrity = Object.entries(report.file_integrity || {})
    .filter(([key, value]) => !['sha256', 'size_bytes', 'header_hex', 'meaning'].includes(key)
      && value !== undefined && value !== null && value !== '');
  const metadata = Object.entries(report.metadata_summary || {})
    .filter(([, value]) => value !== undefined && value !== null && value !== '' && value !== 'unknown');
  const audioSpectrumBands = audioMeasurements?.spectrogram_summary?.band_power || [];
  const verdict = safe(report.verdict, 'Unclassified');
  const verdictColor = ['authentic', 'likely original'].includes(verdict.toLowerCase())
    ? COLORS.green
    : ['manipulated', 'likely deepfake'].includes(verdict.toLowerCase()) ? COLORS.red : COLORS.amber;

  const children = [
    new docx.Paragraph({
      children: [text('DEEPTRACE AI  /  MEDIA FORENSICS', { bold: true, size: 18, color: COLORS.green })],
      spacing: { after: 130 },
    }),
    new docx.Paragraph({
      children: [text('Forensic Analysis Report', { bold: true, size: 38, color: COLORS.ink })],
      spacing: { after: 90 },
    }),
    new docx.Paragraph({
      children: [text('Evidence-based decision support · Research prototype', { size: 21, color: COLORS.muted })],
      spacing: { after: 280 },
    }),
    makeTable([
      ['CASE ID', safe(report.case_id), 'ANALYSIS DATE', safe(report.created_at)],
      ['CASE TITLE', safe(report.case_title), 'INVESTIGATOR', safe(report.investigator)],
      ['EVIDENCE FILE', safe(report.filename), 'MEDIA TYPE', safe(report.media_type)],
    ], [1800, 3600, 1800, 3600]),
    heading('Executive summary'),
    new docx.Paragraph({
      children: [
        text(`${verdict}  `, { bold: true, color: verdictColor, size: 23 }),
        text(`${xception
          ? `Automated assessment: ${verdict}`
          : report.evidence_context?.is_synthetic_demo
            ? 'Synthetic demo sample — classifier intentionally skipped'
            : 'Verdict withheld — no trained classifier configured'}  ·  Classifier status: ${safe(report.decision_status, 'unknown').replaceAll('_', ' ')}`, { bold: true, size: 20 }),
      ],
      shading: { fill: 'FFF7E8' },
      spacing: { before: 30, after: 120 },
      indent: { left: 120, right: 120 },
    }),
    paragraph(report.summary),
    ...(report.evidence_context?.is_synthetic_demo
      ? [paragraph(`SYNTHETIC DEMONSTRATION SAMPLE — NOT GROUND TRUTH: ${safe(report.evidence_context.notice)}`, { shading: { fill: 'F3EEFF' } })]
      : []),
    paragraph(`Decision basis: ${safe(report.decision_basis)}`),
    paragraph(`Automated authenticity ${uncalibrated ? 'softmax score' : 'probability'}: ${probabilityText(report.authenticity_probability)}`),
    paragraph(`Automated deepfake ${uncalibrated ? 'softmax score' : 'probability'}: ${probabilityText(report.deepfake_probability)}`),
    paragraph(xception
      ? uncalibrated
        ? 'The softmax scores are uncalibrated model outputs, not validated probabilities of authenticity. The displayed class is the model’s higher-scoring class only.'
        : 'Per-file model confidence is not separately calibrated; a calibrated probability is not a guarantee of correctness.'
      : 'Model confidence was not computed because classifier inference did not run.'),
    ...(xception
      ? [
        heading('Xception model assessment'),
        makeTable([
          ['Model property', 'Recorded value'],
          ['Architecture', safe(xception.model.architecture)],
          ...(xception.model.inference_device ? [['Inference device', xception.model.inference_device]] : []),
          ['Validation', xception.model.reported_validation_accuracy || (Number.isFinite(xception.model.validation_auc) ? `${Number(xception.model.validation_auc).toFixed(4)} ROC-AUC (validation ranking metric, not per-file confidence)` : 'No independently verified validation result')],
          ['Calibration', `${safe(xception.model.calibration)}${Number.isFinite(xception.model.calibration_temperature) ? `; temperature ${Number(xception.model.calibration_temperature).toFixed(4)}` : ''}`],
          ['Operating thresholds', JSON.stringify(xception.model.thresholds)],
          ...(xception.conformal_prediction
            ? [[
              'Conformal prediction set',
              `${xception.conformal_prediction.prediction_set.join(', ') || 'Empty'} · alpha ${safe(xception.conformal_prediction.alpha)} · ${xception.conformal_prediction.abstained ? 'decision withheld' : 'threshold class supported'}; coverage assumes exchangeability.`,
            ]]
            : []),
          ['Face review', JSON.stringify(xception.face_review)],
          ...(xception.model.input_preprocessing ? [['Model input', xception.model.input_preprocessing]] : []),
          ...(xception.model.training_data ? [['Training-data scope', xception.model.training_data]] : []),
          ...(xception.model.calibration_metrics_after_scaling
            ? [['Calibration diagnostics', calibrationText(
              xception.model.calibration_metrics_before_scaling,
              xception.model.calibration_metrics_after_scaling,
            )]]
            : []),
          ...(xception.model.test_metrics
            ? [['Independent test benchmark', `${benchmarkText(xception.model.test_metrics)} · Dataset-level, not per-file confidence.`]]
            : []),
          ...Object.entries(xception.model.test_metrics_by_source || {}).map(([source, metrics]) => [
            `Test source · ${source}`,
            `${benchmarkText(metrics)} · ${metrics.source_not_seen_in_train_validation_or_calibration ? 'Source held out from model development.' : 'Source also appears in model-development splits.'}`,
          ]),
          ...(xception.model.dataset_audit?.group_overlap_status
            ? [['Group-leakage audit', `${safe(xception.model.dataset_audit.group_overlap_status).replaceAll('_', ' ')} · ${safe(xception.model.dataset_audit.manifest_sha256, 'No audit digest')}`]]
            : []),
        ], [3200, 6800]),
        ...xception.limitations.map((limitation) => paragraph(`Limitation: ${limitation}`)),
      ]
      : []),
    ...(explainability?.status === 'generated' && explainability.overlay_data_url
      ? [
        heading('Model attention / evidence visualization'),
        paragraph(`Grad-CAM tells us which regions contributed most to the model's prediction. It does not independently establish that those pixels were manipulated. Target: ${safe(explainability.target)} · layer: ${safe(explainability.target_layer)}.`),
        new docx.Paragraph({
          children: [new docx.ImageRun({
            data: Uint8Array.from(
              atob(explainability.overlay_data_url.split(',')[1]),
              (character) => character.charCodeAt(0),
            ),
            transformation: {
              width: Math.min(480, explainability.overlay_width),
              height: Math.min(480, explainability.overlay_width)
                * explainability.overlay_height / explainability.overlay_width,
            },
          })],
          spacing: { after: 120 },
        }),
        paragraph('This coarse model explanation highlights image regions that influenced the fake-class logit. It is not a manipulation mask, pixel-level localization, or proof of authenticity.'),
      ]
      : []),
    heading('Classifier pipeline execution trace'),
    ...(modelPipeline.length
      ? [makeTable([
        ['Stage', 'Status', 'Execution detail'],
        ...modelPipeline.map((stage) => [
          `${stage.number}. ${safe(stage.name)}`,
          safe(stage.status).replaceAll('_', ' '),
          safe(stage.detail),
        ]),
      ], [2800, 2100, 5100])]
      : [paragraph('Detailed model pipeline execution was not recorded in this older report. Reanalyze the retained evidence to capture per-stage statuses.')]),
    heading('Four-stage evidence review'),
    ...(pipelineStages.length
      ? [makeTable([
        ['Stage', 'Status', 'Review outcome'],
        ...pipelineStages.map((stage) => [`${stage.number}. ${stage.name}`, safe(stage.status).replaceAll('_', ' '), safe(stage.summary)]),
      ], [2600, 2400, 5000])]
      : []),
    heading('Evidence integrity — SHA-256 fingerprint'),
    paragraph(`SHA-256: ${safe(report.file_hash)}`),
    paragraph(`Byte size: ${safe(report.file_size)} bytes`),
    paragraph(`Tamper-evident audit chain: ${safe(chainOfCustody.verification_status, 'not sealed')} · head SHA-256 ${safe(chainOfCustody.head_sha256)}`),
    ...(chainOfCustody.storage_limitation
      ? [paragraph(chainOfCustody.storage_limitation)]
      : []),
    ...(integrity.length
      ? [makeTable([
        ['File integrity property', 'Observed value'],
        ...integrity.map(([key, value]) => [labelText(key), propertyValue(key, value)]),
      ], [4200, 6600])]
      : []),
    heading('Metadata review — container and file properties'),
    ...(report.metadata_findings?.length
      ? [makeTable([
        ['Metadata finding', 'Interpretation', 'Type'],
        ...report.metadata_findings.map((finding) => [safe(finding.signal), safe(finding.interpretation), safe(finding.severity)]),
      ], [2800, 5800, 2200])]
      : [paragraph('No metadata-specific findings were recorded.')]),
    paragraph('Metadata is contextual evidence. Missing or editable metadata does not prove manipulation.'),
    heading('Evidence indicators'),
    ...(assessments.length
      ? [makeTable([
        ['Signal / modality', 'Status', 'Assessment'],
        ...assessments.map((assessment) => [safe(assessment.name), safe(assessment.status).replaceAll('_', ' '), safe(assessment.interpretation)]),
      ], [2700, 2100, 6000])]
      : evidenceScores.length
      ? [makeTable([
        ['Signal', 'Score'],
        ...evidenceScores.map(([name, score]) => [labelText(name), `${safe(score)}%`]),
      ], [7000, 3800])]
      : [paragraph('No validated model signal scores were provided.')]),
    heading('Descriptive image measurements — not detector scores'),
    ...(descriptiveMeasurements.length
      ? [
        makeTable([
          ['Measurement', 'Observed value'],
          ...descriptiveMeasurements,
        ], [4200, 6600]),
        paragraph(safe(report.frequency_analysis?.interpretation || report.noise_analysis?.interpretation,
          'Face-candidate, spectrum, and residual measurements describe decoded image properties only. They are not validated deepfake indicators or authenticity scores.')),
      ]
      : [paragraph('No static-image face, FFT, or high-pass residual measurements were recorded for this media type.')]),
    ...(videoAnalysis
      ? [
        heading('Video keyframe review — descriptive only'),
        makeTable([
          ['Video property', 'Observed value'],
          ['Decoder frame rate', `${safe(videoAnalysis.decoder_reported_fps)} FPS`],
          ['Container frame count', safe(videoAnalysis.container_reported_frame_count)],
          ['Estimated duration', `${safe(videoAnalysis.duration_estimate_seconds)} seconds`],
          ['Keyframes decoded', `${safe(videoAnalysis.sampled_frame_count, String(videoFrames.length))} of ${safe(videoAnalysis.requested_frame_count)}`],
          ['Sampling strategy', safe(videoAnalysis.sampling_strategy)],
          ['Maximum preview dimension', `${safe(videoAnalysis.max_frame_side)} pixels`],
          ...(videoFrameAnalysis
            ? [['Frame-model analysis', `${safe(videoFrameAnalysis.analyzed_frame_count, '0')} sampled frames · ${safe(videoFrameAnalysis.architecture)} · ${safe(videoFrameAnalysis.calibration)}`]]
            : []),
        ], [4200, 6600]),
        ...(videoFrames.length
          ? [makeTable([
            ['Frame', 'Timestamp', 'Dimensions', 'Face review / Xception estimate'],
            ...videoFrames.map((frame) => [
              safe(frame.index + 1),
              `${safe(frame.timestamp_seconds)} seconds`,
              `${safe(frame.width)} × ${safe(frame.height)}`,
              `${frame.face_review?.status === 'unavailable'
                ? 'Detector unavailable; no count computed'
                : `${safe(frame.face_review?.detected_face_count, '0')} candidates · ${safe(frame.face_review?.status)}`}${frame.xception_estimate ? `; Xception deepfake score ${metricText(frame.xception_estimate.deepfake_score)} (${safe(frame.xception_estimate.calibration)})` : ''}`,
            ]),
          ], [1400, 2200, 2200, 5000])]
          : []),
        paragraph(`${safe(videoAnalysis.interpretation)} No temporal manipulation classifier or clip-level verdict was produced.`),
      ]
      : []),
    ...(audioAnalysis
      ? [
        heading('Audio signal review — descriptive only'),
        makeTable([
          ['Audio property', 'Observed value'],
          ['Container', safe(audioAnalysis.container)],
          ['Codec MIME', safe(audioAnalysis.codec_mime)],
          ['Duration from container', `${safe(audioAnalysis.duration_seconds)} seconds`],
          ['Sample rate', `${safe(audioAnalysis.sample_rate_hz)} Hz`],
          ['Channels', safe(audioAnalysis.channels)],
          ['Signal scope', `First ${safe(audioAnalysis.analysis_limit_seconds, '0')} seconds maximum`],
          ['Decoded duration', `${safe(audioMeasurements?.analysis_duration_seconds)} seconds`],
          ['Decoded samples', safe(audioMeasurements?.sample_count_analyzed)],
          ['Mean absolute amplitude', safe(audioMeasurements?.mean_absolute_amplitude)],
          ['Zero-crossing rate', safe(audioMeasurements?.zero_crossing_rate)],
          ['RMS amplitude', safe(audioMeasurements?.rms_amplitude)],
          ['Peak amplitude', safe(audioMeasurements?.peak_amplitude)],
          ['Clipped-sample fraction', probabilityText(audioMeasurements?.clipped_sample_fraction)],
          ['FFT windows / size / hop', audioMeasurements
            ? `${safe(audioMeasurements.spectrogram_summary.window_count)} / ${safe(audioMeasurements.spectrogram_summary.fft_size)} / ${safe(audioMeasurements.spectrogram_summary.hop_size)}`
            : 'Not available'],
          ['Power-weighted spectral centroid', `${safe(audioMeasurements?.spectrogram_summary?.power_weighted_spectral_centroid_hz)} Hz`],
          ['Spectral flatness', safe(audioMeasurements?.spectrogram_summary?.spectral_flatness)],
        ], [4200, 6600]),
        ...(audioSpectrumBands.length
          ? [makeTable([
            ['Equal-width FFT band', 'Mean power'],
            ...audioSpectrumBands.map((power, index) => [`Band ${index + 1}`, safe(power)]),
          ], [5000, 5800])]
          : []),
        paragraph(`${safe(audioAnalysis.interpretation)} No synthetic-voice or audio-video synchronization classifier was run.`),
      ]
      : []),
    heading('Region-level indicators'),
    ...(regions.length
      ? [makeTable([
        ['Region', 'Indicator score', 'Risk label'],
        ...regions.map((region) => [
          safe(region.label),
          `${safe(region.score)}%`,
          safe(region.risk),
        ]),
      ], [3600, 3600, 3600])]
      : [paragraph('No region-level indicators were provided.')]),
    heading('Media metadata'),
    ...(metadata.length
      ? [makeTable([
        ['Property', 'Value'],
        ...metadata.map(([key, value]) => [labelText(key), propertyValue(key, value)]),
      ], [3800, 7000])]
      : [paragraph('No media metadata was provided.')]),
    ...(Array.isArray(report.explanation) && report.explanation.length
      ? [heading('Analysis notes'), ...report.explanation.map((note) => paragraph(`• ${note}`))]
      : []),
    ...(report.analyst_review
      ? [
        heading('Analyst interpretation — human assessment'),
        paragraph(`Reviewer conclusion: ${safe(report.analyst_review.outcome).replaceAll('_', ' ')}`),
        paragraph(`Reviewer: ${safe(report.analyst_review.reviewer)}`),
        paragraph(`Reviewed at: ${safe(report.analyst_review.reviewed_at)}`),
        paragraph(`Rationale: ${safe(report.analyst_review.rationale)}`),
        paragraph('This is a recorded human assessment, not an automated model output.'),
      ]
      : [
        heading('Analyst interpretation — pending'),
        paragraph('No reviewer conclusion has been recorded. Compare to trusted source material, examine chain of custody, document independent evidence, then record a reasoned opinion.'),
      ]),
    ...(Array.isArray(report.audit_log) && report.audit_log.length
      ? [heading('Audit trail'), makeTable([
        ['Timestamp (UTC)', 'Event type', 'Event', 'Event SHA-256'],
        ...report.audit_log.map((entry) => [
          safe(entry.timestamp),
          safe(entry.event_type, 'legacy'),
          safe(entry.message),
          safe(entry.event_sha256),
        ]),
      ], [2000, 1900, 3500, 3000])]
      : []),
    heading('Interpretation and limitations'),
    new docx.Paragraph({
      children: [text(
        xception
          ? uncalibrated
            ? 'The Xception classifier provides raw, uncalibrated softmax scores and a higher-scoring class, not a calibrated probability or proof. Its training scope is FFHQ/StyleGAN faces; performance may change under domain shift. File hashes identify byte identity only when compared with a trusted reference; metadata can be absent or edited. Corroborate independently and have a qualified analyst review the original media.'
            : xception.status === 'conformal_abstention'
              ? 'The calibrated Xception model withheld its class because the conformal prediction set did not uniquely support the threshold-selected label. Coverage assumes exchangeability with calibration data and is not guaranteed under domain shift. The score is not proof or a per-file confidence guarantee.'
              : 'The Xception classifier provides calibrated model decision support for this static image; it does not prove authenticity or manipulation and does not provide per-file confidence or pixel-level localization. Its estimate may not generalize under dataset or domain shift. File hashes identify byte identity only when compared with a trusted reference; metadata can be absent or edited. Corroborate independently and have a qualified analyst review the original media.'
          : 'The current system does not have a trained and validated authenticity classifier. It therefore withholds the original-versus-deepfake verdict, probabilities, and model confidence. File hashes verify byte identity only when compared with a trusted reference; metadata can be absent or edited. Do not treat this report as proof of authenticity or manipulation. Corroborate evidence independently and have a qualified analyst review the original media.',
        { bold: true, color: COLORS.amber, size: 19 },
      )],
      shading: { fill: 'FFF7E8' },
      spacing: { before: 50, after: 150 },
      indent: { left: 120, right: 120 },
    }),
    new docx.Paragraph({
      children: [text('Generated by DeepTrace AI. Preserve the original evidence and verify chain of custody independently.', { size: 16, color: COLORS.muted })],
      spacing: { before: 180 },
    }),
  ];

  const document = new docx.Document({
    creator: 'DeepTrace AI',
    title: `Forensic Analysis Report - ${safe(report.case_id, 'case')}`,
    subject: 'Media forensics decision-support report',
    description: 'Generated report from the DeepTrace AI research prototype.',
    styles: {
      default: {
        document: { run: { font: 'Aptos', size: 20, color: COLORS.ink } },
        heading1: { run: { font: 'Aptos Display', size: 32, bold: true, color: COLORS.ink } },
        heading2: { run: { font: 'Aptos Display', size: 25, bold: true, color: COLORS.ink } },
      },
    },
    sections: [{
      properties: {
        page: {
          margin: { top: 850, right: 900, bottom: 850, left: 900 },
        },
      },
      children,
      footers: {
        default: new docx.Footer({
          children: [new docx.Paragraph({
            alignment: docx.AlignmentType.CENTER,
            children: [text('DEEPTRACE AI · RESEARCH PROTOTYPE · NOT A STANDALONE FORENSIC CONCLUSION', { size: 14, color: COLORS.muted })],
          })],
        }),
      },
    }],
  });

  return document;
}

export async function downloadAnalysisReport(report) {
  const document = await createAnalysisReportDocument(report);
  const blob = await docx.Packer.toBlob(document);
  const url = URL.createObjectURL(blob);
  const link = window.document.createElement('a');
  link.href = url;
  link.download = `${safe(report.case_id, 'deeptrace-analysis-report').replace(/[<>:"/\\|?*]/g, '-')}-analysis-report.docx`;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
