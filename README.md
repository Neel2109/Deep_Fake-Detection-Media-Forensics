# DeepTrace AI: Media Forensics Prototype

DeepTrace AI is a local web application for organizing media-forensics reviews. It combines a React dashboard with a FastAPI service for evidence intake, file-integrity checks, metadata inspection, descriptive signal measurements, case/report storage, and analyst review.

This repository is a research and workflow prototype, not a production deepfake detector. Most image, video, and audio authenticity classifiers and the proposed multimodal fusion pipeline are not configured. The application intentionally withholds a verdict when it has no supported, configured model. See [Current capabilities and limitations](#current-capabilities-and-limitations) before interpreting results.

## Contents

- [Features](#features)
- [Technology](#technology)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [API overview](#api-overview)
- [Tests](#tests)
- [Current capabilities and limitations](#current-capabilities-and-limitations)
- [Dataset audit and model training](#dataset-audit-and-calibrated-training)
- [Project layout](#project-layout)

## Features

- Upload supported image, video, and audio files for a structured review.
- Check file signatures, extension consistency, size, and SHA-256 fingerprints; inspect image metadata and selected EXIF fields.
- Review available image properties, video keyframes, and descriptive audio measurements without presenting these as authenticity proof.
- Compare two still images using exact byte hashes, equal-size pixel differences, metadata/face-candidate observations, and separate model assessments where available.
- Create cases and reports, record analyst review, retrieve stored evidence, export audit events to CSV, and verify a hash-linked audit-event sequence.
- Preview example images and export a report document from the dashboard.
- Optionally run a locally available Xception checkpoint for still-image assessment and a bounded Grad-CAM visualization.

## Technology

- Frontend: React 18, Vite 5, JavaScript.
- Backend: Python, FastAPI, Uvicorn, SQLite.
- Media processing: Pillow, NumPy, OpenCV, SoundFile, and Mutagen.
- Optional model runtime: PyTorch, Torchvision, timm, and SciPy (see `requirements-ml.txt`).

## Quick start

Prerequisites: Python 3.11 or newer and Node.js with npm. Python 3.12 is a practical default. Install the optional ML requirements only if you intend to use the image model; PyTorch installation may depend on your operating system and accelerator.

### 1. Install dependencies

From the repository root, create and activate a virtual environment, then install the backend and frontend dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
npm ci
```

If PowerShell prevents virtual-environment activation, run the interpreter directly as `.\.venv\Scripts\python.exe` in the commands below.

### 2. Configure the backend

Copy `.env.example` to `.env` and adjust values if needed. Defaults support the local Vite development server. See [Configuration](#configuration).

### 3. Start the application

Start the API and dashboard in separate terminals from the repository root:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

```powershell
npm run dev
```

Open <http://localhost:5173>. The API health check is at <http://localhost:8000/api/health>, and interactive API documentation is at <http://localhost:8000/docs>.

The optional launcher opens both services in separate PowerShell windows:

```powershell
npm run dev:all
```

It uses `.backend-venv` when present and otherwise `.venv` for the backend interpreter.

## Configuration

Backend settings can be provided in `.env` or as environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_TITLE` | `DeepTrace AI` | FastAPI application title. |
| `APP_VERSION` | `0.1.0` | FastAPI application version. |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated allowed browser origins. |
| `DEEPTRACE_DATABASE_PATH` | `data/deeptrace.sqlite3` | Local SQLite database location. |
| `MAX_UPLOAD_BYTES` | `524288000` | Maximum size of an individual uploaded file (500 MiB). |
| `XCEPTION_CHECKPOINT` | `weights/xception_deepfake.pth` | Optional local image-model checkpoint. |

The frontend API URL defaults to `http://localhost:8000`. To point it elsewhere, set `VITE_API_BASE` when starting/building Vite, for example `VITE_API_BASE=http://127.0.0.1:8000` in the frontend environment.

Local databases, uploaded evidence, generated reports, environment files, virtual environments, frontend build output, and model weight files are intended to stay out of source control. Review `.gitignore` before adding local data or model artifacts.

## API overview

All routes are under `/api`; request and response schemas are described in the interactive [OpenAPI documentation](http://localhost:8000/docs).

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service health check. |
| `POST` | `/auth/register`, `/auth/login` | Demonstration-only authentication responses. |
| `GET`, `POST` | `/cases` | List and create case records. |
| `POST` | `/analyze` | Upload and inspect one supported media file. |
| `POST` | `/compare-images` | Compare two still images. |
| `GET` | `/reports` | List stored analysis reports. |
| `GET` | `/reports/{report_id}` | Retrieve one report. |
| `POST` | `/reports/{report_id}/reanalyze` | Re-run analysis on stored evidence. |
| `GET` | `/reports/{report_id}/evidence` | Retrieve report evidence when permitted by the route's evidence token. |
| `PATCH` | `/reports/{report_id}/review` | Append an analyst review. |
| `GET` | `/reports/audit.csv` | Export report audit events. |

The API stores case and report JSON payloads in local SQLite. Evidence files are kept under `data/uploads/`; this prototype does not provide production-grade identity, authorization, multi-user isolation, or remote storage.

## Tests

Install the test runner if it is not already available, then run the test suite from the repository root:

```powershell
python -m pip install pytest
python -m pytest
```

Tests that exercise optional model behavior may require `requirements-ml.txt`, model weights, or additional local fixtures. The public Xception weights are not committed to this repository.

## Current capabilities and limitations

Implemented review steps must not be mistaken for a validated deepfake verdict. File hashes identify bytes; metadata, compression observations, image differences, and heatmaps do not independently prove provenance or manipulation. The authentication routes return demo responses and are not an identity or access-control system. Treat uploaded media as sensitive and run this prototype only in a trusted environment.

The application is not a substitute for validated forensic tools, source verification, chain-of-custody procedures, or a qualified analyst. Its local SQLite audit chain is tamper-evident, not immutable, remotely anchored, or digitally signed. Further implementation-specific constraints are described below.

## Project layout

| Path | Responsibility |
| --- | --- |
| `src/` | React dashboard and frontend styles. |
| `app/` | FastAPI application, API routes, configuration, schemas, and media-analysis services. |
| `api/`, `audio/`, `video/`, `image/`, `forensics/`, `frequency/`, `fusion/`, `explainability/` | Domain modules and analysis components; some are extension points rather than active classifiers. |
| `database/` | SQLite connection, persistence, and schema code. |
| `training/` | Dataset audit and model-training utilities. |
| `tests/` | Backend and analysis test suite. |
| `weights/` | Local model checkpoints; not committed. |
| `data/` | Local database, uploaded evidence, and generated reports; not committed. |

## Backend architecture

This directory contains the existing FastAPI demo and a modular extension
scaffold. Run from this directory:

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

`main.py` is a compatibility import for `app.main:app`. The current API routes
and in-memory case/report behavior remain in `app/`. Video keyframe extraction
and audio signal measurements are implemented as descriptive review only.
The EXIF analyzer extracts selected tags and validates GPS coordinates, and
`POST /api/compare-images` returns a two-image comparison report. Other modules
under `api/`, `database/`, `video/`, `audio/`, `frequency/`, `fusion/`, and
related packages may remain extension points; remaining scaffold-only modules
must not be used to generate authenticity scores.

## Future Upgrades and Major Milestones

For DeepTrace AI, the next major upgrade should be to combine the current descriptive steps into a fused evidence pipeline:

**Xception prediction → Grad-CAM++ → face/region localization → FFT/DCT evidence → compression analysis → metadata → ensemble models → calibrated confidence → final forensic assessment**

### Important Forensic Interpretation

When using the heatmap, it should be presented as **Model Attention / Evidence Visualization** rather than a "DeepFake Manipulation Boundary".
Grad-CAM tells us **which regions contributed most to the model's prediction**. It does **not** independently establish that those pixels were manipulated.

The image classifier contracts are in `app/services/xception_detector.py` and
`training/train_xception.py`. Static-image inference supports the pinned public
two-class Xception checkpoint at `weights/xception_deepfake.pth` as well as a
locally trained and calibrated checkpoint at that path. The current public
checkpoint is locally downloaded, SHA-256 verified, and ignored by source
control; see the root README for its exact revision and restore command. Its
softmax scores are explicitly uncalibrated and its training domain is limited
to FFHQ real faces and StyleGAN-generated fake faces. Video-frame inference can
run this image model on at most eight evenly spaced, downscaled keyframes; each
score remains an image estimate, is not calibrated for video, and is never
aggregated into a clip verdict. Static-image Xception inference can generate a
bounded Grad-CAM overlay for the fake-class logit; the coarse map is not pixel-
level manipulation localization. Temporal/video-level models, audio authenticity,
multimodal fusion, frequency-CNN, other-model Grad-CAM, alternative model architectures, authentication,
persistent relational user/authentication models, background workers, and
related components remain unconfigured. Cases and analysis reports are stored
as JSON payloads in local SQLite at `data/deeptrace.sqlite3` (override with
`DEEPTRACE_DATABASE_PATH`); uploaded evidence remains under `data/uploads/`.
The public checkpoint is not independently validated on this project's data;
its estimate must not be treated as proof or used as sole evidence.

The comparison endpoint accepts two still-image uploads as `image_a` and
`image_b`, checks size and file signatures, rejects animated images, then
reports exact byte-hash equality, direct pixel differences for equal-sized
images, metadata and face-candidate observations, and the two separate model
assessments. If both calibrated model decisions conflict, the report marks a
qualified model-supported contrast but does not claim proof or identify source
authenticity. If either image lacks a supported class, the pair conclusion is
withheld. Face observations are non-identifying detections; the endpoint does
not identify people or infer personal attributes.

## Dataset audit and calibrated training

Training requires four separate folders: `train/`, `validation/`, `calibration/`,
and `test/`, each with `real/` and `fake/` subfolders. Calibration and test
require at least 20 images per class. It also requires a complete CSV group/
provenance manifest with `path,group_id,source,license` columns. Paths are
relative to the dataset root; person/source-media groups must not cross splits.
`source` identifies the dataset or collection and `license` records the rights
basis. The audit cannot verify those claims or discover transformed
near-duplicates from file hashes.

Run before training:

```powershell
python -m training.dataset_audit --data-root ..\dataset --group-manifest dataset-groups.csv
```

The audit validates decodability, hashes, class counts, exact duplicates, and
source-group overlap, and writes a per-file JSON manifest. Training repeats the
audit, uses validation only for model selection, calibration only for
temperature and threshold fitting, and test only for final benchmark reporting.
The checkpoint records ROC-AUC, PR-AUC, balanced accuracy, precision, recall,
specificity, F1, confusion matrix, Brier score, 10-bin expected calibration
error, and calibration-derived threshold coverage/abstention. Calibration
metrics are not independent; test results remain scoped to the supplied data
and do not establish cross-domain performance. Test metrics are reported per
manifest `source` as well as overall, and the report marks sources absent from
training, validation, and calibration. The calibration split also fits
class-conditional split-conformal prediction sets at alpha 0.10. A likely-real
or likely-fake decision is withheld when the set does not uniquely support the
threshold-selected class. Coverage interpretation assumes exchangeability with
calibration examples and is not guaranteed under distribution shift.

## Audio signal review

Audio inspection reads available container properties and decodes up to the
first 30 seconds for descriptive waveform/amplitude and coarse FFT measurements,
including zero-crossing rate, power-weighted spectral centroid, and spectral
flatness. These signal properties are not voice-cloning indicators. No trained
synthetic-speech detector, speaker embedding model, transcription, or
audio-video synchronization classifier is configured; authenticity verdicts
remain withheld.

## Evidence audit chain

Each analysis report hash-links its audit events with SHA-256, binding the
event sequence to the case ID and evidence file hash. Case association, analyst
review, and reanalysis append chained events; report retrieval verifies the
chain and reports tampering. The chain is stored alongside reports in mutable
local SQLite, so it is tamper-evident, not immutable, remotely anchored, or
digitally signed.
