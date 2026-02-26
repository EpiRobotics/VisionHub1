# VisionHub Python AI Service

Industrial Visual Inspection Unified Platform - Python AI Backend Service.

## Quick Start

### Prerequisites

- Python 3.10+
- Poetry

### Installation

```bash
cd python_service
poetry install
```

### Configuration

1. Create your data root directory (default: `E:\AIInspect`)
2. Copy `config_templates/service_config.yaml` to `<DATA_ROOT>/service/service_config.yaml`
3. For each project, create `<DATA_ROOT>/projects/<project_id>/project.yaml` from `config_templates/project_example.yaml`

### Run

```bash
# Set data root (optional, defaults to E:\AIInspect)
export VISIONHUB_DATA_ROOT=/path/to/data

# Run the service
poetry run visionhub
```

The HTTP API will be available at `http://localhost:8100` by default.
Each project's TCP server will run on its configured port.

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Service health check |
| GET | /projects | List all projects |
| GET | /projects/{id} | Get project details |
| POST | /projects/reload | Reload all project configs |
| POST | /projects/{id}/enable | Enable/disable project |
| POST | /projects/{id}/train | Start training |
| GET | /train/{job_id} | Get training status |
| POST | /projects/{id}/infer | Test inference (HTTP) |
| GET | /projects/{id}/stats | Get project statistics |

### TCP Protocol

Each project listens on its configured TCP port. Protocol: JSON Lines (one JSON per line, `\n` terminated).

**Commands:**
- `INFER` - Run inference: `{"cmd":"INFER","job_id":"001","image_path":"path/to/img.jpg"}`
- `PING` - Health check
- `STATUS` - Get project status
- `SET_ACTIVE_MODEL` - Switch model version: `{"cmd":"SET_ACTIVE_MODEL","version":"20260226_153000"}`
