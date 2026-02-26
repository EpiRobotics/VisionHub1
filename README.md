# VisionHub - Industrial Visual Inspection Unified Platform

A unified platform for industrial appearance inspection and character detection. One software manages multiple "projects" (surfaces/stations/tasks).

## Architecture

```
+-------------------+          HTTP (FastAPI)          +-------------------+
|   C# WinForms UI  | <-----------------------------> |  Python AI Service |
|   (Telerik)       |                                  |  (常驻进程)        |
+-------------------+                                  +-------------------+
                                                              |
+-------------------+     TCP (JSONL per project port)        |
| External Software | <-------------------------------------->+
| (Vision System)   |                                  |
+-------------------+                                  |  GPU Inference
                                                       |  (PyTorch)
                                                       +-------------------+
```

## Components

### Python AI Service (`python_service/`)
- Project management with hot-reload
- Algorithm plugin system (PatchCore, PaDiM, CNN, etc.)
- Per-project TCP JSONL servers for external integration
- FastAPI HTTP API for UI management
- GPU inference with job queue and scheduling
- Model cache with LRU eviction

### C# WinForms UI (`csharp_ui/`)
- Project list management (add/remove/enable/disable)
- Training control (start/stop/progress/logs)
- Test inference with result visualization
- Real-time monitoring (throughput, latency, queue)
- Service health monitoring

## Quick Start

### 1. Setup Data Directory

Create your data root (default: `E:\AIInspect`):

```
E:\AIInspect\
  service\
    service_config.yaml    (copy from config_templates/)
  projects\
    <project_id>\
      project.yaml         (copy from config_templates/)
      datasets\ok\         (training images)
      models\              (trained models)
```

### 2. Start Python Service

```bash
cd python_service
poetry install
poetry run visionhub
```

### 3. Open C# UI

Open `csharp_ui/VisionHubUI.sln` in Visual Studio and run.

## Directory Structure

```
VisionHub/
  config_templates/           # Template config files
    service_config.yaml
    project_example.yaml
  python_service/             # Python AI backend
    app/
      main.py                 # Entry point & service wiring
      config.py               # Configuration models
      project_manager.py      # Project lifecycle management
      scheduler.py            # Job queue & GPU worker pool
      tcp_server.py           # TCP JSONL servers
      http_api.py             # FastAPI HTTP endpoints
      train_manager.py        # Training job management
      result_schema.py        # Result/error schemas
      plugins/
        base.py               # AlgoPluginBase interface
        registry.py           # Plugin registry
        patchcore_plugin.py   # PatchCore stub (real algo TBD)
      utils/
        file_utils.py         # File readiness checks
        logging_utils.py      # Logging helpers
  csharp_ui/                  # C# WinForms management UI
    VisionHubUI.sln
    VisionHubUI/
      Program.cs
      MainForm.cs
      Models/                 # Data models
      Services/               # API client
```

## Communication Protocols

### TCP (External Software)
- JSON Lines protocol (one JSON per line, `\n` terminated)
- Each project on its own port (configured in project.yaml)
- Commands: INFER, PING, STATUS, SET_ACTIVE_MODEL

### HTTP (UI Management)
- FastAPI on port 8100 (configurable)
- RESTful endpoints for project management, training, monitoring
