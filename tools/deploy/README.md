# VisionHub1 Offline Deployment Tools

Two-step offline deployment for machines without internet access.

## Step 1: Pack (on a PC with internet)

```cmd
tools\deploy\pack_offline.bat
```

This downloads Python installer + all wheel packages + copies source code into a `VisionHub1_offline\` folder.

## Step 2: Install (on the target PC, no internet needed)

Copy the `VisionHub1_offline\` folder to the target machine, then run:

```cmd
install.bat
```

This installs Python (if needed), creates a virtual environment, installs all packages offline, and creates startup scripts.

## Output Structure

```
VisionHub1_offline\
    python-3.10.11-amd64.exe    # Python installer
    packages\                    # All .whl files
    VisionHub1\                  # Source code
    install.bat                  # One-click installer
```

## Notes

- CUDA Toolkit must be installed separately if GPU acceleration is needed
- Project data (`E:\AIInspect\projects\`) must be copied separately
- Default CUDA version: cu118. Edit `pack_offline.bat` to change
