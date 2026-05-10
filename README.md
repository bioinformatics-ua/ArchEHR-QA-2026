# ArchEHR-QA-2026

This repository contains the ArchEHR-QA-2026 Shared Task codebase for Subtask 1 through Subtask 4.

## Overview

Each `SubtaskX/` directory contains the code, scripts, and data layout for that specific shared task:
- `Subtask1/`
- `Subtask2/`
- `Subtask3/`
- `Subtask4/`

## Environment

This repo uses `uv` for Python environment management in the subtask folders.

- `Subtask1/` uses separate inference and evaluation environments
  - inference: Python 3.13
  - evaluation: Python 3.8
- `Subtask2/` uses Python 3.10+ for both inference and evaluation
- `Subtask3/` and `Subtask4/` follow the same per-subtask structure, with SLURM wrappers present

## Quickstart

Start by choosing the subtask you want to run:

```bash
cd Subtask1
cat README.md