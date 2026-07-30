# Phase 2 Task 1: Setup Elasticsearch Infrastructure

## 1. Context and Coding Rules
Before starting any code modifications, you **MUST** read and understand our project's coding rules. 
Please refer to the rules file located at: `Backend/Back-End/.agents/AGENTS.md`.
Also, refer to Phase 2 in `ARCHITECTURE_UPGRADE_PLAN.md`.

## 2. Objective
Establish the foundation for Elasticsearch (ES) text retrieval by configuring the local development environment and adding the necessary Python dependencies.

## 3. Requirements
- Create a `docker-compose.yml` file in the root of the backend (`Backend/Back-End/docker-compose.yml`).
  - It should spin up a single-node Elasticsearch 8.x instance.
  - Disable security (`xpack.security.enabled=false`) for local development ease.
  - Expose port `9200`.
- Update `requirements.txt` to include the official Python `elasticsearch` client (`elasticsearch>=8.0.0`).
- Create `src/config/elastic_settings.py` (or extend `settings.py`) to hold the ES connection URL (default to `http://localhost:9200`).

## 4. Expected Output & Reporting
- Generate a `phase2_task1_report.md` explaining the changes.
- Provide instructions on how to start the ES container (`docker-compose up -d`) and test the connection.
