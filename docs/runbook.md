# Runbook

Use an approved Amarel compute allocation. Keep project-specific usernames,
job IDs, and credentials outside versioned docs.

```bash
ssh -i "$HOME/.ssh/<key_name>" <netid>@amarel.hpc.rutgers.edu
cd "<project_root>"
srun --jobid=<allocation_job_id> --overlap --ntasks=1 --cpus-per-task=8 \
  env PROJECT_ROOT="<project_root>" PYTHON_BIN="<python>" \
  bash jobs/run_stage_on_allocation.sh smoke smoke
```

Scale only after smoke passes:

```bash
srun --jobid=<allocation_job_id> --overlap --ntasks=1 --cpus-per-task=32 \
  env PROJECT_ROOT="<project_root>" PYTHON_BIN="<python>" \
  bash jobs/run_post_full_on_allocation.sh
```

If WRDS authentication fails or MFA is not accepted, stop WRDS work and wait.
