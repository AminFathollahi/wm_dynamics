# Contributing

## Code

- Prefer established libraries and shared project modules over local reimplementations.
- Use short, descriptive module, function, variable, and command names.
- Keep functions focused and interfaces explicit.
- Avoid comments and docstrings when names and structure make the code clear.
- Add comments only for constraints or behavior that cannot be expressed clearly in code.
- Keep scientific estimands, preprocessing scope, split boundaries, and random seeds explicit.
- Fit all learned preprocessing and representations on training data only.
- Preserve existing work and keep unrelated changes outside the patch.

## Validation

- Add focused tests for each changed scientific or software contract.
- Run targeted tests first and the broader suite after integration.
- Treat successful execution and scientific validity as separate checks.
- Record exclusions, failures, denominators, and invalid draws rather than dropping them silently.

## Long analyses

- Write atomic incremental results and resumable checkpoints.
- Run long jobs as persistent user services.
- Estimate runtime before launch and check jobs at the expected completion time instead of polling.
- Record the exact command, environment, input identity, implementation identity, and output path.
- Queue GPU analyses when the shared device is occupied.

## Delegation

- Give each worker a bounded objective and non-overlapping file ownership.
- Save useful progress in small patches before starting expensive validation.
- Request one ETA, then review work at completion or a hard blocker.

## Documentation

- Update user documentation, task status, tests, provenance, and ignore rules with the implementation.
- Keep generated outputs and runtime state out of version control.
- State the biological inference unit and the boundary of every scientific interpretation.
