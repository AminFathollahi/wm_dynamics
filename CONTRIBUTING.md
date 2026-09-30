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
- Run long analyses as persistent user services, from a frozen copy of the code.
- Estimate runtime before launch.
- Record the exact command, environment, input identity, implementation identity, and output path.

## Documentation

- Update documentation, tests, provenance and ignore rules with the implementation.
- Keep generated outputs and runtime state out of version control.
- State the biological inference unit and the boundary of every scientific interpretation.
