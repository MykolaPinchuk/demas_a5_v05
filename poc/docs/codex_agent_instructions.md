# Codex Agent Instructions

## Scope

- Work strictly within this repository/workspace.

## Execution Discipline

- Iterate rapidly in small increments and validate each change with real data.
- Finish and validate the current iteration before writing code for the next one.
- Run code to verify behavior; do not rely on assumptions or placeholders.
- When in doubt, err on the side of not overengineering.

## Performance Guardrails

- Estimate how long commands should take and stop them if they exceed expectations (≲ 3 minutes unless justified).
- Use timeouts for any command that might run longer than 30 seconds.
- Never run an agent without a timeout; default to 90 seconds.

## Fidelity & Troubleshooting

- Never replace user-provided values with placeholders.
- When code fails repeatedly, capture detailed logs and investigate.
- Maintain rapid iteration: run code frequently, move forward only after the previous step is validated.
