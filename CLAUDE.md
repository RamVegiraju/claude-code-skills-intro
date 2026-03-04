# skill-sample

This project explores the benefits of using Claude Code skills vs not using skills for ML deployment workflows.

## Project Structure
- `claude-first-skill/` — deployment using Claude Code skills
- `no-skill-deployment/` — deployment without skills (manual/raw approach)

## Code Guidelines
- Keep code concise and focused
- Use AWS best practices for all deployments
- Prefer the **SageMaker High-Level Python SDK** where applicable
- Fall back to **Boto3** for lower-level AWS operations not covered by the high-level SDK
- Python only
