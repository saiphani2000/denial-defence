# Contributing

## Development setup

```bash
pip install -r requirements.txt
python scripts/bootstrap_eval_set.py
cp .env.example .env   # add OPENAI_API_KEY
python scripts/verify_setup.py
pytest
```

## Running locally

```bash
python web/app.py
```

## Evaluation

```bash
python eval/compare_eval.py
```

Requires `WANDB_API_KEY`. Set `SKIP_WANDB_TEST=1` to skip connectivity check.

## Code guidelines

- Tune agent prompts in `agents/prompts.py`, not in agent logic
- Add tests for playbook matching, routing, and API routes
- Do not commit `.env` or API keys
