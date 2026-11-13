# Deployment

## Backend (Railway)

1. Deploy this repository with Railway/Railpack, using the repository root as the service root.
2. Set the start command to:
   `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
3. Add `CRASHCAST_CORS_ORIGINS=https://crashtest.rjbassig.dev` (and the Vercel preview origin when testing).
4. Ensure generated research artifacts exist in the deployable image or are
   fetched from an approved artifact store. The API deliberately returns 503
   when they are absent rather than serving placeholder data.
5. Use `/health` as the health check.

## Dashboard (Vercel)

1. Set the Vercel root directory to `dashboard`.
2. Set `CRASHCAST_API_URL=https://api.crashtest.rjbassig.dev` only after that Railway custom domain has verified DNS and a successful `/health` response. This is server-side only: the dashboard proxies browser requests through `/api/research/*`, so the API origin is not exposed to the browser and production CORS remains narrowly scoped.
3. Build with `npm run build`.

## Required artifact behavior

The Railway image must contain `experiments/expanded/summary.json`,
`held_out_predictions.csv`, `calibration_curve.csv`, `error_by_beat.csv`, and
`data/provenance.json`. The API intentionally returns HTTP 503 when artifacts
are absent; this prevents a deployed shell with fabricated or empty charts.

## Verification order

1. Visit `https://api.crash-test.rjbassig.dev/health`; require `status: ok` and both artifact flags true.
2. Visit `/api/overview`; verify the 2025 MAE matches `experiments/expanded/summary.json`.
3. Deploy the dashboard and verify it displays those values using the configured API URL.
4. Add `crashtest.rjbassig.dev` and `api.crashtest.rjbassig.dev` DNS records only after each provider supplies its target.
5. Update the portfolio’s external links only after the two public URLs return real artifacts.
