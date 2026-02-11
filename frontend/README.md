# Frontend Dashboard

Next.js 16 app for NBA picks and model diagnostics.

## Requirements

- Node.js 20+
- npm 10+

## Local Development

From `/Users/lukesmac/nba-stats-project/frontend`:

```bash
npm ci
npm run dev
```

App URL:
- `http://localhost:3000`

Backend API URL is controlled by:
- `NEXT_PUBLIC_API_URL`

Default fallback:
- `https://nba-model-production.up.railway.app`

Optional request timeout override:
- `NEXT_PUBLIC_API_TIMEOUT_MS` (default `45000`)

## Quality Checks

```bash
npm run lint
npm run build
```

## Key Files

- `/Users/lukesmac/nba-stats-project/frontend/app/page.tsx` - picks dashboard
- `/Users/lukesmac/nba-stats-project/frontend/app/stats/page.tsx` - model/stats dashboard
- `/Users/lukesmac/nba-stats-project/frontend/lib/api.ts` - typed API client
- `/Users/lukesmac/nba-stats-project/frontend/lib/use-polling.ts` - polling hook
