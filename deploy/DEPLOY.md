# Deploy (Telegram mini app)

Serves `miniapp/index.html` and the FastAPI backend on one box behind Caddy
(auto-HTTPS) with a systemd unit for the backend. The miniapp calls
`https://YOUR-DOMAIN/api/...`; Caddy strips `/api` and proxies to
`127.0.0.1:8000` where uvicorn runs `backend.game:app`.

Prereqs on the server: the repo checked out (with `.venv` built and deps
installed per README), a domain pointing at the box, and Caddy installed
(`https://caddyserver.com/docs/install`).

## 1. Create the bot

Talk to [@BotFather](https://t.me/BotFather):

- `/newbot` — create the bot, copy the API token.
- `/newapp` (or the Mini App / Apps menu) — set the Mini App URL to
  `https://YOUR-DOMAIN/` (this page is `miniapp/index.html`). Note:
  Telegram requires the URL to be HTTPS.

## 2. Configure the backend

In the repo's `.env` set:

```
TELEGRAM_BOT_TOKEN=123456789:AA...   # from BotFather
REQUIRE_TG_AUTH=true
```

`REQUIRE_TG_AUTH=true` turns on server-side verification of Telegram
initData (see `backend/auth.py`). Leave it `false` to run in dev/demo mode
with no auth.

## 3. Point a domain at the miniapp

The domain's web root serves `miniapp/`. In `deploy/Caddyfile` replace
`YOUR-DOMAIN` and `<REPO>` (the absolute path of the repo). The static root
must serve the repo's `miniapp/` directory so BotFather's Mini App URL loads
it.

## 4. Install

```
sudo bash deploy/setup.sh
```

The script checks Caddy is installed and that the placeholders were replaced,
copies `Caddyfile` → `/etc/caddy/Caddyfile` and the service unit →
`/etc/systemd/system/crowdcheck.service`, starts the backend
(`systemctl enable --now crowdcheck`) and reloads Caddy. It is idempotent:
rerunning it reinstalls the same files and restarts nothing that is already
running.

Caddy will mint a TLS certificate automatically on first request.

## Security notes

- initData is signed with the bot token and **must** be verified server-side
  (this repo now does that in `backend/auth.py` whenever
  `REQUIRE_TG_AUTH=true`). Never trust `telegram-web-app.js`-side data alone.
- Keep `TELEGRAM_BOT_TOKEN` secret — anyone holding it can forge valid
  initData. It lives in `.env` (not committed) and is read by systemd via
  `EnvironmentFile`.
- The backend binds only to `127.0.0.1:8000`; all public traffic flows
  through Caddy on 80/443.
- Check the expiry: initData older than `max_age_seconds` (86400) is
  rejected.