#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v caddy >/dev/null 2>&1; then
  echo "ERROR: caddy is not installed. Install it first (https://caddyserver.com/docs/install)." >&2
  exit 1
fi

if grep -q "YOUR-DOMAIN" Caddyfile; then
  echo "ERROR: you must replace YOUR-DOMAIN in deploy/Caddyfile with your real domain before running this script." >&2
  exit 1
fi

if grep -q "<REPO>" Caddyfile crowdcheck.service; then
  echo "ERROR: you must replace <REPO> in deploy/Caddyfile and deploy/crowdcheck.service with the absolute path to the repo before running this script." >&2
  exit 1
fi

sudo install -m 0644 Caddyfile /etc/caddy/Caddyfile
sudo install -m 0644 crowdcheck.service /etc/systemd/system/crowdcheck.service

sudo systemctl daemon-reload
sudo systemctl enable --now crowdcheck
sudo systemctl reload caddy

echo
echo "Deployed. Next steps with @BotFather in Telegram:"
echo "  1. /newbot to create the bot, note the token."
echo "  2. For the API token, use /setprivacy, /setinlinefeedback, etc. as needed."
echo "  3. /newapp (or the Mini App menu) to register https://YOUR-DOMAIN/ as the Mini App URL."
echo "  4. Put TELEGRAM_BOT_TOKEN=<token> and REQUIRE_TG_AUTH=true in $PWD/../.env, then: sudo systemctl restart crowdcheck"