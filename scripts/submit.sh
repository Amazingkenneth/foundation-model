#!/usr/bin/env bash
# Package (unless an artifact is given) and submit to AIcrowd.
#
#   AICROWD_API_KEY=... scripts/submit.sh                  # package HEAD's submission/ and submit
#   scripts/submit.sh dist/submission-abc123.tar.gz        # submit an existing artifact
#   DRY_RUN=1 scripts/submit.sh                            # show what would upload
#   WATCH=1 scripts/submit.sh                              # follow grading
#
# Auth: AICROWD_API_KEY in the environment, or a prior `whest login`.
# The challenge allows 10 submissions per team per UTC day.
set -euo pipefail
cd "$(dirname "$0")/.."

ART="${1:-}"
if [ -z "${ART}" ]; then
  scripts/package.sh
  ART=$(cat dist/LATEST)
fi
SHA=$(git rev-parse --short HEAD 2>/dev/null || echo local)
MSG=$(git log -1 --pretty=%s 2>/dev/null | cut -c1-80 || echo "")
DESC="${DESCRIPTION:-foundation-model ${SHA}: ${MSG}}"

ARGS=(--description "${DESC}" --yes --format plain)
[ "${DRY_RUN:-0}" = "1" ] && ARGS+=(--dry-run)
[ "${WATCH:-0}" = "1" ] && ARGS+=(--watch --watch-timeout "${WATCH_TIMEOUT:-3600}")

if [ "${DRY_RUN:-0}" != "1" ] && [ -z "${AICROWD_API_KEY:-}" ] && [ ! -f "${HOME}/.aicrowd/config.toml" ] && [ ! -d "${HOME}/.config/aicrowd" ]; then
  echo "warning: no AICROWD_API_KEY set; relying on a previous 'whest login'" >&2
fi
echo "== whest submit ${ART} (${DESC})"
whest submit "${ART}" "${ARGS[@]}"
