#!/usr/bin/env bash
# Install a git pre-push hook: whenever a push carries changes to submission/ (code or
# weights), validate + package it, and submit if FM_AUTO_SUBMIT=1.
#
#   scripts/install-hooks.sh
#   FM_AUTO_SUBMIT=1 git push          # package + submit on push
set -euo pipefail
cd "$(dirname "$0")/.."
HOOK=.git/hooks/pre-push
cat > "${HOOK}" <<'EOF'
#!/usr/bin/env bash
# installed by scripts/install-hooks.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
changed=0
while read -r local_ref local_sha remote_ref remote_sha; do
  [ "${local_sha}" = "0000000000000000000000000000000000000000" ] && continue
  if [ "${remote_sha}" = "0000000000000000000000000000000000000000" ]; then
    range="${local_sha}"
  else
    range="${remote_sha}..${local_sha}"
  fi
  if git diff --name-only ${range} -- submission | grep -q .; then changed=1; fi
done
[ "${changed}" = "1" ] || exit 0
echo "[pre-push] submission/ changed: validating and packaging"
scripts/package.sh
if [ "${FM_AUTO_SUBMIT:-0}" = "1" ]; then
  scripts/submit.sh "$(cat dist/LATEST)"
fi
EOF
chmod +x "${HOOK}"
echo "installed ${HOOK}"
