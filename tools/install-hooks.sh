#!/usr/bin/env sh
# Install the pre-commit hook. Plain git hook, no framework required.
#
# A .pre-commit-config.yaml is also provided for people who use the pre-commit
# framework; this script exists so the protection works with nothing installed.
set -eu

root=$(git rev-parse --show-toplevel)
hook="$root/.git/hooks/pre-commit"

cat > "$hook" <<'HOOK'
#!/usr/bin/env sh
# Refuse commits containing credentials, personal identifiers or machine paths.
# Installed by tools/install-hooks.sh. Bypass with --no-verify (and then explain
# yourself in the commit message).
set -eu
python3 "$(git rev-parse --show-toplevel)/tools/scan_secrets.py" --staged
HOOK

chmod +x "$hook"
echo "installed $hook"
