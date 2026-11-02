#!/usr/bin/env bash
# Fail when client data or a local-only record is in the repository.
#
# Client project configs, site outlines, client numbers, the engineering log,
# review records, code snapshots, data products and credentials stay on the
# operator's machine (README.md, "This repository"; .gitignore). This checks
# every tracked file and everything staged for the next commit by path, and,
# when ~/.config/insar/client_markers.txt exists (one client name or
# coordinate per line, never committed), the lines a commit adds.
#
#   scripts/check_repo_clean.sh            tracked + staged paths (the GitHub checks)
#   scripts/check_repo_clean.sh --staged   also the staged lines (.githooks/pre-commit)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

FORBIDDEN='^(tests/test_local_|configs/config_[^/]*\.ya?ml$|context/|tests/golden_client_numbers\.json$|HANDOFF\.md$|report_assets/|review_[^/]*/|backups/|\.claude/|config\.yaml$|project_zone\.geojson$|configs/archive/.*\.ya?ml$)|:Zone\.Identifier$|\.(h5|tif|tiff|kmz|npz|gpkg|zip|SAFE)(/|$)|\.bak(_[0-9_]+)?$|(^|/)\.(netrc|cdsapirc)$'

paths="$( { git ls-files; git diff --cached --name-only --diff-filter=ACMR; } | sort -u )"
# the exceptions: the public mine layers, and the validation protocols and derived truth of the
# public validation sites (the example configs left the repository on 28 Oct 2026)
ALLOWED='^context/(SOURCE\.txt|cmis_[^/]*\.geojson|coal_mine[^/]*\.geojson)$|^review_2026-10-08/(protocol_[A-Za-z0-9]+\.yaml|truth/[^/]+\.(csv|geojson|json|md)|truth/scripts/[^/]+|truth/checks/[^/]+\.json)$'
bad="$(printf '%s\n' "$paths" | grep -E "$FORBIDDEN" | grep -vE "$ALLOWED" || true)"
if [[ -n "$bad" ]]; then
    echo "check_repo_clean: these must not be in the repository (client data or local records):"
    printf '  %s\n' $bad
    echo "Unstage them (git rm --cached <file>) and keep them on this machine; see README.md, 'This repository'."
    exit 1
fi

MARKERS="${INSAR_CLIENT_MARKERS:-$HOME/.config/insar/client_markers.txt}"
if [[ "${1:-}" == "--staged" && -f "$MARKERS" ]]; then
    added="$(git diff --cached -U0 --diff-filter=ACMR | grep -E '^\+[^+]' || true)"
    hits=""
    while IFS= read -r m; do
        m="${m%%#*}"; m="$(echo "$m" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        [[ -z "$m" ]] && continue
        if printf '%s\n' "$added" | grep -qiF -- "$m"; then hits+="  $m"$'\n'; fi
    done < "$MARKERS"
    if [[ -n "$hits" ]]; then
        echo "check_repo_clean: the staged changes name client markers from $MARKERS:"
        printf '%s' "$hits"
        echo "Use a neutral name (a project's folder name, 'the example site') instead."
        exit 1
    fi
fi
echo "check_repo_clean: no client data or local records in the repository."
