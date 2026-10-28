#!/bin/sh
# CLI for the mcp gateway service: list tools, show one tool's
# input schema, call a tool. Talks to the service's HTTP API; the
# port is read from the sibling config.yaml.
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
BASE="http://127.0.0.1:$(grep -E '^port:' "$HERE/config.yaml" | head -1 | awk '{print $2}')"

usage() {
    cat <<'EOF'
mcp.sh -- call MCP tools from the shell

Commands:
  mcp.sh list
      Print every connected MCP server and its tools as JSON.

  mcp.sh schema <server> <tool>
      Print the input JSON schema of one tool, plus its
      description. Check this before calling a tool for the
      first time.

  mcp.sh call <server> <tool> ['<json-arguments>']
      Call a tool; the third argument is the tool's arguments
      as a JSON object (default '{}'). Response is JSON:
      {"ok": bool, "result": "..."} -- on ok=false, "result"
      carries the error text.

Examples:
  mcp.sh list
  mcp.sh schema files list_directory
  mcp.sh call calculation calculate '{"expression": "2+2"}'
  mcp.sh call search web_search '{"query": "uv python", "max_results": 3}'

Servers and tools are configured in config.yaml (same folder),
which the gateway hot-reloads on change.
EOF
    [ "${1:-}" = "err" ] && exit 2
    exit 0
}

[ $# -ge 1 ] || usage err
cmd=$1

case "$cmd" in
    -h|--help|help) usage ;;
    list)
        curl -fsS "$BASE/tools"
        ;;
    schema)
        [ $# -eq 3 ] || usage err
        curl -fsS "$BASE/schema/$2/$3"
        ;;
    call)
        [ $# -ge 3 ] || usage err
        args=${4:-'{}'}
        printf '{"server":"%s","tool":"%s","arguments":%s}' "$2" "$3" "$args" |
            curl -fsS -X POST "$BASE/call" -H 'Content-Type: application/json' -d @-
        ;;
    *)
        usage err
        ;;
esac
