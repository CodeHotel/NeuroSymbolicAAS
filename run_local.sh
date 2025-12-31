#!/usr/bin/env bash
set -euo pipefail

AUTO_YES=0
DO_RUN=0
FORCE_OVERWRITE=0

usage() {
  cat <<EOF
Usage: ./run_local.sh [options]

Options:
  -y, --yes            Non-interactive: choose the DEFAULT for each prompt (recommended)
  --run                After deploy, port-forward services and keep running
  --force-overwrite    Delete/recreate stack resources if they already exist (destructive)
  -h, --help           Show help

Examples:
  ./run_local.sh
  ./run_local.sh -y
  ./run_local.sh -y --run
  ./run_local.sh -y --force-overwrite
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes) AUTO_YES=1; shift ;;
    --run) DO_RUN=1; shift ;;
    --force-overwrite) FORCE_OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] [INFO] $*"; }
warn() { echo "[$(ts)] [WARN] $*" >&2; }
err() { echo "[$(ts)] [ERROR] $*" >&2; }
die() { err "$*"; exit 1; }

# confirm "Question" "Y" or "N"
confirm() {
  local q="$1"
  local default="${2:-N}"   # default answer for interactive mode too
  local prompt="[y/N]"
  [[ "$default" == "Y" ]] && prompt="[Y/n]"

  if [[ "$AUTO_YES" -eq 1 ]]; then
    echo
    echo ">>> AUTO mode (-y): $q -> default=$default"
    [[ "$default" == "Y" ]] && return 0 || return 1
  fi

  while true; do
    echo
    read -r -p "$q $prompt: " ans </dev/tty || true
    ans="${ans:-$default}"
    case "$ans" in
      y|Y|yes|YES) return 0 ;;
      n|N|no|NO) return 1 ;;
      *) echo "Please answer y or n." ;;
    esac
  done
}

need_cmd() { command -v "$1" >/dev/null 2>&1; }

ensure_cmd() {
  local cmd="$1"
  if need_cmd "$cmd"; then
    log "Tool OK: $cmd"
    return 0
  fi
  die "Missing required tool: $cmd"
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

K8S_DIR="${K8S_DIR:-k8s}"
[[ -d "$K8S_DIR" ]] || die "Missing directory: $K8S_DIR"

NAMESPACE="${NAMESPACE:-default}"
API_IMAGE="${API_IMAGE:-factory-api-server:latest}"

AASX_LOCAL_PORT="${AASX_LOCAL_PORT:-5001}"
API_LOCAL_PORT="${API_LOCAL_PORT:-8000}"
PORT_FORWARD_ADDRESS="${PORT_FORWARD_ADDRESS:-127.0.0.1}"

cluster_reachable() { kubectl cluster-info >/dev/null 2>&1; }

detect_provider() {
  if need_cmd k3s && sudo systemctl is-active --quiet k3s 2>/dev/null; then echo "k3s"; return; fi
  if need_cmd kind && kind get clusters >/dev/null 2>&1; then echo "kind"; return; fi
  if need_cmd minikube && minikube status >/dev/null 2>&1; then echo "minikube"; return; fi
  if need_cmd microk8s; then echo "microk8s"; return; fi
  echo "unknown"
}

CLUSTER_PROVIDER="unknown"

ensure_cluster_or_die() {
  cluster_reachable || die "kubectl cannot reach a cluster."
  CLUSTER_PROVIDER="$(detect_provider)"
  log "Kubernetes reachable. Provider: $CLUSTER_PROVIDER"
}

ensure_namespace() {
  if kubectl get ns "$NAMESPACE" >/dev/null 2>&1; then
    log "Namespace OK: $NAMESPACE"
  else
    warn "Namespace does not exist: $NAMESPACE"
    confirm "Create namespace '$NAMESPACE'?" "Y" || die "Namespace required."
    kubectl create ns "$NAMESPACE"
  fi
}

set_default_envs_with_warnings() {
  if [[ -z "${USE_STANDARD_SERVER+x}" ]]; then USE_STANDARD_SERVER="true"; warn "USE_STANDARD_SERVER default=true"; fi
  if [[ -z "${AAS_SERVER_IP+x}" ]]; then AAS_SERVER_IP="aasx-server-service"; warn "AAS_SERVER_IP default=aasx-server-service"; fi
  if [[ -z "${AAS_SERVER_PORT+x}" ]]; then AAS_SERVER_PORT="5001"; warn "AAS_SERVER_PORT default=5001"; fi
  if [[ -z "${FORCE_LOCAL_MODE+x}" ]]; then FORCE_LOCAL_MODE="true"; warn "FORCE_LOCAL_MODE default=true"; fi

  log "API env to apply:"
  echo "  USE_STANDARD_SERVER=$USE_STANDARD_SERVER"
  echo "  AAS_SERVER_IP=$AAS_SERVER_IP"
  echo "  AAS_SERVER_PORT=$AAS_SERVER_PORT"
  echo "  FORCE_LOCAL_MODE=$FORCE_LOCAL_MODE"
}

check_collisions_and_prompt_overwrite() {
  local -a deps=(api-server-deployment aasx-server-deployment)
  local -a svcs=(api-server-service aasx-server-service)
  local -a pvcs=(factory-shared-pvc)
  local -a sas=(api-service-account)
  local -a roles=(job-creator)
  local -a rbs=(api-job-creator-binding)

  local -a found=()
  for x in "${deps[@]}"; do kubectl -n "$NAMESPACE" get deploy "$x" >/dev/null 2>&1 && found+=("Deployment/$x"); done
  for x in "${svcs[@]}"; do kubectl -n "$NAMESPACE" get svc  "$x" >/dev/null 2>&1 && found+=("Service/$x"); done
  for x in "${pvcs[@]}"; do kubectl -n "$NAMESPACE" get pvc  "$x" >/dev/null 2>&1 && found+=("PVC/$x"); done
  for x in "${sas[@]}";  do kubectl -n "$NAMESPACE" get sa   "$x" >/dev/null 2>&1 && found+=("ServiceAccount/$x"); done
  for x in "${roles[@]}";do kubectl -n "$NAMESPACE" get role "$x" >/dev/null 2>&1 && found+=("Role/$x"); done
  for x in "${rbs[@]}";  do kubectl -n "$NAMESPACE" get rolebinding "$x" >/dev/null 2>&1 && found+=("RoleBinding/$x"); done

  if (( ${#found[@]} == 0 )); then
    log "No collisions found."
    return 0
  fi

  warn "Existing resources detected in namespace '$NAMESPACE':"
  for item in "${found[@]}"; do echo "  - $item"; done

  if [[ "$FORCE_OVERWRITE" -eq 1 ]]; then
    warn "--force-overwrite set: will delete and recreate."
  else
    # IMPORTANT: default is NO (safe) so we don't constantly remake stable stuff.
    confirm "Overwrite (DELETE & recreate) the resources above?" "N" || {
      log "Keeping existing resources (recommended for stable AASX/PVC/RBAC)."
      return 0
    }
  fi

  log "Deleting stack resources (destructive)..."
  kubectl -n "$NAMESPACE" delete deploy "${deps[@]}" --ignore-not-found
  kubectl -n "$NAMESPACE" delete svc  "${svcs[@]}" --ignore-not-found
  kubectl -n "$NAMESPACE" delete pvc  "${pvcs[@]}" --ignore-not-found
  kubectl -n "$NAMESPACE" delete rolebinding "${rbs[@]}" --ignore-not-found
  kubectl -n "$NAMESPACE" delete role "${roles[@]}" --ignore-not-found
  kubectl -n "$NAMESPACE" delete sa "${sas[@]}" --ignore-not-found
  log "Delete complete."
}

resolve_api_build_inputs() {
  API_CONTEXT_DIR="${API_CONTEXT_DIR:-$ROOT_DIR}"

  if [[ -n "${API_DOCKERFILE:-}" ]]; then
    [[ -f "$ROOT_DIR/$API_DOCKERFILE" ]] || [[ -f "$API_CONTEXT_DIR/$API_DOCKERFILE" ]] || die "API_DOCKERFILE not found: $API_DOCKERFILE"
    [[ -f "$API_CONTEXT_DIR/$API_DOCKERFILE" ]] && API_DOCKERFILE_PATH="$API_CONTEXT_DIR/$API_DOCKERFILE" || API_DOCKERFILE_PATH="$ROOT_DIR/$API_DOCKERFILE"
    log "Using API Dockerfile: $API_DOCKERFILE_PATH"
    return 0
  fi

  [[ -f "$ROOT_DIR/api.Dockerfile" ]] || die "Expected api.Dockerfile in repo root."
  API_DOCKERFILE_PATH="$ROOT_DIR/api.Dockerfile"
  log "Auto-selected API Dockerfile: $API_DOCKERFILE_PATH"
}

build_api_image() {
  log "API image build step (dev-loop expected)."
  echo "  docker build -t \"$API_IMAGE\" -f \"$API_DOCKERFILE_PATH\" \"$API_CONTEXT_DIR\""
  # IMPORTANT: default YES (API changes frequently)
  confirm "Build/refresh API image now?" "Y" || { warn "Skipping API image build."; return 0; }
  docker build -t "$API_IMAGE" -f "$API_DOCKERFILE_PATH" "$API_CONTEXT_DIR"
  log "Build complete: $API_IMAGE"
}

load_image_into_cluster() {
  log "Load image into cluster runtime (so K8S doesn't pull from registry)."
  # IMPORTANT: default YES in dev loop
  confirm "Load API image into cluster runtime now?" "Y" || { warn "Skipping image load."; return 0; }

  case "$CLUSTER_PROVIDER" in
    k3s) docker save "$API_IMAGE" | sudo k3s ctr images import - ;;
    kind) kind load docker-image "$API_IMAGE" ;;
    minikube) minikube image load "$API_IMAGE" ;;
    microk8s) docker save "$API_IMAGE" | microk8s ctr images import - ;;
    *) die "Unknown provider; cannot safely load local image." ;;
  esac
  log "Image load complete."
}

apply_k8s() {
  log "Applying kustomize manifests."
  # Default YES (safe: apply is idempotent; won’t restart unchanged deployments)
  confirm "kubectl apply -k \"$K8S_DIR\" ?" "Y" || die "Cancelled."
  kubectl apply -k "$K8S_DIR" -n "$NAMESPACE"
}

set_api_env() {
  log "Setting API env (this restarts ONLY api-server)."
  # Default YES (API dev)
  confirm "Apply env vars to api-server-deployment?" "Y" || { warn "Skipping env update."; return 0; }

  kubectl -n "$NAMESPACE" set env deployment/api-server-deployment \
    USE_STANDARD_SERVER="$USE_STANDARD_SERVER" \
    AAS_SERVER_IP="$AAS_SERVER_IP" \
    AAS_SERVER_PORT="$AAS_SERVER_PORT" \
    FORCE_LOCAL_MODE="$FORCE_LOCAL_MODE"
}

wait_ready() {
  log "Waiting for rollouts..."
  kubectl -n "$NAMESPACE" rollout status deploy/aasx-server-deployment --timeout=240s
  kubectl -n "$NAMESPACE" rollout status deploy/api-server-deployment  --timeout=240s
  log "Rollouts ready."
}

PF_PIDS=()
cleanup_port_forwards() {
  warn "Stopping port-forwards..."
  for pid in "${PF_PIDS[@]:-}"; do kill "$pid" >/dev/null 2>&1 || true; done
}

run_port_forwards_forever() {
  trap cleanup_port_forwards EXIT INT TERM
  log "--run enabled: starting port-forwards and staying alive (Ctrl+C to stop)."

  kubectl -n "$NAMESPACE" port-forward --address "$PORT_FORWARD_ADDRESS" svc/aasx-server-service "${AASX_LOCAL_PORT}:5001" >/tmp/pf_aasx.log 2>&1 &
  PF_PIDS+=("$!")
  kubectl -n "$NAMESPACE" port-forward --address "$PORT_FORWARD_ADDRESS" svc/api-server-service  "${API_LOCAL_PORT}:8000"  >/tmp/pf_api.log  2>&1 &
  PF_PIDS+=("$!")

  sleep 1
  for pid in "${PF_PIDS[@]}"; do
    kill -0 "$pid" >/dev/null 2>&1 || die "Port-forward failed. Check /tmp/pf_aasx.log and /tmp/pf_api.log"
  done

  log "AASX: http://$PORT_FORWARD_ADDRESS:$AASX_LOCAL_PORT"
  log "API : http://$PORT_FORWARD_ADDRESS:$API_LOCAL_PORT"
  while true; do sleep 3600; done
}

main() {
  ensure_cmd kubectl
  ensure_cmd docker
  ensure_cmd sudo

  ensure_cluster_or_die
  ensure_namespace
  set_default_envs_with_warnings

  # IMPORTANT: Overwrite defaults to NO (protect stable resources)
  check_collisions_and_prompt_overwrite

  resolve_api_build_inputs
  build_api_image
  load_image_into_cluster

  apply_k8s
  set_api_env
  wait_ready

  log "DONE."
  if [[ "$DO_RUN" -eq 1 ]]; then
    run_port_forwards_forever
  else
    echo "Run with --run to keep port-forwards alive:"
    echo "  ./run_local.sh -y --run"
  fi
}

main
