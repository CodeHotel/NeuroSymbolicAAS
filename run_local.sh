#!/usr/bin/env bash
# run_local.sh
#
# One-command local dev deploy for k3s/kind/minikube/microk8s
#
# WHAT THIS SCRIPT DOES (in order):
#  0) Parse args (-y/--yes, --run, --force-overwrite, --run-nsga2)
#  1) Verify core tools exist; if missing, offer to install (Debian/Ubuntu focused)
#     - sudo, curl, git, docker, kubectl
#  2) Ensure Docker is usable (auto-switch to "sudo docker" if needed)
#  3) Ensure Kubernetes cluster reachable; if not reachable, offer to bootstrap k3s:
#     - install k3s, start service, write kubeconfig to ~/.kube/config
#  4) Ensure namespace exists (default: "default")
#  5) Collision handling (default: NO delete; safe for stable resources)
#  6) Build + load API image (default: YES; dev loop)
#  7) Build + load NSGA2 image (DEFAULT: YES; assumed required)
#  8) kubectl apply -k k8s
#  9) Set API env vars (default: YES; restarts only api-server)
# 10) Wait for rollouts to be ready
# 11) Optionally run one-off NSGA2 job (--run-nsga2)
# 12) Optionally port-forward and stay alive (--run)
#
# Defaults are "dev-friendly":
# - Stable stack resources are NOT deleted unless you explicitly choose overwrite
# - API image build/load defaults YES
# - NSGA2 image build/load defaults YES (as requested)
#
set -euo pipefail

AUTO_YES=0
DO_RUN=0
FORCE_OVERWRITE=0
RUN_NSGA2_ONCE=0

usage() {
  cat <<EOF
Usage: ./run_local.sh [options]

Options:
  -y, --yes            Non-interactive: choose the DEFAULT for each prompt
  --run                After deploy, port-forward services and keep running
  --force-overwrite    Delete/recreate stack resources if they already exist (destructive)
  --run-nsga2           Run one NSGA2 Job once (from CronJob template) after deploy
  -h, --help           Show help

Examples:
  ./run_local.sh
  ./run_local.sh -y
  ./run_local.sh -y --run
  ./run_local.sh -y --run-nsga2
  ./run_local.sh -y --force-overwrite
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes) AUTO_YES=1; shift ;;
    --run) DO_RUN=1; shift ;;
    --force-overwrite) FORCE_OVERWRITE=1; shift ;;
    --run-nsga2) RUN_NSGA2_ONCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] [INFO] $*"; }
warn() { echo "[$(ts)] [WARN] $*" >&2; }
err() { echo "[$(ts)] [ERROR] $*" >&2; }
die() { err "$*"; exit 1; }

confirm() {
  local q="$1"
  local default="${2:-N}"
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

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

K8S_DIR="${K8S_DIR:-k8s}"
[[ -d "$K8S_DIR" ]] || die "Missing directory: $K8S_DIR"

NAMESPACE="${NAMESPACE:-default}"

API_IMAGE="${API_IMAGE:-factory-api-server:latest}"
AASX_LOCAL_PORT="${AASX_LOCAL_PORT:-5001}"
API_LOCAL_PORT="${API_LOCAL_PORT:-8000}"
PORT_FORWARD_ADDRESS="${PORT_FORWARD_ADDRESS:-127.0.0.1}"

# NSGA2 integration (assumed required)
NSGA2_IMAGE="${NSGA2_IMAGE:-factory-nsga2-simulator:latest}"
NSGA2_DOCKERFILE="${NSGA2_DOCKERFILE:-Dockerfile.nsga2}"
NSGA2_CONTEXT_DIR="${NSGA2_CONTEXT_DIR:-$ROOT_DIR}"
NSGA2_CRONJOB="${NSGA2_CRONJOB:-nsga2-simulator-template}"

# Docker command (may become "sudo docker" if needed)
DOCKER="docker"

# ------------------------------ OS/package helpers ------------------------------
is_debian_like() {
  [[ -f /etc/debian_version ]] || grep -qiE 'ubuntu|debian' /etc/os-release 2>/dev/null || return 1
}

apt_install() {
  local -a pkgs=("$@")
  is_debian_like || die "Auto-install only implemented for Debian/Ubuntu right now."
  sudo apt-get update -y
  sudo apt-get install -y "${pkgs[@]}"
}

ensure_tool_or_install() {
  local cmd="$1"

  if need_cmd "$cmd"; then
    log "Tool OK: $cmd"
    return 0
  fi

  warn "Tool MISSING: $cmd"

  need_cmd sudo || die "Missing required tool: sudo"
  if ! is_debian_like; then
    die "Missing tool '$cmd' and auto-install not supported on this OS."
  fi

  # Missing required tools default YES (otherwise script cannot proceed)
  confirm "Install missing dependency '$cmd' now?" "Y" || die "Cannot continue without: $cmd"

  case "$cmd" in
    curl) apt_install curl ;;
    git) apt_install git ;;
    docker)
      apt_install docker.io
      sudo systemctl enable --now docker || true
      ;;
    kubectl)
      # If k3s exists, link kubectl to k3s. Otherwise try snap.
      if need_cmd k3s; then
        warn "kubectl missing but k3s exists -> linking kubectl -> k3s"
        sudo ln -sf "$(command -v k3s)" /usr/local/bin/kubectl
      else
        if need_cmd snap; then
          sudo snap install kubectl --classic
        else
          die "kubectl install path not available (no k3s, no snap). Install kubectl manually."
        fi
      fi
      ;;
    k3s)
      ensure_tool_or_install curl
      curl -sfL https://get.k3s.io | sh -
      sudo systemctl enable --now k3s
      ;;
    *) die "No installer mapping for missing tool: $cmd" ;;
  esac

  need_cmd "$cmd" || die "Install attempted but '$cmd' is still missing."
  log "Installed: $cmd"
}

ensure_docker_works() {
  if $DOCKER version >/dev/null 2>&1; then
    log "Docker usable as current user."
    return 0
  fi

  warn "Docker exists but not usable without sudo (permissions)."
  confirm "Use 'sudo docker' for this run?" "Y" || die "Docker not usable."
  DOCKER="sudo docker"
  log "Using DOCKER='$DOCKER'"
}

# ------------------------------ Cluster/provider detection ------------------------------
cluster_reachable() { kubectl cluster-info >/dev/null 2>&1; }

detect_provider() {
  if need_cmd k3s && sudo systemctl is-active --quiet k3s 2>/dev/null; then echo "k3s"; return; fi
  if need_cmd kind && kind get clusters >/dev/null 2>&1; then echo "kind"; return; fi
  if need_cmd minikube && minikube status >/dev/null 2>&1; then echo "minikube"; return; fi
  if need_cmd microk8s; then echo "microk8s"; return; fi
  echo "unknown"
}

CLUSTER_PROVIDER="unknown"

ensure_kubeconfig_for_k3s() {
  mkdir -p ~/.kube
  if [[ -f /etc/rancher/k3s/k3s.yaml ]]; then
    sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
    sudo chown "$USER":"$USER" ~/.kube/config
    chmod 600 ~/.kube/config
    log "kubeconfig set: ~/.kube/config (from /etc/rancher/k3s/k3s.yaml)"
  else
    die "k3s installed but /etc/rancher/k3s/k3s.yaml not found."
  fi
}

ensure_cluster_or_bootstrap() {
  if cluster_reachable; then
    CLUSTER_PROVIDER="$(detect_provider)"
    log "Kubernetes reachable. Provider: $CLUSTER_PROVIDER"
    return 0
  fi

  warn "kubectl cannot reach a Kubernetes cluster."
  echo
  echo "This script can bootstrap a local k3s cluster automatically."
  echo "It will:"
  echo "  - install k3s (if missing)"
  echo "  - start k3s service"
  echo "  - write kubeconfig to ~/.kube/config"
  echo
  confirm "Bootstrap local k3s cluster now?" "Y" || die "No reachable cluster."

  ensure_tool_or_install k3s
  ensure_kubeconfig_for_k3s

  cluster_reachable || die "Cluster still not reachable after k3s bootstrap."
  CLUSTER_PROVIDER="k3s"
  log "k3s bootstrap complete; cluster is reachable."
}

# ------------------------------ Namespace + env defaults ------------------------------
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

# ------------------------------ Collision handling ------------------------------
check_collisions_and_prompt_overwrite() {
  local -a deps=(api-server-deployment aasx-server-deployment)
  local -a svcs=(api-server-service aasx-server-service)
  local -a pvcs=(factory-shared-pvc)
  local -a sas=(api-service-account)
  local -a roles=(job-creator)
  local -a rbs=(api-job-creator-binding)
  local -a cjs=("$NSGA2_CRONJOB")

  local -a found=()
  for x in "${deps[@]}"; do kubectl -n "$NAMESPACE" get deploy "$x" >/dev/null 2>&1 && found+=("Deployment/$x"); done
  for x in "${svcs[@]}"; do kubectl -n "$NAMESPACE" get svc  "$x" >/dev/null 2>&1 && found+=("Service/$x"); done
  for x in "${pvcs[@]}"; do kubectl -n "$NAMESPACE" get pvc  "$x" >/dev/null 2>&1 && found+=("PVC/$x"); done
  for x in "${sas[@]}";  do kubectl -n "$NAMESPACE" get sa   "$x" >/dev/null 2>&1 && found+=("ServiceAccount/$x"); done
  for x in "${roles[@]}";do kubectl -n "$NAMESPACE" get role "$x" >/dev/null 2>&1 && found+=("Role/$x"); done
  for x in "${rbs[@]}";  do kubectl -n "$NAMESPACE" get rolebinding "$x" >/dev/null 2>&1 && found+=("RoleBinding/$x"); done
  for x in "${cjs[@]}";  do kubectl -n "$NAMESPACE" get cronjob "$x" >/dev/null 2>&1 && found+=("CronJob/$x"); done

  if (( ${#found[@]} == 0 )); then
    log "No collisions found."
    return 0
  fi

  warn "Existing resources detected in namespace '$NAMESPACE':"
  for item in "${found[@]}"; do echo "  - $item"; done

  if [[ "$FORCE_OVERWRITE" -eq 1 ]]; then
    warn "--force-overwrite set: will delete and recreate."
  else
    # default NO: protect stable AASX/PVC/RBAC/CronJob-template
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
  kubectl -n "$NAMESPACE" delete cronjob "${cjs[@]}" --ignore-not-found
  log "Delete complete."
}

# ------------------------------ API build+load ------------------------------
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
  echo "  $DOCKER build -t \"$API_IMAGE\" -f \"$API_DOCKERFILE_PATH\" \"$API_CONTEXT_DIR\""
  # default YES: API changes frequently
  confirm "Build/refresh API image now?" "Y" || { warn "Skipping API image build."; return 0; }
  $DOCKER build -t "$API_IMAGE" -f "$API_DOCKERFILE_PATH" "$API_CONTEXT_DIR"
  log "Build complete: $API_IMAGE"
}

load_image_into_cluster() {
  local image="$1"
  log "Loading image into cluster runtime (so K8S won't pull from the internet)."
  log "  Provider: $CLUSTER_PROVIDER"
  log "  Image   : $image"

  case "$CLUSTER_PROVIDER" in
    k3s) $DOCKER save "$image" | sudo k3s ctr images import - ;;
    kind) kind load docker-image "$image" ;;
    minikube) minikube image load "$image" ;;
    microk8s) $DOCKER save "$image" | microk8s ctr images import - ;;
    *) die "Unknown provider; cannot safely load local image." ;;
  esac
}

load_api_into_cluster() {
  log "API image load step."
  confirm "Load API image into cluster runtime now?" "Y" || { warn "Skipping API image load."; return 0; }
  load_image_into_cluster "$API_IMAGE"
  log "API image load complete."
}

# ------------------------------ NSGA2 build+load (DEFAULT ON) ------------------------------
build_nsga2_image() {
  [[ -f "$ROOT_DIR/$NSGA2_DOCKERFILE" ]] || die "Missing NSGA2 Dockerfile: $NSGA2_DOCKERFILE"

  log "NSGA2 image build step (assumed required)."
  echo "  $DOCKER build -t \"$NSGA2_IMAGE\" -f \"$ROOT_DIR/$NSGA2_DOCKERFILE\" \"$NSGA2_CONTEXT_DIR\""
  # default YES: requested default behavior
  confirm "Build/refresh NSGA2 image now?" "Y" || die "NSGA2 is assumed required; refusing to proceed without it."
  $DOCKER build -t "$NSGA2_IMAGE" -f "$ROOT_DIR/$NSGA2_DOCKERFILE" "$NSGA2_CONTEXT_DIR"
  log "NSGA2 build complete: $NSGA2_IMAGE"
}

load_nsga2_into_cluster() {
  log "NSGA2 image load step (assumed required)."
  confirm "Load NSGA2 image into cluster runtime now?" "Y" || die "NSGA2 is assumed required; refusing to proceed without it."
  load_image_into_cluster "$NSGA2_IMAGE"
  log "NSGA2 image load complete."
}

# ------------------------------ Apply + rollout + env ------------------------------
apply_k8s() {
  log "Applying kustomize manifests."
  confirm "kubectl apply -k \"$K8S_DIR\" ?" "Y" || die "Cancelled."
  kubectl apply -k "$K8S_DIR" -n "$NAMESPACE"
}

set_api_env() {
  log "Setting API env (this restarts ONLY api-server)."
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

run_nsga2_once() {
  [[ "$RUN_NSGA2_ONCE" -eq 1 ]] || return 0

  kubectl -n "$NAMESPACE" get cronjob "$NSGA2_CRONJOB" >/dev/null 2>&1 || die "Missing CronJob template: $NSGA2_CRONJOB (did apply succeed?)"

  local job="nsga2-run-$(date +%s)"
  log "Creating one-off NSGA2 Job from CronJob template: $NSGA2_CRONJOB -> $job"
  confirm "Run NSGA2 once now?" "Y" || { warn "Skipping NSGA2 run."; return 0; }

  kubectl -n "$NAMESPACE" create job --from=cronjob/"$NSGA2_CRONJOB" "$job"
  log "Waiting for Job completion (timeout 30m): $job"
  kubectl -n "$NAMESPACE" wait --for=condition=complete job/"$job" --timeout=1800s

  log "NSGA2 job completed. Logs:"
  kubectl -n "$NAMESPACE" logs job/"$job" --all-containers=true || true
}

# ------------------------------ Port-forward run mode ------------------------------
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
  # Tool installation is part of the script (restored + explicit)
  ensure_tool_or_install sudo
  ensure_tool_or_install curl
  ensure_tool_or_install git
  ensure_tool_or_install docker
  ensure_tool_or_install kubectl

  ensure_docker_works

  # Cluster bootstrap if needed (k3s)
  ensure_cluster_or_bootstrap

  ensure_namespace
  set_default_envs_with_warnings

  # default overwrite = NO (protect stable resources)
  check_collisions_and_prompt_overwrite

  resolve_api_build_inputs

  # API: default YES (dev changes)
  build_api_image
  load_api_into_cluster

  # NSGA2: DEFAULT ON (assumed required)
  build_nsga2_image
  load_nsga2_into_cluster

  apply_k8s
  set_api_env
  wait_ready

  run_nsga2_once

  log "DONE."
  if [[ "$DO_RUN" -eq 1 ]]; then
    run_port_forwards_forever
  else
    echo "Run with --run to keep port-forwards alive:"
    echo "  ./run_local.sh -y --run"
  fi
}

main
