#!/usr/bin/env bash
# ============================================================================
# AI Financial Agent — EC2 Deployment Script (HTTPS via Caddy)
#
# Usage:
#   1. Launch an EC2 instance (Amazon Linux 2023 or Ubuntu 22.04, t3.small+)
#   2. Open security group: inbound TCP 80, 443, and 22
#   3. SSH into the instance
#   4. Create a .env file with your API keys:
#        echo 'ANTHROPIC_API_KEY=sk-ant-...' > ~/.env
#        echo 'X_BEARER_TOKEN=...' >> ~/.env   # optional
#   5. Run this script:
#        cd ~/ai-financial-agent && bash deploy/setup.sh
#
# The script is idempotent — safe to run multiple times.
# ============================================================================

set -euo pipefail

APP_NAME="ai-financial-agent"
REPO_URL="${REPO_URL:-https://github.com/yimeng-blake/ai-financial-agent.git}"
BRANCH="${BRANCH:-main}"
ENV_FILE="${ENV_FILE:-$HOME/.env}"

echo "=========================================="
echo "  AI Financial Agent — EC2 Deployment"
echo "=========================================="

# ------------------------------------------------------------------
# 1. Install Docker & Docker Compose (if not already installed)
# ------------------------------------------------------------------
if ! command -v docker &>/dev/null; then
    echo "[1/5] Installing Docker..."

    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS_ID="$ID"
    else
        OS_ID="unknown"
    fi

    case "$OS_ID" in
        amzn)
            sudo dnf update -y
            sudo dnf install -y docker git
            sudo systemctl start docker
            sudo systemctl enable docker
            sudo usermod -aG docker "$USER"
            ;;
        ubuntu|debian)
            sudo apt-get update -y
            sudo apt-get install -y docker.io docker-compose git
            sudo systemctl start docker
            sudo systemctl enable docker
            sudo usermod -aG docker "$USER"
            ;;
        *)
            echo "Unsupported OS: $OS_ID. Please install Docker manually."
            exit 1
            ;;
    esac
else
    echo "[1/5] Docker already installed."
fi

# Install docker-compose plugin if not available
if ! sudo docker compose version &>/dev/null; then
    echo "Installing Docker Compose plugin..."
    sudo mkdir -p /usr/local/lib/docker/cli-plugins
    COMPOSE_URL="https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)"
    sudo curl -SL "$COMPOSE_URL" -o /usr/local/lib/docker/cli-plugins/docker-compose
    sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
fi

echo "Docker Compose: $(sudo docker compose version)"

# ------------------------------------------------------------------
# 2. Clone or update the repository
# ------------------------------------------------------------------
DEPLOY_DIR="$HOME/$APP_NAME"

if [ -d "$DEPLOY_DIR/.git" ]; then
    echo "[2/5] Updating repository..."
    cd "$DEPLOY_DIR"
    git fetch origin
    git reset --hard "origin/$BRANCH"
else
    echo "[2/5] Cloning repository..."
    git clone --branch "$BRANCH" "$REPO_URL" "$DEPLOY_DIR"
    cd "$DEPLOY_DIR"
fi

# ------------------------------------------------------------------
# 3. Check for .env file — copy into project dir for docker-compose
# ------------------------------------------------------------------
echo "[3/5] Checking environment file..."
if [ ! -f "$ENV_FILE" ]; then
    echo ""
    echo "ERROR: No .env file found at $ENV_FILE"
    echo ""
    echo "Create one with at minimum:"
    echo "  echo 'ANTHROPIC_API_KEY=your-key-here' > $ENV_FILE"
    echo ""
    echo "Then re-run this script."
    exit 1
fi

if ! grep -q "ANTHROPIC_API_KEY=." "$ENV_FILE"; then
    echo "WARNING: ANTHROPIC_API_KEY appears empty in $ENV_FILE"
fi

# docker-compose reads .env from the project directory
cp "$ENV_FILE" "$DEPLOY_DIR/.env"
echo "Environment file ready."

# ------------------------------------------------------------------
# 4. Build with docker-compose
# ------------------------------------------------------------------
echo "[4/5] Building Docker images..."
sudo docker compose build

# ------------------------------------------------------------------
# 5. Start (or restart) services
# ------------------------------------------------------------------
echo "[5/5] Starting services..."
sudo docker compose down 2>/dev/null || true
sudo docker compose up -d

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo '<your-ec2-public-ip>')

echo ""
echo "=========================================="
echo "  Deployment complete!"
echo "=========================================="
echo ""
echo "  App running at:"
echo "    https://basicmarket.net"
echo "    http://$PUBLIC_IP (direct IP, no HTTPS)"
echo ""
echo "  Caddy will auto-provision an SSL certificate"
echo "  from Let's Encrypt on first request (takes ~30s)."
echo ""
echo "  Useful commands:"
echo "    sudo docker compose logs -f          # View all logs"
echo "    sudo docker compose logs -f app      # App logs only"
echo "    sudo docker compose logs -f caddy    # Caddy/SSL logs"
echo "    sudo docker compose restart          # Restart all"
echo "    cd ~/$APP_NAME && bash deploy/setup.sh  # Redeploy"
echo ""
