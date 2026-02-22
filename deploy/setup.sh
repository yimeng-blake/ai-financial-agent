#!/usr/bin/env bash
# ============================================================================
# AI Financial Agent — EC2 Deployment Script
#
# Usage:
#   1. Launch an EC2 instance (Amazon Linux 2023 or Ubuntu 22.04, t3.small+)
#   2. Open security group: inbound TCP 80 (HTTP) and 22 (SSH)
#   3. SSH into the instance
#   4. Create a .env file with your API keys:
#        echo 'ANTHROPIC_API_KEY=sk-ant-...' > ~/.env
#        echo 'X_BEARER_TOKEN=...' >> ~/.env   # optional
#   5. Run this script:
#        curl -sSL https://raw.githubusercontent.com/yimeng-blake/ai-financial-agent/main/deploy/setup.sh | bash
#      OR copy it to the instance and run:
#        chmod +x setup.sh && ./setup.sh
#
# The script is idempotent — safe to run multiple times.
# ============================================================================

set -euo pipefail

APP_NAME="ai-financial-agent"
REPO_URL="${REPO_URL:-https://github.com/yimeng-blake/ai-financial-agent.git}"
BRANCH="${BRANCH:-main}"
ENV_FILE="${ENV_FILE:-$HOME/.env}"
HOST_PORT="${HOST_PORT:-80}"
CONTAINER_PORT="8000"

echo "=========================================="
echo "  AI Financial Agent — EC2 Deployment"
echo "=========================================="

# ------------------------------------------------------------------
# 1. Install Docker (if not already installed)
# ------------------------------------------------------------------
if ! command -v docker &>/dev/null; then
    echo "[1/5] Installing Docker..."

    # Detect OS
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS_ID="$ID"
    else
        OS_ID="unknown"
    fi

    case "$OS_ID" in
        amzn)
            # Amazon Linux 2023
            sudo dnf update -y
            sudo dnf install -y docker git
            sudo systemctl start docker
            sudo systemctl enable docker
            sudo usermod -aG docker "$USER"
            ;;
        ubuntu|debian)
            sudo apt-get update -y
            sudo apt-get install -y docker.io git
            sudo systemctl start docker
            sudo systemctl enable docker
            sudo usermod -aG docker "$USER"
            ;;
        *)
            echo "Unsupported OS: $OS_ID. Please install Docker manually."
            exit 1
            ;;
    esac

    echo "Docker installed. You may need to log out and back in for group changes."
else
    echo "[1/5] Docker already installed."
fi

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
# 3. Check for .env file
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

# Verify ANTHROPIC_API_KEY is set
if ! grep -q "ANTHROPIC_API_KEY=." "$ENV_FILE"; then
    echo "WARNING: ANTHROPIC_API_KEY appears empty in $ENV_FILE"
fi

echo "Environment file found at $ENV_FILE"

# ------------------------------------------------------------------
# 4. Build Docker image
# ------------------------------------------------------------------
echo "[4/5] Building Docker image..."
sudo docker build -t "$APP_NAME" "$DEPLOY_DIR"

# ------------------------------------------------------------------
# 5. Run (or restart) the container
# ------------------------------------------------------------------
echo "[5/5] Starting container..."

# Stop existing container if running
if sudo docker ps -q -f "name=$APP_NAME" | grep -q .; then
    echo "Stopping existing container..."
    sudo docker stop "$APP_NAME"
    sudo docker rm "$APP_NAME"
elif sudo docker ps -aq -f "name=$APP_NAME" | grep -q .; then
    sudo docker rm "$APP_NAME"
fi

sudo docker run -d \
    --name "$APP_NAME" \
    --restart unless-stopped \
    --env-file "$ENV_FILE" \
    -p "$HOST_PORT:$CONTAINER_PORT" \
    "$APP_NAME"

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo ""
echo "=========================================="
echo "  Deployment complete!"
echo "=========================================="
echo ""
echo "  App running at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo '<your-ec2-public-ip>'):$HOST_PORT"
echo ""
echo "  Useful commands:"
echo "    docker logs -f $APP_NAME          # View logs"
echo "    docker restart $APP_NAME          # Restart"
echo "    docker stop $APP_NAME             # Stop"
echo "    cd ~/$APP_NAME && bash deploy/setup.sh  # Redeploy"
echo ""
