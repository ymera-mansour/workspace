#!/bin/bash

# ============================================
# YMERA AI Platform - MCP Tools Installation
# ============================================
# Installs all 18 MCP servers required for the platform

set -e

echo "================================================"
echo "YMERA AI Platform - MCP Tools Installation"
echo "================================================"
echo ""
echo "This will install 18 MCP servers globally"
echo "Estimated time: 30-35 minutes"
echo ""

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed"
    echo "Please install Node.js v18+ from https://nodejs.org/"
    exit 1
fi

NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "Error: Node.js v18+ is required (found v$NODE_VERSION)"
    exit 1
fi

echo "✓ Node.js $(node --version) detected"
echo ""

# Phase 1: Critical Infrastructure (7 MCPs)
echo "Phase 1: Installing Critical Infrastructure MCPs (15 min)..."
echo ""

echo "1/7 Installing Python Execution MCP..."
npm install -g @modelcontextprotocol/server-python

echo "2/7 Installing Node.js Execution MCP..."
npm install -g @modelcontextprotocol/server-node

echo "3/7 Installing Filesystem MCP..."
npm install -g @modelcontextprotocol/server-filesystem

echo "4/7 Installing Git/GitHub MCP..."
npm install -g @modelcontextprotocol/server-github

echo "5/7 Installing PostgreSQL MCP..."
npm install -g @modelcontextprotocol/server-postgres

echo "6/7 Installing SQLite MCP..."
npm install -g @modelcontextprotocol/server-sqlite

echo "7/7 Installing Redis MCP..."
npm install -g mcp-redis

echo "✓ Phase 1 Complete"
echo ""

# Phase 2: Development Tools (6 MCPs)
echo "Phase 2: Installing Development Tools MCPs (10 min)..."
echo ""

echo "1/6 Installing Docker MCP..."
npm install -g mcp-docker

echo "2/6 Installing Kubernetes MCP..."
npm install -g mcp-kubernetes

echo "3/6 Installing Jest MCP..."
npm install -g mcp-jest

echo "4/6 Installing Pytest MCP..."
npm install -g mcp-pytest

echo "5/6 Installing Fetch/HTTP MCP..."
npm install -g @modelcontextprotocol/server-fetch

echo "6/6 Installing Brave Search MCP..."
npm install -g @modelcontextprotocol/server-brave-search

echo "✓ Phase 2 Complete"
echo ""

# Phase 3: Specialized Tools (5 MCPs)
echo "Phase 3: Installing Specialized Tools MCPs (10 min)..."
echo ""

echo "1/5 Installing Prometheus MCP..."
npm install -g mcp-prometheus

echo "2/5 Installing Elasticsearch MCP..."
npm install -g mcp-elasticsearch

echo "3/5 Installing Email MCP..."
npm install -g mcp-email

echo "4/5 Installing Slack MCP..."
npm install -g mcp-slack

echo "5/5 Installing Cloud Storage (S3) MCP..."
npm install -g mcp-s3

echo "✓ Phase 3 Complete"
echo ""

# Verify installations
echo "Verifying installations..."
echo ""

INSTALLED_COUNT=0
TOTAL_COUNT=18

check_package() {
    if npm list -g "$1" &> /dev/null; then
        echo "✓ $1"
        ((INSTALLED_COUNT++))
    else
        echo "✗ $1 (not found)"
    fi
}

check_package "@modelcontextprotocol/server-python"
check_package "@modelcontextprotocol/server-node"
check_package "@modelcontextprotocol/server-filesystem"
check_package "@modelcontextprotocol/server-github"
check_package "@modelcontextprotocol/server-postgres"
check_package "@modelcontextprotocol/server-sqlite"
check_package "mcp-redis"
check_package "mcp-docker"
check_package "mcp-kubernetes"
check_package "mcp-jest"
check_package "mcp-pytest"
check_package "@modelcontextprotocol/server-fetch"
check_package "@modelcontextprotocol/server-brave-search"
check_package "mcp-prometheus"
check_package "mcp-elasticsearch"
check_package "mcp-email"
check_package "mcp-slack"
check_package "mcp-s3"

echo ""
echo "================================================"
echo "MCP Tools Installation Complete!"
echo "================================================"
echo ""
echo "Installed: $INSTALLED_COUNT/$TOTAL_COUNT MCP servers"
echo ""

if [ $INSTALLED_COUNT -eq $TOTAL_COUNT ]; then
    echo "✓ All MCP tools installed successfully"
else
    echo "⚠ Some MCP tools failed to install"
    echo "  Check the output above for errors"
fi

echo ""
echo "Next steps:"
echo "1. Configure API keys in .env file"
echo "2. Review mcp_config.json for MCP server settings"
echo "3. Test MCP connections: python test_mcp.py"
echo ""
