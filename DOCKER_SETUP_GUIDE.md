# YMERA Platform - Docker Setup Guide

## Overview

This guide provides comprehensive instructions for setting up and running the YMERA Platform using Docker and Docker Compose, including special configurations for Desktop Commander and local filesystem access.

## Prerequisites

### Required Software
- **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux) - version 20.10 or later
- **Docker Compose** - version 2.0 or later
- **Git** - for cloning the repository

### System Requirements
- **RAM**: Minimum 8GB (16GB recommended)
- **Disk Space**: Minimum 10GB free space
- **CPU**: Multi-core processor recommended

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/ymera-mansour/workspace.git
cd workspace

# Copy environment template
cp 00-FOUNDATION/.env.template .env

# Edit .env file with your API keys
# At minimum, configure one AI provider (Gemini, Groq, Mistral, etc.)
nano .env  # or use your preferred editor
```

### 2. Configure Environment Variables

Edit the `.env` file and set at least these required variables:

```bash
# Required: At least one AI provider
GEMINI_API_KEY=your_actual_api_key_here
# or
GROQ_API_KEY=your_actual_api_key_here

# Database credentials (defaults are fine for development)
POSTGRES_PASSWORD=ymera_secure_password
REDIS_URL=redis://redis:6379/0
```

### 3. Build and Run with Docker Compose

```bash
# Build all services
docker-compose build

# Start all services in background
docker-compose up -d

# View logs
docker-compose logs -f

# Check service status
docker-compose ps
```

### 4. Access the Platform

Once all services are running:

- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

## Filesystem Access Configuration

### For Local Development

The default configuration mounts a `./workspace` directory:

```yaml
volumes:
  - ./workspace:/mnt/local_workspace
```

### For Desktop Commander (Windows)

If you're using Desktop Commander and need to access files on your Windows Desktop:

#### Option 1: Docker Desktop File Sharing

1. Open **Docker Desktop**
2. Go to **Settings** → **Resources** → **File Sharing**
3. Add the directory you want to access (e.g., `C:\Users\YourName\Desktop`)
4. Click **Apply & Restart**

#### Option 2: Update docker-compose.yml

Edit `docker-compose.yml` and update the volume path:

```yaml
services:
  ymera:
    volumes:
      # Windows Desktop access
      - /c/Users/YourName/Desktop/workspace:/mnt/local_workspace
      # or use environment variable
      - ${LOCAL_WORKSPACE_PATH:-./workspace}:/mnt/local_workspace
```

#### Option 3: Use Environment Variable

Add to your `.env` file:

```bash
# Windows path format (Docker for Windows converts this)
LOCAL_WORKSPACE_PATH=/c/Users/Mohamed Mansour/Desktop/ymera-fullstack

# Or on Linux/Mac
LOCAL_WORKSPACE_PATH=/home/username/workspace
```

### For Linux/Mac

```yaml
volumes:
  - ~/workspace:/mnt/local_workspace
  # or specific path
  - /path/to/your/workspace:/mnt/local_workspace
```

## Docker Commands Reference

### Building

```bash
# Build specific service
docker-compose build ymera

# Build without cache
docker-compose build --no-cache

# Build with specific Dockerfile
docker build -t ymera-platform:latest .
```

### Running

```bash
# Start all services
docker-compose up

# Start in background (detached)
docker-compose up -d

# Start specific service
docker-compose up ymera

# Run with custom command
docker-compose run ymera python main.py --help
```

### Monitoring

```bash
# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f ymera

# View last 100 lines
docker-compose logs --tail=100 ymera

# Check resource usage
docker stats
```

### Maintenance

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (⚠️ DELETES DATA)
docker-compose down -v

# Restart specific service
docker-compose restart ymera

# Execute command in running container
docker-compose exec ymera bash
docker-compose exec ymera python main.py --help

# View running processes
docker-compose ps
```

### Cleanup

```bash
# Remove stopped containers
docker-compose rm

# Clean up unused images
docker image prune -a

# Clean up everything (⚠️ USE WITH CAUTION)
docker system prune -a --volumes
```

## Running Without Docker Compose

### Single Container Mode

```bash
# Build image
docker build -t ymera-platform:latest .

# Run with volume mount
docker run -it --rm \
  --name ymera-platform \
  -p 8000:8000 \
  -v "$(pwd)/workspace:/mnt/local_workspace" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/.env:/app/.env" \
  --env-file .env \
  ymera-platform:latest \
  python main.py --repo-path /mnt/local_workspace --phases all

# Windows PowerShell format
docker run -it --rm `
  --name ymera-platform `
  -p 8000:8000 `
  -v "${PWD}/workspace:/mnt/local_workspace" `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/.env:/app/.env" `
  --env-file .env `
  ymera-platform:latest `
  python main.py --repo-path /mnt/local_workspace --phases all
```

### API Mode

```bash
# Run API server
docker run -d \
  --name ymera-api \
  -p 8000:8000 \
  --env-file .env \
  ymera-platform:latest \
  uvicorn api:app --host 0.0.0.0 --port 8000
```

## Troubleshooting

### Issue: "failed to fetch metadata: no such file or directory"

**Solution**: This usually means Docker BuildKit is having issues. Try:

```bash
# Disable BuildKit
export DOCKER_BUILDKIT=0
docker build -t ymera-platform:latest .

# Or update Docker to latest version
```

### Issue: "unable to evaluate symlinks in Dockerfile path"

**Solution**: Ensure you're in the correct directory:

```bash
cd /path/to/workspace
ls -la Dockerfile  # Should exist
docker build -t ymera-platform:latest .
```

### Issue: "Cannot connect to the Docker daemon"

**Solution**: Ensure Docker Desktop is running:

- Windows/Mac: Start Docker Desktop application
- Linux: `sudo systemctl start docker`

### Issue: Permission denied for volume mounts

**Solution**: 

On Linux:
```bash
# Change ownership
sudo chown -R 1000:1000 ./data ./logs

# Or run with current user
docker-compose run --user $(id -u):$(id -g) ymera
```

On Windows: Ensure the directory is added to Docker Desktop File Sharing settings.

### Issue: Port already in use

**Solution**:

```bash
# Check what's using the port
netstat -ano | findstr :8000  # Windows
lsof -i :8000                  # Linux/Mac

# Change port in docker-compose.yml or use different port
docker-compose up -d --scale ymera=0
docker run -p 8001:8000 ...
```

### Issue: Out of disk space

**Solution**:

```bash
# Check Docker disk usage
docker system df

# Clean up
docker system prune -a
docker volume prune
```

## Desktop Commander Specific Setup

### Windows Configuration

1. **Enable WSL2** (if not already enabled):
   ```powershell
   wsl --install
   ```

2. **Configure Docker Desktop**:
   - Open Docker Desktop
   - Settings → General → Use WSL2 based engine (enable)
   - Settings → Resources → File Sharing → Add your Desktop directory

3. **Update docker-compose.yml**:
   ```yaml
   services:
     ymera:
       volumes:
         - /c/Users/Mohamed Mansour/Desktop/ymera-fullstack:/mnt/local_workspace
   ```

4. **Run**:
   ```bash
   docker-compose up -d
   ```

### Verify Filesystem Access

```bash
# Access container
docker-compose exec ymera bash

# Check mounted directory
ls -la /mnt/local_workspace

# Test Python access
python -c "import os; print(os.listdir('/mnt/local_workspace'))"
```

## Best Practices

### Security
- ✅ Never commit `.env` file to version control
- ✅ Use strong passwords for PostgreSQL and Redis
- ✅ Limit volume mounts to only necessary directories
- ✅ Use read-only mounts where possible (`:ro` flag)

### Performance
- ✅ Use Docker volumes for database data (faster than bind mounts)
- ✅ Use `.dockerignore` to exclude unnecessary files
- ✅ Multi-stage builds to reduce image size
- ✅ Use specific image tags (not `latest`) in production

### Development
- ✅ Use `docker-compose.override.yml` for local customizations
- ✅ Mount source code as volume for hot reloading
- ✅ Use health checks for dependencies
- ✅ Set appropriate resource limits

## Advanced Configuration

### Custom Network Configuration

```yaml
networks:
  ymera-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.28.0.0/16
```

### Resource Limits

```yaml
services:
  ymera:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### Development Override

Create `docker-compose.override.yml`:

```yaml
version: '3.8'

services:
  ymera:
    volumes:
      - ./:/app  # Mount entire source for development
    environment:
      - DEBUG=true
      - LOG_LEVEL=DEBUG
    command: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
- [YMERA Platform README](README.md)

## Support

If you encounter issues:

1. Check Docker logs: `docker-compose logs -f`
2. Verify configuration: `docker-compose config`
3. Check service health: `docker-compose ps`
4. Review this guide's troubleshooting section
5. Create an issue on GitHub with logs and configuration

## Summary

This Docker setup provides:

- ✅ Complete containerized environment
- ✅ Easy filesystem access configuration
- ✅ Monitoring with Prometheus and Grafana
- ✅ Database persistence with volumes
- ✅ Network isolation for security
- ✅ Health checks and auto-restart
- ✅ Development and production configurations
