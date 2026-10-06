#!/bin/bash
# Secrets are loaded from .env — never commit .env to git
set -a
source /Users/oluwatosinayeni/embedding-server/.env
set +a

cd /Users/oluwatosinayeni/embedding-server
exec /opt/homebrew/opt/python@3.11/bin/python3.11 server.py >> /Users/oluwatosinayeni/embedding-server/server.log 2>&1
