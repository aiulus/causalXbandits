#!/bin/bash
echo "🔁 Updating submodules..."
git submodule update --init --recursive --remote
echo "✅ Done."
