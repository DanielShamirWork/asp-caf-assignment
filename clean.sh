#!/bin/bash
# Clean up all pycache and build files

echo "🧹 Cleaning up build artifacts and cache files..."

# Remove Python cache files
echo "  Removing __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Remove .pyc files
echo "  Removing .pyc files..."
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Remove .pyo files
echo "  Removing .pyo files..."
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Remove build directories
echo "  Removing build directories..."
rm -rf libcaf/build
rm -rf libcaf/libcaf.egg-info
rm -rf caf/caf.egg-info

# Remove compiled extensions
echo "  Removing compiled extensions..."
find libcaf -type f -name "*.so" -delete 2>/dev/null || true

# Remove coverage files
echo "  Removing coverage files..."
rm -f libcaf/*.gcda 2>/dev/null || true
rm -rf tests/.coverage 2>/dev/null || true
rm -rf coverage 2>/dev/null || true

# Remove pytest cache
echo "  Removing pytest cache..."
rm -rf .pytest_cache 2>/dev/null || true

echo "✅ Cleanup complete!"
