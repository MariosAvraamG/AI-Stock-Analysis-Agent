"""
Pytest configuration and fixtures for the test suite
"""
import sys
import os

# Add the app directory to Python path so tests can import from app
app_dir = os.path.join(os.path.dirname(__file__), '..', 'app')
sys.path.insert(0, os.path.abspath(app_dir))