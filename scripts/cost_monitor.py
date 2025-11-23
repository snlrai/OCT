#!/usr/bin/env python3
"""
AWS Cost Monitoring Script
Run this script to check your AWS costs and usage
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlops.monitor import CostMonitor


def main():
    print("OCT MLOps - AWS Cost Monitor")
    print("=" * 60)
    print("This script shows your current AWS costs and resource usage")
    print("=" * 60 + "\n")
    
    monitor = CostMonitor()
    monitor.print_cost_report()


if __name__ == '__main__':
    main()

