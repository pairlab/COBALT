#!/usr/bin/env python3
"""
Frontend Setup Script for IsaacLab Teleoperation System

This script configures the frontend services by generating configuration files
based on the user's IP address and domain information.
"""

import sys
import argparse
import socket
import re
from pathlib import Path


def get_local_ip():
    """Get the local IP address of the machine."""
    try:
        # Connect to a remote address to determine the local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def validate_ip(ip_address):
    """Validate if the provided string is a valid IP address."""
    pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    if re.match(pattern, ip_address):
        parts = ip_address.split('.')
        return all(0 <= int(part) <= 255 for part in parts)
    return False


def generate_nginx_frontend_conf(server_name, overwrite):
    """Generate the nginx frontend configuration file."""

    template = f"""server {{
    listen 80;
    listen [::]:80;
    server_name {server_name};

    location / {{
        proxy_pass http://teleop-frontend:${{PORT}};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }}
}}
"""

    output_path = Path(__file__).parent / "nginx" / "frontend.conf.template"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not output_path.exists() or overwrite:
        with open(output_path, 'w') as f:
            f.write(template)
        print(f"✓ Generated nginx frontend configuration: {output_path}")
    else:
        print(f"✓ Skipped generating nginx frontend configuration (file exists and overwrite not allowed): {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Setup frontend configuration for IsaacLab Teleoperation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_frontend.py --domain myserver.example.com
        """
    )

    parser.add_argument(
        "--domain",
        type=str,
        help='Domain name for the frontend server (e.g., myserver.example.com)'
    )

    parser.add_argument(
        "--overwrite",
        action='store_true',
        help='Overwrite existing configuration files if they exist'
    )

    args = parser.parse_args()

    print(f"\n🚀 Setting up frontend configuration")

    try:
        if not args.domain:
            args.domain = get_local_ip()
            if not validate_ip(args.domain):
                print(f"❌ Invalid IP address: {args.domain}")
                sys.exit(1)

        # Generate configuration files
        generate_nginx_frontend_conf(args.domain, args.overwrite)

    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
