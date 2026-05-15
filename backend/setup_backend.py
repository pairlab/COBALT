"""
Backend Setup Script for Teleoperation System

This script configures the backend services by generating configuration files
based on the user's IP address and other optional parameters.
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


def generate_nginx_backend_conf(domain, overwrite):
    """Generate the nginx backend configuration file."""

    template = f"""server {{
    listen 8080;
    listen [::]:8080;
    server_name {domain};  # Server domain

    location ~ /ws$ {{
        proxy_pass http://websocket-server:${{WEBSOCKET_SERVER_PORT}};
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }}

    location /media/ {{
        # When using host network mode for the media server,
        # use host.docker.internal to reference the host's network
        proxy_pass http://host.docker.internal:${{MEDIA_SERVER_PORT}}/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }}
}}
"""

    output_path = Path(__file__).parent / "nginx" / "backend.conf.template"

    if not output_path.exists() or overwrite:
        with open(output_path, 'w') as f:
            f.write(template)
        print(f"✓ Generated nginx backend configuration: {output_path}")
    else:
        print(f"✓ Skipped generating nginx backend configuration (file exists and overwrite not allowed): {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Setup backend configuration for IsaacLab Teleoperation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_backend.py --domain 192.168.1.100
  python setup_backend.py --auto-detect
        """
    )

    parser.add_argument(
        '--domain',
        type=str,
        help='Domain name of the server (e.g. example.com)'
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing configuration files"
    )

    args = parser.parse_args()

    if not args.domain:
        # Determine IP address
        domain = get_local_ip()
        print(f"Auto-detected domain: {domain}")
    else:
        domain = args.domain
        if not validate_ip(domain):
            print(f"Error: Invalid IP address format: {domain}")
            sys.exit(1)

    print(f"\n🚀 Setting up backend configuration for IP: {domain}")
    print()

    try:
        # Generate configuration files
        generate_nginx_backend_conf(domain, args.overwrite)

    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
