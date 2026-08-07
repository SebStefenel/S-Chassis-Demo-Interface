#!/bin/bash
# One-time kiosk setup for S-Chassis Demo on Ubuntu/GNOME Pi
# Run as: bash setup_kiosk.sh

set -e

APP_DIR="/home/ubuntu/S-Chassis-Demo-Interface"
PYTHON="$APP_DIR/venv/bin/python"
LOG="$HOME/sdv-demo.log"

echo "=== S-Chassis Kiosk Setup ==="

# ── 1. GDM auto-login (no login screen on boot) ───────────────────────────────
echo "[1/5] Configuring auto-login..."
sudo bash -c "cat > /etc/gdm3/custom.conf << 'EOF'
[daemon]
AutomaticLoginEnable=true
AutomaticLogin=ubuntu

[security]

[xdmcp]

[chooser]

[debug]
EOF"

# ── 2. GNOME autostart entry (launch app when desktop loads) ──────────────────
echo "[2/5] Creating autostart entry..."
mkdir -p "$HOME/.config/autostart"
cat > "$HOME/.config/autostart/sdv-demo.desktop" << EOF
[Desktop Entry]
Type=Application
Name=S-Chassis Demo
Exec=bash -c 'while true; do cd $APP_DIR && $PYTHON main.py >> $LOG 2>&1; sleep 2; done'
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
EOF

# ── 3. Disable screensaver, lock screen, and power saving ─────────────────────
echo "[3/5] Disabling screensaver and lock screen..."
gsettings set org.gnome.desktop.screensaver lock-enabled false
gsettings set org.gnome.desktop.session idle-delay 0
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing'
gsettings set org.gnome.settings-daemon.plugins.power power-button-action 'nothing'

# ── 4. Black desktop background (hides any brief flash before app starts) ─────
echo "[4/5] Setting black desktop background..."
gsettings set org.gnome.desktop.background picture-uri ''
gsettings set org.gnome.desktop.background picture-uri-dark ''
gsettings set org.gnome.desktop.background primary-color '#000000'
gsettings set org.gnome.desktop.background color-shading-type 'solid'

# ── 5. Auto-hide the dock so it never appears over the app ────────────────────
echo "[5/5] Auto-hiding GNOME dock..."
gsettings set org.gnome.shell.extensions.dash-to-dock autohide true
gsettings set org.gnome.shell.extensions.dash-to-dock dock-fixed false
gsettings set org.gnome.shell.extensions.dash-to-dock intellihide false 2>/dev/null || true

echo ""
echo "=== Setup complete ==="
echo "App log will be written to: $LOG"
echo "Reboot to apply: sudo reboot"
