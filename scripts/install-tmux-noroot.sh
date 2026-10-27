#!/bin/bash
# Install tmux into $HOME without root (apt-get download + dpkg -x), for hosts where sudo needs a password.
# Leaves a wrapper at ~/bin/tmux that sets LD_LIBRARY_PATH itself.
set -uo pipefail
D=$HOME/tmux-local
mkdir -p "$D/debs" "$D/root" "$HOME/bin"
cd "$D/debs" || exit 1

pkgs=$(apt-cache depends --recurse --no-recommends --no-suggests --no-conflicts --no-breaks \
        --no-replaces --no-enhances tmux 2>/dev/null \
        | grep -E '^[a-z0-9][a-z0-9.+-]*$' | sort -u | tr '\n' ' ')
echo "packages: $pkgs"
apt-get download $pkgs > /tmp/tmux-dl.log 2>&1 || echo "(some packages failed; see /tmp/tmux-dl.log)"
echo "downloaded: $(ls -1 *.deb 2>/dev/null | wc -l) debs"
for f in ./*.deb; do [ -f "$f" ] && dpkg -x "$f" "$D/root"; done

cat > "$HOME/bin/tmux" <<'WRAP'
#!/bin/bash
export LD_LIBRARY_PATH="$HOME/tmux-local/root/usr/lib/x86_64-linux-gnu:$HOME/tmux-local/root/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$HOME/tmux-local/root/usr/bin/tmux" "$@"
WRAP
chmod +x "$HOME/bin/tmux"
"$HOME/bin/tmux" -V || { echo "FAILED: tmux does not run"; exit 1; }
