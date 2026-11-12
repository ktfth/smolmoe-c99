#!/bin/sh
# Prints, for each maintenance program with a manual page on this machine,
# "name<TAB>options<TAB>subcommands": what the page cites. Run it on a Linux
# with manual pages (the index in the binary came from Ubuntu 24.04), then
# turn the output into src/cmdcheck_index.inc with scripts/cmdcheck/gen-index.ts:
#   sh scripts/cmdcheck/index-man.sh > index.tsv
#   bun scripts/cmdcheck/gen-index.ts index.tsv "Ubuntu 24.04" > src/cmdcheck_index.inc
# Only option and subcommand names are kept, never the pages' text.
set -u

PROGRAMS="ls cp mv rm mkdir rmdir ln chmod chown chgrp touch cat tac head tail less more grep egrep fgrep
find xargs sed awk gawk mawk sort uniq cut paste tr wc tee diff cmp comm stat file du df free uname
hostname hostnamectl uptime who w whoami id groups date timedatectl cal ps top htop kill pkill pgrep
killall nice renice nohup timeout watch sleep tar gzip gunzip zcat bzip2 xz unxz zstd unzip zip rsync
scp sftp ssh ssh-keygen ssh-copy-id curl wget ping ping6 traceroute tracepath ip ss netstat dig
nslookup host resolvectl nmcli mount umount findmnt lsblk blkid fdisk sfdisk parted mkfs mkfs.ext4
mkswap swapon swapoff fsck e2fsck tune2fs resize2fs systemctl journalctl loginctl service apt apt-get
apt-cache apt-mark dpkg dpkg-query useradd userdel usermod groupadd groupdel passwd chpasswd sudo su
crontab chattr lsattr lsof dmesg sysctl modprobe lsmod lspci lsusb shutdown reboot poweroff env
printenv which whereis basename dirname realpath readlink openssl git ufw iptables nft dd shred
truncate wipefs hwclock timeout strace ldd lscpu nproc vmstat iostat sar journalctl logger
update-alternatives snap docker podman base64 md5sum sha256sum sha1sum split csplit fold fmt expand
nl od hexdump xxd iconv locale localectl chroot sync tree"

norm() {
    sed -e 's/\\f[BIRPV]//g' -e 's/\\f([A-Za-z][A-Za-z]//g' -e 's/\\f\[[^]]*\]//g' -e 's/\\-/-/g' \
        -e 's/\\(hy/-/g' -e 's/\\(en/-/g' -e 's/\\(em/-/g' -e 's/\\&//g' -e 's/\\[ ~]/ /g' -e 's/"/ /g'
}

page_text() {
    case $1 in
        *.gz) gzip -dc "$1" ;;
        *.bz2) bzip2 -dc "$1" ;;
        *.xz) xz -dc "$1" ;;
        *.zst) zstd -dc "$1" ;;
        *) cat "$1" ;;
    esac 2>/dev/null | norm
}

seen=" "
for name in $PROGRAMS; do
    case $seen in *" $name "*) continue ;; esac
    seen="$seen$name "
    page=$(man -w "$name" 2>/dev/null | head -n 1)
    [ -n "$page" ] && [ -r "$page" ] || continue
    text=$(page_text "$page")
    [ -n "$text" ] || continue
    options=$(printf '%s\n' "$text" | grep -o -E -e '(^|[^A-Za-z0-9-])--?[A-Za-z0-9][A-Za-z0-9-]*' |
        sed -e 's/^[^-]*//' | sort -u | tr '\n' ' ')
    # Subcommands: a word opening a line (after a roff macro), and git's git-<command> pages.
    subs=$(printf '%s\n' "$text" | sed -n -e 's/^\.[A-Z][A-Z]* //' -e 's/^\([a-z][a-z0-9-]*\).*/\1/p' |
        sort -u | tr '\n' ' ')
    if [ "$name" = git ]; then
        subs="$subs $(printf '%s\n' "$text" | grep -o -E 'git-[a-z][a-z0-9-]*' | sed 's/^git-//' | sort -u |
            tr '\n' ' ')"
    fi
    printf '%s\t%s\t%s\n' "$name" "$options" "$subs"
done
