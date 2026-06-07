#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  NewsReX Environment Setup
#  Interactive installer — arrow keys to navigate, Enter to select.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

# ── Colors & symbols ─────────────────────────────────────────
BOLD="\033[1m"
DIM="\033[2m"
CYAN="\033[36m"
GREEN="\033[32m"
YELLOW="\033[33m"
MAGENTA="\033[35m"
RESET="\033[0m"
CHECK="${GREEN}✓${RESET}"
ARROW="${CYAN}❯${RESET}"
HIDE_CURSOR="\033[?25l"
SHOW_CURSOR="\033[?25h"

# Restore cursor on exit
trap 'printf "${SHOW_CURSOR}"' EXIT

# ── Helpers ──────────────────────────────────────────────────

banner() {
    echo ""
    echo -e "${BOLD}${CYAN}"
    cat << 'EOF'
    _   __                ____       _  __
   / | / /__ _      _____|  _ \ ___ | |/ /
  /  |/ / _ \ \ /\ / / __| |_) / _ \|   /
 / /|  /  __/\ V  V /\__ \  _ <  __/|   \
/_/ |_/\___| \_/\_/ |___/_| \_\___||_|\_\
EOF
    echo -e "${RESET}"
    echo -e "  ${DIM}Modular News Recommendation Research Framework${RESET}"
    echo ""
}

separator() {
    echo -e "${DIM}  ────────────────────────────────────────────${RESET}"
}

# Interactive arrow-key menu.
#   $1  = prompt
#   $2  = default index (0-based)
#   $3+ = "label|description" pairs
#
# Sets global REPLY to the chosen label.
pick() {
    local prompt="$1"; shift
    local selected="$1"; shift
    local -a items=("$@")
    local count=${#items[@]}

    # Parse labels and descriptions
    local -a labels=()
    local -a descs=()
    for item in "${items[@]}"; do
        labels+=("${item%%|*}")
        descs+=("${item#*|}")
    done

    printf "\n"
    printf "  ${BOLD}${prompt}${RESET}  ${DIM}(↑/↓ to move, Enter to select)${RESET}\n"
    printf "\n"

    # Hide cursor during selection
    printf "${HIDE_CURSOR}"

    # Draw menu
    _draw_menu() {
        local i
        for ((i = 0; i < count; i++)); do
            if ((i == selected)); then
                printf "    ${ARROW} ${BOLD}${CYAN}%-10s${RESET} ${DIM}%s${RESET}\n" "${labels[$i]}" "${descs[$i]}"
            else
                printf "      ${DIM}%-10s %s${RESET}\n" "${labels[$i]}" "${descs[$i]}"
            fi
        done
    }

    # Move cursor up N lines
    _move_up() {
        printf "\033[%dA" "$1"
    }

    _draw_menu

    while true; do
        # Read a single keypress
        IFS= read -rsn1 key || true

        # Arrow keys send escape sequences: ESC [ A/B
        if [[ "$key" == $'\x1b' ]]; then
            read -rsn2 -t 0.1 rest || true
            case "$rest" in
                "[A")  # Up
                    ((selected > 0)) && ((selected--)) || true
                    ;;
                "[B")  # Down
                    ((selected < count - 1)) && ((selected++)) || true
                    ;;
            esac
        elif [[ "$key" == "" ]]; then
            # Enter pressed
            break
        elif [[ "$key" == "k" ]]; then
            ((selected > 0)) && ((selected--)) || true
        elif [[ "$key" == "j" ]]; then
            ((selected < count - 1)) && ((selected++)) || true
        fi

        # Redraw
        _move_up "$count"
        _draw_menu
    done

    printf "${SHOW_CURSOR}"

    # Replace menu with the confirmed selection
    _move_up "$count"
    local i
    for ((i = 0; i < count; i++)); do
        printf "\033[2K"  # clear line
        if ((i == 0)); then
            printf "    ${CHECK} ${BOLD}${GREEN}%s${RESET}  ${DIM}%s${RESET}\n" "${labels[$selected]}" "${descs[$selected]}"
        else
            printf "\n"
        fi
    done
    # Move cursor back up past the blank lines
    if ((count > 1)); then
        _move_up $((count - 1))
        # Clear the blank lines by moving down and clearing
        for ((i = 1; i < count; i++)); do
            printf "\033[1B\033[2K"
        done
        # Position cursor after the selection line
        _move_up $((count - 1))
        printf "\033[1B"
    fi

    REPLY="${labels[$selected]}"
}

# Interactive yes/no with arrow keys
yesno() {
    local prompt="$1"
    local default="${2:-1}"  # 0=yes, 1=no
    local selected=$default

    printf "\n"
    printf "${HIDE_CURSOR}"

    _draw_yesno() {
        printf "\r\033[2K"
        if ((selected == 0)); then
            printf "  ${BOLD}%s${RESET}  ${CYAN}❯ ${BOLD}Yes${RESET}    ${DIM}No${RESET}" "$prompt"
        else
            printf "  ${BOLD}%s${RESET}    ${DIM}Yes${RESET}  ${CYAN}❯ ${BOLD}No${RESET}" "$prompt"
        fi
    }

    _draw_yesno

    while true; do
        IFS= read -rsn1 key || true
        if [[ "$key" == $'\x1b' ]]; then
            read -rsn2 -t 0.1 rest || true
            case "$rest" in
                "[C"|"[D")  # Left/Right arrow
                    selected=$(( (selected + 1) % 2 ))
                    ;;
            esac
        elif [[ "$key" == "h" || "$key" == "l" ]]; then
            selected=$(( (selected + 1) % 2 ))
        elif [[ "$key" == "y" || "$key" == "Y" ]]; then
            selected=0; break
        elif [[ "$key" == "n" || "$key" == "N" ]]; then
            selected=1; break
        elif [[ "$key" == "" ]]; then
            break
        fi
        _draw_yesno
    done

    printf "${SHOW_CURSOR}"

    # Replace with confirmed answer
    printf "\r\033[2K"
    if ((selected == 0)); then
        printf "  ${BOLD}%s${RESET}  ${CHECK} ${GREEN}Yes${RESET}\n" "$prompt"
        return 0
    else
        printf "  ${BOLD}%s${RESET}  ${CHECK} No\n" "$prompt"
        return 1
    fi
}

# ── Main ─────────────────────────────────────────────────────

banner

EXTRAS=()

# 1. Framework
pick "Framework" 0 \
    "keras|Keras 3 (JAX/Torch backend)" \
    "jax|JAX + Flax NNX" \
    "pytorch|PyTorch" \
    "all|All frameworks"
FRAMEWORK="$REPLY"

separator

# 2. CUDA
pick "CUDA version" 0 \
    "none|CPU only / macOS" \
    "cu118|CUDA 11.8" \
    "cu124|CUDA 12.4" \
    "cu126|CUDA 12.6" \
    "cu130|CUDA 13.0"
CUDA="$REPLY"

separator

# Build extras from selections
if [[ "$CUDA" == "none" ]]; then
    if [[ "$FRAMEWORK" == "all" ]]; then
        EXTRAS+=("all")
    else
        EXTRAS+=("$FRAMEWORK")
    fi
else
    EXTRAS+=("$CUDA")
    case "$FRAMEWORK" in
        all)     EXTRAS+=("all-${CUDA}");;
        keras)   EXTRAS+=("keras");;
        jax)     EXTRAS+=("jax" "jax-cuda");;
        pytorch) ;;  # cu* extra already includes torch
    esac
fi

# 3. Optional tools
if yesno "Modal (remote GPU training)?" 1; then
    EXTRAS+=("modal")
fi

if yesno "Dev tools (pytest, ruff, mypy)?" 1; then
    EXTRAS+=("dev")
fi

# ── Run ──────────────────────────────────────────────────────

CMD="uv sync"
for e in "${EXTRAS[@]}"; do
    CMD+=" --extra $e"
done

echo ""
separator
echo ""
echo -e "  ${ARROW} ${BOLD}Running:${RESET} ${MAGENTA}${CMD}${RESET}"
echo ""

eval "$CMD"

echo ""
echo -e "  ${CHECK} ${BOLD}${GREEN}Setup complete!${RESET}"
echo ""
