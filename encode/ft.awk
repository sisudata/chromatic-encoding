#!/usr/bin/env awk -f
{
    printf "%s", $1
    for (i=2; i<=NF; i++) {
        if (0.0 + $i < 0.0 + TRUNCATE) {
            printf " %s:1", $i
        }
    }
    printf "\n"
}
        
