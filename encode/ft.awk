#!/usr/bin/env awk -f

{
    printf "%s", $1
    for (i=2; i<=NF; i++) {
        if ($i < $TRUNCATE) {
            printf " %s:1", $i
        }
    }
    printf "\n"
}
        
