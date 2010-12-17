#!/bin/bash

BASEDIR="$(dirname "$0")/../.."

PROG="$BASEDIR/build/genre"

[ -x "$PROG" ] || exit 1

NNFILE="$1"
shift

GENRES="pop
rock
classical
electronic
rap"

HITS=0
TOTAL=0

for F in "$@"
do
    CLS=$("$PROG" classify -f "$NNFILE" "$F" | sed -nr 's/^--- \[.*\]\((.*)\)/\1/p' | sed 's/,/\n/g')
    CLS=$(echo "$CLS" | cat -n | sort -nrk 2 | awk '{print $1}' | head -1)
    CLS=$(echo "$GENRES" | sed -n "$CLS p")
    TAG=$(cat "${F%%.wav}.tag" | sed 's/[0-9]* //' | grep -xiF "$GENRES" | head -1 | tr '[[:upper:]]' '[[:lower:]]')
    echo "$F: '$CLS', should be '$TAG'"
    [ "$CLS" == "$TAG" ] && ((HITS++))
    ((TOTAL++))
done

PERCENT=$((100 * $HITS / $TOTAL))

echo "$HITS / $TOTAL = $PERCENT%"

