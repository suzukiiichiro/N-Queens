#!/bin/sh

#for f in *_README_append.md; do echo "===== $f ====="; cat "$f"; echo; done \
#  > "merge_README_$(date +%Y%m%d_%H%M%S).md"

# out="merge_README_$(date +%Y%m%d_%H%M%S).md"
# {
#   echo "# merged README appends — $(date -Is)"
#   echo
#   for f in *_README_append.md; do
#     echo "===== $f ====="
#     cat "$f"
#     echo
#   done
# } > "$out"
# wc -l "$out"; ls -la "$out"
# 

# grep -H '^## ' *_README_append.md | sed 's/_README_append.md:/  |  /' > readme_index.txt
ls *_README_append.md | grep -vE '^(merge|394_merge)_' \
  | xargs grep -H '^## ' \
  | sed 's/_README_append.md:/  |  /' > readme_index.txt
cat readme_index.txt

