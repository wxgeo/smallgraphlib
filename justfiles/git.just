clone:
    git clone git@github.com:wxgeo/smallgraphlib.git

push:
    git push
    git push --tags

see:
    @output=$(git -c color.status=always status -s); \
     ahead=$(git rev-list --count @{u}..HEAD 2>/dev/null || echo 0); \
     ahead_str=$([ "$ahead" -gt 0 ] && echo -e " \e[1;34m(+$ahead)\e[0m" || echo ""); \
     if [ -z "$output" ]; then echo -e "\n\e[1;97;44m {{project}} \e[0m$ahead_str \e[32m✔\e[0m"; \
     else echo -e "\n\e[1;97;44m {{project}} \e[0m$ahead_str\n$output"; fi

commit-lock:
    git commit uv.lock -m "dev: update uv.lock"
