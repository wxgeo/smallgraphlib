doc:
    {{uv}} run make -C doc html || true

_check-clean:
    git diff --exit-code || (echo -e "\e[1;97;43m[Warning]\e[0m Uncommitted changes in {{project}}! " && exit 1)
    git diff --cached --exit-code || (echo -e "\e[1;97;43m[Warning]\e[0m Staged changes in {{project}}! " && exit 1)

_check-branch branch="main":
    git rev-parse --abbrev-ref HEAD | grep -qx {{branch}} || (echo "\e[1;97;43m[Warning]\e[0m {{project}} is not on {{branch}} branch!" && exit 1)

release: (_check-clean) (_check-branch) (test) (doc)
    rm -rf dist/
    {{uv}} run semantic-release version
    uv publish