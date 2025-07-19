{lib', pkgs}:
pkgs.writeShellApplication {
  name = "lint";
  runtimeInputs = [
    lib'.envs.default
  ];
  text = ''
    if test "''${CI:-}"; then
     ruff format --config ruff.toml --diff
     ruff check --config ruff.toml
    else
     ruff format --config ruff.toml
     ruff check --config ruff.toml --fix
    fi
    mypy --explicit-package-bases --config-file mypy.ini -m mcp_graphql
  '';

  bashOptions = ["errexit" "pipefail" "nounset"	];
}
