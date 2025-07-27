{ lib', pkgs }:
pkgs.writeShellApplication {
  name = "mcp_graphql-deploy";
  runtimeInputs = pkgs.lib.flatten [
    pkgs.uv
    lib'.envs.default
  ];
  text = ''
    pyproject_toml="pyproject.toml"

    if ! git diff HEAD~1 HEAD -- "''${pyproject_toml}" | grep -q '^[-+]version'; then
      : && info "''${pyproject_toml} version has not changed. Skipping deployment." \
        && return 0
    fi

    set -o allexport
    eval "$(sops -d --output-type dotenv secrets.yaml)"
    set +o allexport


    echo "Publishing new version for mcp_graphql"
    rm -rf "dist"
    uv build
    uv publish --token "''${PYPI_API_TOKEN}"
  '';
}
