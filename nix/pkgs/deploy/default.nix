{ lib', pkgs }:
pkgs.writeShellApplication {
  name = "mcp_graphql-deploy";
  runtimeInputs = pkgs.lib.flatten [
    pkgs.uv
    pkgs.sops
    lib'.envs.default
  ];
  text = ''
    pyproject_toml="pyproject.toml"

    # Determine previous revision if it exists (repository may have fewer than two commits)
    prev_rev=$(git rev-parse --quiet --verify HEAD~1)

    if [[ -n "''${prev_rev}" ]]; then
      if ! git diff "''${prev_rev}" HEAD -- "''${pyproject_toml}" | grep -q '^[-+]version'; then
        echo "''${pyproject_toml} version has not changed. Skipping deployment."
        exit 0
      fi
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
