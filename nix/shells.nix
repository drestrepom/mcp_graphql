{
  lib',
  pkgs,
  self',
}:
{
  default = pkgs.mkShell {
    packages = [
      pkgs.uv
    ];

    env = {
      UV_NO_SYNC = "1";
      UV_PYTHON_DOWNLOADS = "never";
    };
    shellHook = ''
      unset PYTHONPATH
      export PYTHON_INTERPRETER="${lib'.envs.default}/bin/python"
      ${pkgs.envsubst}/bin/envsubst '$PYTHON_INTERPRETER' < .vscode/settings.json.template > .vscode/settings.json
    '';
  };
}
