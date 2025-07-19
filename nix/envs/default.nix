{
  inputs,
  pkgs,
  projectPath,
  inputs',
}:

let
  python = pkgs.python312;
  workspace = inputs.uv2nix.lib.workspace.loadWorkspace {
    workspaceRoot = ../..;
  };
  overlay = workspace.mkPyprojectOverlay {
    sourcePreference = "wheel";
  };
  pythonSet =
    # Use base package set from pyproject.nix builders
    (pkgs.callPackage inputs.pyproject-nix.build.packages {
      inherit python;
    }).overrideScope
      (
        pkgs.lib.composeManyExtensions [
          inputs.pyproject-build-systems.overlays.default
          overlay
        ]
      );
in
{
  default =
    (pythonSet.mkVirtualEnv "shared-expenses-api-env" workspace.deps.all).overrideAttrs
      (old: {
        venvIgnoreCollisions = [ "*" ];
      });

}
