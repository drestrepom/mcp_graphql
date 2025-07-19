{ lib', pkgs }: {
  lint = import ./lint { inherit lib' pkgs; };
}
