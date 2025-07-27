{ lib', pkgs }: {
  lint = import ./lint { inherit lib' pkgs; };
  deploy = import ./deploy { inherit lib' pkgs; };
}
