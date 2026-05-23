{
  description = "PRML assignment2 dev environment (PyTorch CUDA via uv)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      perSystem =
        { system, ... }:
        let
          pkgs = import inputs.nixpkgs { inherit system; };
          python = pkgs.python312;
        in
        {
          devShells.default = pkgs.mkShell {
            name = "prml-assignment2-dev";

            packages = with pkgs; [
              python
              uv
              pyright
              ruff
              git
              stdenv.cc.cc
              zlib
              libGL
            ];

            env = {
              UV_PYTHON = python.interpreter;
              UV_PYTHON_DOWNLOADS = "never";
              LIBRARY_PATH = pkgs.lib.makeLibraryPath [
                pkgs.glibc
                pkgs.stdenv.cc.cc.lib
                pkgs.zlib
                pkgs.libGL
              ];
            };

            shellHook = ''
              export VIRTUAL_ENV="$PWD/.venv"
              export PATH="$VIRTUAL_ENV/bin:$PATH"
              export LD_LIBRARY_PATH="${
                pkgs.lib.makeLibraryPath [
                  pkgs.stdenv.cc.cc.lib
                  pkgs.zlib
                  pkgs.libGL
                ]
              }''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

              if [ ! -f "$VIRTUAL_ENV/bin/python" ]; then
                uv venv "$VIRTUAL_ENV" --python ${python.interpreter}
                uv sync
              fi
            '';
          };

          formatter = pkgs.nixfmt;
        };
    };
}
