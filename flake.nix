{
  description = "Python dev environment with dependencies managed by Nix";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      pythonEnv = pkgs.python313.withPackages (
        ps: with ps; [
          fenics
          ipython
          joblib
          jupyterlab
          nbdev
          numba
          pot
          pynvim
          rich
          seaborn
          # tqdm
        ]
      );
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [ pythonEnv ];
        shellHook = ''
          # Append src directory to PYTHONPATH for development
          export PYTHONPATH="$PYTHONPATH:$PWD/src"
          echo 'Python development environment for statFEM activated'
        '';
      };
    };
}
