# CHANGELOG

<!-- version list -->

## v0.2.0 (2026-05-03)

### Bug Fixes

- Benchmark now uses MLE restart, burn-in chain default increased to 500 and lambda_xi defaulted to
  on in training, fractional prior width not operating bug fix
  ([`7eae84b`](https://github.com/sirocco-rt/speculate/commit/7eae84b7bb632e3aa2f0c87425e1421903a806b0))

- Correcting log flux transformations to use different calculations from the linear regime.
  ([`c57a3f8`](https://github.com/sirocco-rt/speculate/commit/c57a3f810a29fba90a84407fcb2a304aed9cfb6d))

- Jitter consistency across V11 construction paths
  ([`b1f938c`](https://github.com/sirocco-rt/speculate/commit/b1f938c1aac4d4bfca82e2b33c76b9593cf5b03d))

- MCMC and inference optimisation fixes
  ([`4fb4bb0`](https://github.com/sirocco-rt/speculate/commit/4fb4bb008aafcbbbc9611698f92ba676dfe4994f))

- NUISANCE TRANSFORMS WERE APPLIED TWICE VIA `bulk_fluxes` BROADCAST, higher ceiling on GP kernel
  variance to prevent runaway fits and small bug fixes
  ([`6bdb9d2`](https://github.com/sirocco-rt/speculate/commit/6bdb9d2eaf26630ca9d60b1d63186e36c1636898))

- Patch to make speculate to check for free memory, not memory capacity.
  ([`bb9d879`](https://github.com/sirocco-rt/speculate/commit/bb9d8790ae341d564d7ee8f4ac8ded5c013dd3bb))

- Possible fix to token bypass not working
  ([`2f58e80`](https://github.com/sirocco-rt/speculate/commit/2f58e80bbb1135a3291d52797eca29581a6bb7ba))

- Reducing memory requirements for standard operation
  ([`0d1f73a`](https://github.com/sirocco-rt/speculate/commit/0d1f73a50d7fbf1ec52f3d3ed5f07c116be61ab0))

- Unifying the extinction laws across flux scales, set to F99 at R_V=3.1
  ([`a1591db`](https://github.com/sirocco-rt/speculate/commit/a1591db9e709332fd1b7e773e6be374e18dd5005))

- **benchmark**: Av NEVER ADDED TO MODEL WHEN freeze_av=False and POST-MCMC LABEL MISALIGNMENT
  RESOLVED
  ([`962b46b`](https://github.com/sirocco-rt/speculate/commit/962b46bb2692cf7a44d9ae489c321a7a5003194e))

- **benchmark**: Corrected tier 1 score before likely refactor
  ([`ad48140`](https://github.com/sirocco-rt/speculate/commit/ad481405f8e83e6c017fc141ae73148125b2e09c))

- **benchmark**: Tier3 fixes to the flux double scaling and the log propogated errors use log flux,
  not original fluxes
  ([`35a5017`](https://github.com/sirocco-rt/speculate/commit/35a5017b9293b0575f300dc774faf15dc20466e5))

- **Inference**: Fix to continuum_normlised missing option in dropdown menu
  ([`2182756`](https://github.com/sirocco-rt/speculate/commit/218275630854f4f066efba9587ed218f3d0a9244))

- **quick**: Fixing import errors and RMSE envelope data to np.float64
  ([`aada018`](https://github.com/sirocco-rt/speculate/commit/aada018631917b7a689c6815c3864916d04d5e3c))

- **starfish**: COVARIANCE PROPAGATION FORMULA INVERTED IN SPECTRUM MODEL [Resolved]
  ([`b2d9110`](https://github.com/sirocco-rt/speculate/commit/b2d91105eccbaf88a92bb6861eb2fab5685acb6c))

- **starfish**: REMOVED INHERITED GRID INFERFACES THAT AREN'T RELAVENT FOR SPECULATE
  ([`d2e7f11`](https://github.com/sirocco-rt/speculate/commit/d2e7f111d072533d8fc3f67f8b3465e5393edca8))

- **starfish**: SPECTRUM.MASKS RETURNS WAVELENGTHS INSTEAD OF BOOLEAN MASKS [RESOLVED]
  ([`2f18a05`](https://github.com/sirocco-rt/speculate/commit/2f18a056beab20017b9e98e4733d0459449c1258))

### Features

- Ability to freeze or thaw nuicense parameters in benchmark MLE / MCMC plus some benchmark altair
  fixes
  ([`6394ed1`](https://github.com/sirocco-rt/speculate/commit/6394ed16fa8dbc6d0909e44bcf3524e4f886a1b9))

- Add cornerplot data export functionality to inference and benchmark viewer
  ([`4b7b0e2`](https://github.com/sirocco-rt/speculate/commit/4b7b0e2918d17afd9debae8c2d83ddafe265eca4))

- Adding in pre-trained emulator models downloads to speculate for immediate inference use
  ([`2275fe6`](https://github.com/sirocco-rt/speculate/commit/2275fe6fd8d5405bb9686d82968affe6517f1b4e))

- Adding system resources usage bars to the sidebar of each notebook and MCMC plots to benchmark
  notebooks.
  ([`bf7a0d2`](https://github.com/sirocco-rt/speculate/commit/bf7a0d2bc2d774856f9dfecccc55ec6d54e0810d))

- Benchmark altering diagnostic plot levels and the adding decorrelated residual q-q plot
  ([`8565407`](https://github.com/sirocco-rt/speculate/commit/85654071201d586ec29b584419c6756ffa41298f))

- Enhance emulator configuration handling and fixed inclination support. Parameter playground in
  quick_fit style updates.
  ([`56d7c8a`](https://github.com/sirocco-rt/speculate/commit/56d7c8a30f524ec4ec9b6dca7650c5b0ce30a7e2))

- Physical parameters added to the inference playground
  ([`10023c2`](https://github.com/sirocco-rt/speculate/commit/10023c21626ce9e3ed82024d71c05a0ee7f37cc2))

- **benchmark**: Added 2sigms confidence intervals on emulated spectra and PCA Cumulative and
  Individual Component Reconstruction diagnostic plots
  ([`9525d7f`](https://github.com/sirocco-rt/speculate/commit/9525d7ff9aadef944bbd7bf81cee9dbbb65c97e0))

- **benchmark**: Introducing more scientific (publishable) tier 1 benchmark scores
  ([`6415b94`](https://github.com/sirocco-rt/speculate/commit/6415b940e9e1d5a68ae063a458137486cef8496d))

- **benchmark**: Starfish best fit mcmc spectrum plots and 99.7% coverage stats
  ([`e770bfc`](https://github.com/sirocco-rt/speculate/commit/e770bfc1e472843f7968de96e3ea0a27d4f027c8))

- **benchmark**: Tier 3 benchmark now find an observations best MLE/MCMC parameters, runs (the best)
  sirocco model, and overplots against the observation / emulation model
  ([`48b407d`](https://github.com/sirocco-rt/speculate/commit/48b407d86a8445a2b103fd4138d10a759f45af6d))

- **inference**: Added a test NLL button into the parameter playground
  ([`d93a7a6`](https://github.com/sirocco-rt/speculate/commit/d93a7a6bbf062821b4feb74ded395132733bcd8d))

- **inference**: Auto-calculating sensible prior centres depending upon data transformations used.
  ([`cd48649`](https://github.com/sirocco-rt/speculate/commit/cd48649a6c945f6ef16e1e71300ccefb315da15d))

- **inference**: Expose Chebyshev Continuum Tilt in Inference Pipeline
  ([`d76d80c`](https://github.com/sirocco-rt/speculate/commit/d76d80c94dc3aa5b4f8b9d4fff4a67fe1f5d5c92))

- **inference**: Restart runs on MLE to avoid chance of local minima.
  ([`665ff5e`](https://github.com/sirocco-rt/speculate/commit/665ff5e82b07ef7209ff89534038c88869f45db6))

- **quick**: Initial implimentation of lightweight emulator models with the Quick Fit tool
  ([`1a4e633`](https://github.com/sirocco-rt/speculate/commit/1a4e633bc9310551ac72969ba23e8eab839649ab))

- **training**: Added a load existing emulator dropdown to auto adjust dropdown settings to the
  correct inputs.
  ([`3822da3`](https://github.com/sirocco-rt/speculate/commit/3822da3a6324df5e26c5c2a768df9bc86f07fc93))

- **training**: Additional GP matern kernels for the emulator
  ([`f085e14`](https://github.com/sirocco-rt/speculate/commit/f085e14daf335a9d68d3f7744499dfb6c7170802))

- **training**: Emulator summary accordion in the training notebook, also fix to make sure only 1 GP
  diagnostic accordion
  ([`7a08770`](https://github.com/sirocco-rt/speculate/commit/7a087704a57f74335546825678c81234cb23655f))

### Performance Improvements

- Hard 5% sirocco test grid error changed to softer, study derived 2% plus a number of matplotlib to
  altair plot changes.
  ([`684baab`](https://github.com/sirocco-rt/speculate/commit/684baabec9273c9057c2618975b43faa90abb55d))

- **benchmark**: Improvement to the tier2 benchmarking against test data set, (Mid edit changes)
  ([`51574c2`](https://github.com/sirocco-rt/speculate/commit/51574c2513f248ce505b88822303c198d2603de1))

- **training**: Adding GP training for indiviual PCA components
  ([`2358946`](https://github.com/sirocco-rt/speculate/commit/2358946709981d5d53379b5debddcb6792938ea6))

- **training**: Adding in L-BFGS-B and CMA-ES optimiser.
  ([`54845ec`](https://github.com/sirocco-rt/speculate/commit/54845ec9b7f0f04db3dd43ea2929d575a678bf55))


## v0.1.1 (2026-03-20)

### Bug Fixes

- Versioning fixes and github actions to HF repair
  ([`b8d4db2`](https://github.com/sirocco-rt/speculate/commit/b8d4db290a52888b7a6c42acc82141c24b333a19))


## v0.1.0 (2026-03-20)

- Initial Release
