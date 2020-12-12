* Read bass xml files (wip)
* Try and only detect when one note is played, without detecting which one (wip)
** Maybe try and detect which guitar played it (lead, rhythm, bass)
* Train a denoising autoencoder on guitar-only midi (could be used as extra preprocessing step later on)
* Improve dataloader with several workers
* Refactor some code: training module is overloaded
* Imorove logs and saves
* try some boosting methods
* see if I can use some audio separation tool such as https://github.com/deezer/spleeter to filter guitar sounds
* add tests
