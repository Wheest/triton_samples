<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Wheest/triton_samples">
    <img src="logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Triton Samples</h3>

  <p align="center">
    Some home-brewed Triton kernels, with varying degrees of optimisation
  </p>
</div>

As I've been exploring Triton, I've written some kernels.
I've put some of them here for public use, though note that many of these kernels just implement correctness, rather than using every optimisation under the sun.
I do lazy things like fix my block sizes, and may not explore the most sensible grid dimensions.
Also note that my implementations might make assumptions, e.g., we are always going to have a given dimension be a multiple of the block size.

Be sure to test this if this is relevant to you.

The code was tested using `triton==2.2.0` and `torch==2.1.2`.
This repo has no official connection to the Triton project.
