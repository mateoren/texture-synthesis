# High Resolution Texture Synthesis and Style Transfer

Implementation of a multi-scale texture synthesis algorithm able to generate high-resolution textures, later used for building a style transfer algorithm.

## Motivation

The work by Gatys et al. introduced to the world a method for extracting textures from a given input image. However, this method is limited to images of a certain resolution and it hardly works for high resolution images. This issue was addressed by considering a modified version of the algorithm that used Gaussian Pyramids, as is described in the paper *"High-Resolution Multi-Scale Neural Texture Synthesis"* by X. Snelgrove (https://wxs.ca/research/multiscale-neural-synthesis/).

## Examples

Original texture: ![alt text](https://raw.githubusercontent.com/mateoren/texture-synthesis/master/examples/origin.png)

Extracted texture: ![alt text](https://raw.githubusercontent.com/mateoren/texture-synthesis/master/examples/final_texture.png)

