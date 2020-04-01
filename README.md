# rcms
An ICC color management library; heavily based on Little CMS.

Currently sparsely implemented and prone to crashing from a `todo!()`.

What this library will do:

- read and write ICC profiles
- create pipelines to transform colors between profiles

What this library will not do:

- efficiently transform big arrays of pixels
- handle pixel formats

To facilitate alternate implementations of color transforms (e.g. on the GPU), all pipeline internals are exposed in the API.
